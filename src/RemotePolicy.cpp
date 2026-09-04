#include "RemotePolicy.h"

#include <mc_rtc/Configuration.h>
#include <mc_rtc/logging.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>

namespace collab
{

namespace
{

constexpr char MAGIC[4] = {'C', 'P', 'I', 'H'};
constexpr std::uint32_t VERSION = 1;
//: magic(4) + version(4) + seq(4) + rows(2) + cols(2)
constexpr std::size_t HEADER_SIZE = 16;

void writeHeader(char * into, std::uint32_t seq, std::uint16_t rows, std::uint16_t cols)
{
  std::memcpy(into, MAGIC, 4);
  const std::uint32_t version = VERSION;
  std::memcpy(into + 4, &version, 4);
  std::memcpy(into + 8, &seq, 4);
  std::memcpy(into + 12, &rows, 2);
  std::memcpy(into + 14, &cols, 2);
}

} // namespace

RemotePolicy::RemotePolicy(const std::string & host, int port)
{
  fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if(fd_ < 0)
  {
    mc_rtc::log::error_and_throw("[RemotePolicy] socket(): {}", std::strerror(errno));
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if(::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1)
  {
    ::close(fd_);
    fd_ = -1;
    mc_rtc::log::error_and_throw("[RemotePolicy] '{}' is not a dotted-quad address", host);
  }

  // Connect while still blocking: this happens once, at load time, and a policy run whose
  // server is not up must fail here rather than as an arm that never moves.
  if(::connect(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0)
  {
    const int saved = errno;
    ::close(fd_);
    fd_ = -1;
    mc_rtc::log::error_and_throw("[RemotePolicy] cannot reach the policy server at {}:{} ({}). Start it first:\n"
                                 "  scripts/serve_policy.py <checkpoint> --port {}",
                                 host, port, std::strerror(saved), port);
  }

  // Nagle would hold a small reply for ~40 ms waiting for more to send. That is twice the
  // inference it is carrying, and it would be invisible -- the policy would simply look slow.
  int one = 1;
  ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  // Read the hello while still blocking. It arrives immediately, it happens once, and every
  // field below has to be known before the first observation can even be assembled.
  readHello(host, port);

  // Only now: every read and write after this assumes it returns immediately.
  ::fcntl(fd_, F_SETFL, ::fcntl(fd_, F_GETFL, 0) | O_NONBLOCK);
}

void RemotePolicy::readHello(const std::string & host, int port)
{
  char header[HEADER_SIZE];
  std::size_t got = 0;
  while(got < HEADER_SIZE)
  {
    const auto n = ::recv(fd_, header + got, HEADER_SIZE - got, 0);
    if(n <= 0)
    {
      mc_rtc::log::error_and_throw("[RemotePolicy] {}:{} accepted the connection but sent no hello. Is it really "
                                   "scripts/serve_policy.py?",
                                   host, port);
    }
    got += static_cast<std::size_t>(n);
  }
  std::uint32_t seq = 0;
  std::uint16_t rows = 0;
  std::uint16_t cols = 0;
  std::memcpy(&seq, header + 8, 4);
  std::memcpy(&rows, header + 12, 2);
  std::memcpy(&cols, header + 14, 2);
  if(std::memcmp(header, MAGIC, 4) != 0 || seq != 0 || rows != 1 || cols == 0)
  {
    mc_rtc::log::error_and_throw("[RemotePolicy] the first frame from {}:{} is not a hello", host, port);
  }

  std::string json(cols, '\0');
  got = 0;
  while(got < cols)
  {
    const auto n = ::recv(fd_, &json[got], cols - got, 0);
    if(n <= 0)
    {
      mc_rtc::log::error_and_throw("[RemotePolicy] the hello from {}:{} was truncated", host, port);
    }
    got += static_cast<std::size_t>(n);
  }

  auto conf = mc_rtc::Configuration::fromData(json);
  conf("n_obs_steps", nObsSteps_);
  conf("state_dim", stateDim_);
  conf("n_action_steps", nActionSteps_);
  conf("action_dim", actionDim_);
  conf("skip", skip_);
  conf("partner", partner_);
  conf("policy", name_);
  if(nObsSteps_ < 1 || stateDim_ < 1 || nActionSteps_ < 1 || actionDim_ < 3)
  {
    mc_rtc::log::error_and_throw("[RemotePolicy] the server describes a nonsense policy: {}x{} in, {}x{} out",
                                 nObsSteps_, stateDim_, nActionSteps_, actionDim_);
  }
  if(partner_.empty())
  {
    mc_rtc::log::error_and_throw("[RemotePolicy] the server names no partner, so nothing decides who stops pushing. "
                                 "Start it with --partner.");
  }
  // Half the chunk, so the request goes out with an inference's worth of actions still to run.
  refillAt_ = static_cast<std::size_t>(nActionSteps_) / 2;

  mc_rtc::log::info("[RemotePolicy] {}:{} serves '{}' for partner '{}': {}x{} in, {}x{} out, skip {} (decimation {})",
                    host, port, name_, partner_, nObsSteps_, stateDim_, nActionSteps_, actionDim_, skip_, skip_ * 10);
}

RemotePolicy::~RemotePolicy()
{
  if(fd_ >= 0)
  {
    ::close(fd_);
  }
}

void RemotePolicy::reset()
{
  history_.clear();
  chunk_.clear();
  haveLastAction_ = false;
  starved_ = false;
  pending_.clear();
  // Bumped whether or not something is in flight, so the reply to the previous episode's last
  // request can never be mistaken for an answer to this episode's first.
  ++seq_;
  inFlight_ = false;
}

void RemotePolicy::observe(const Eigen::VectorXd & x)
{
  if(x.size() != stateDim_)
  {
    mc_rtc::log::error_and_throw("[RemotePolicy] observation is {} long, the policy wants {}", x.size(), stateDim_);
  }
  history_.push_back(x);
  while(static_cast<int>(history_.size()) > nObsSteps_)
  {
    history_.pop_front();
  }
}

bool RemotePolicy::request()
{
  if(inFlight_ || static_cast<int>(history_.size()) < nObsSteps_)
  {
    return false;
  }

  const std::size_t values = static_cast<std::size_t>(nObsSteps_) * static_cast<std::size_t>(stateDim_);
  std::vector<char> frame(HEADER_SIZE + values * sizeof(double));
  ++seq_;
  writeHeader(frame.data(), seq_, static_cast<std::uint16_t>(nObsSteps_), static_cast<std::uint16_t>(stateDim_));
  auto * out = reinterpret_cast<double *>(frame.data() + HEADER_SIZE);
  for(const auto & row : history_)
  {
    for(Eigen::Index i = 0; i < row.size(); ++i)
    {
      *out++ = row(i);
    }
  }

  // The request is ~500 bytes, far below any socket buffer, so a partial write here would
  // mean the server has stopped reading entirely -- which is a dead server, not backpressure.
  std::size_t sent = 0;
  while(sent < frame.size())
  {
    const auto n = ::send(fd_, frame.data() + sent, frame.size() - sent, MSG_NOSIGNAL);
    if(n > 0)
    {
      sent += static_cast<std::size_t>(n);
      continue;
    }
    if(n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    {
      continue;
    }
    mc_rtc::log::error_and_throw("[RemotePolicy] the policy server went away mid-request ({})", std::strerror(errno));
  }

  inFlightSeq_ = seq_;
  inFlight_ = true;
  return true;
}

bool RemotePolicy::poll()
{
  if(fd_ < 0)
  {
    return false;
  }

  // Read whatever is there, appending to what previous ticks collected. Two sizes are known
  // in advance -- the header, then the payload it announces -- so this never has to guess.
  for(;;)
  {
    const std::size_t wanted =
        pending_.size() < HEADER_SIZE
            ? HEADER_SIZE
            : HEADER_SIZE
                  + static_cast<std::size_t>(*reinterpret_cast<const std::uint16_t *>(pending_.data() + 12))
                        * static_cast<std::size_t>(*reinterpret_cast<const std::uint16_t *>(pending_.data() + 14))
                        * sizeof(double);
    if(pending_.size() >= wanted && pending_.size() >= HEADER_SIZE)
    {
      break;
    }

    const std::size_t missing = wanted - pending_.size();
    const std::size_t at = pending_.size();
    pending_.resize(wanted);
    const auto n = ::recv(fd_, pending_.data() + at, missing, 0);
    if(n > 0)
    {
      pending_.resize(at + static_cast<std::size_t>(n));
      continue;
    }
    pending_.resize(at);
    if(n == 0)
    {
      mc_rtc::log::error_and_throw("[RemotePolicy] the policy server closed the connection");
    }
    if(errno == EAGAIN || errno == EWOULDBLOCK)
    {
      return false;
    } // nothing more yet, try next tick
    mc_rtc::log::error_and_throw("[RemotePolicy] recv(): {}", std::strerror(errno));
  }

  if(std::memcmp(pending_.data(), MAGIC, 4) != 0)
  {
    mc_rtc::log::error_and_throw("[RemotePolicy] the server is not speaking this protocol");
  }
  std::uint32_t seq = 0;
  std::uint16_t rows = 0;
  std::uint16_t cols = 0;
  std::memcpy(&seq, pending_.data() + 8, 4);
  std::memcpy(&rows, pending_.data() + 12, 2);
  std::memcpy(&cols, pending_.data() + 14, 2);

  if(rows == 0)
  {
    mc_rtc::log::error_and_throw("[RemotePolicy] the server refused a {}x{} observation. Its checkpoint expects a "
                                 "different shape than this controller assembles.",
                                 nObsSteps_, stateDim_);
  }

  const auto * values = reinterpret_cast<const double *>(pending_.data() + HEADER_SIZE);
  const bool stale = seq != inFlightSeq_ || !inFlight_;
  if(stale)
  {
    // An answer to a question this episode did not ask. Dropped rather than applied: it was
    // computed from observations that are no longer true.
    ++discarded_;
  }
  else if(!std::all_of(values, values + static_cast<std::size_t>(rows) * cols,
                       [](double v) { return std::isfinite(v); }))
  {
    // A NaN in the chunk is not a bad prediction, it is no prediction: it would reach the
    // impedance as an infinite spring stretch and empty the QP's constraint set on the tick
    // it arrives. Dropped like a stale reply, so `next()` holds the last action and the
    // episode carries on with a partner that stopped moving -- which is a scoreable failure
    // rather than an erased episode. `inFlight_` is cleared so the next step asks again.
    ++rejected_;
    if(rejected_ == 1)
    {
      mc_rtc::log::warning("[RemotePolicy] dropped a chunk carrying a non-finite action; holding the last one");
    }
    inFlight_ = false;
  }
  else
  {
    chunk_.clear();
    for(std::uint16_t r = 0; r < rows; ++r)
    {
      Eigen::VectorXd action(cols);
      for(std::uint16_t c = 0; c < cols; ++c)
      {
        action(c) = values[r * cols + c];
      }
      chunk_.push_back(std::move(action));
    }
    actionDim_ = cols;
    inFlight_ = false;
    starved_ = false;
  }
  pending_.clear();
  return !stale;
}

bool RemotePolicy::needsRefill() const noexcept
{
  return !inFlight_ && chunk_.size() <= refillAt_;
}

Eigen::VectorXd RemotePolicy::next()
{
  if(!chunk_.empty())
  {
    lastAction_ = chunk_.front();
    chunk_.pop_front();
    haveLastAction_ = true;
    starved_ = false;
    return lastAction_;
  }
  // Out of actions. Holding the last one is what a hand that stopped moving does, and it is
  // the only safe thing to do -- but the flag is what stops a starved run being read as a
  // result. See the class docs.
  starved_ = true;
  if(haveLastAction_)
  {
    return lastAction_;
  }
  return Eigen::VectorXd::Zero(actionDim_ > 0 ? actionDim_ : 3);
}

} // namespace collab
