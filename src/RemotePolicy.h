#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace collab
{

/** A chunked policy served by another process, over loopback TCP.
 *
 * Not a `PolicyBackend`, for the same reason `IntentReplay` is not: that interface is one
 * observation in, one action out, synchronously. This is none of those things. It carries an
 * observation *history*, it answers with a *sequence* of actions, and the answer arrives some
 * tens of milliseconds after the question -- which is why it exists at all.
 *
 * ### The latency is the design
 *
 * A DDIM-8 diffusion policy costs 22.6 ms on the GPU (88 ms while a training run shares it),
 * against a 1 ms control tick. Inference in the control thread is therefore not slow, it is
 * fatal: 22 ms of it is 22 missed ticks and a dead QP. See doc/serving-latency.md in the
 * learning repo for the measurements.
 *
 * What makes it servable is that the policy answers with `n_action_steps` actions at once. The
 * controller executes the chunk it has while the next one is computed, so the inference has a
 * whole chunk of episode time to finish in -- 240 ms at this policy's rate, against 88 ms of
 * work. Nothing here ever blocks: `request` and `poll` both return immediately, and a chunk
 * that has not arrived yet simply means the previous one is still running.
 *
 * ### What happens when the answer is late
 *
 * The last action is held and `starved()` goes true, so the log says so. That is the honest
 * degradation -- it is what a partner whose hand stops moving does -- but it is *not*
 * acceptable in a result: a run with starved ticks measured the server's latency as much as
 * the policy's predictions. Check the log entry before believing a success rate.
 *
 * ### Raw sockets, not nanomsg
 *
 * mc_rtc already links nanomsg, so the plan assumed it. But the Python side has no nanomsg
 * binding in this environment, and the exchange is one fixed-size request and one fixed-size
 * reply. A 16-byte header and a block of doubles needs no message library at either end, and
 * it leaves the non-blocking behaviour entirely under this class's control.
 */
class RemotePolicy
{
public:
  /** Connect to the server. Throws if it is not there: a policy run whose policy is absent
   * must fail while the config is read, not four seconds into the first episode. */
  RemotePolicy(const std::string & host, int port, int nObsSteps, int stateDim);
  ~RemotePolicy();

  RemotePolicy(const RemotePolicy &) = delete;
  RemotePolicy & operator=(const RemotePolicy &) = delete;

  /** Episode boundary: drop the history, the chunk, and any request in flight.
   *
   * The in-flight request matters most. Its reply was computed from the *previous* episode's
   * observations, and applying it here would be a partner acting on where the arm used to be.
   * The sequence number is bumped so that reply is recognised as stale and discarded.
   */
  void reset();

  /** Push one observation, at the policy rate. Keeps the last `nObsSteps`. */
  void observe(const Eigen::VectorXd & x);

  /** Non-blocking. Installs a chunk if a whole reply has arrived. Returns whether one did. */
  bool poll();

  /** Non-blocking. Sends the history if it is full and nothing is in flight. Returns whether
   * a request went out. */
  bool request();

  /** Whether the chunk is low enough that the next request should go out now.
   *
   * Not "empty": a request sent once the chunk runs out arrives a whole inference late, and
   * every tick in between is starved. The refill point is what hides the latency.
   */
  bool needsRefill() const noexcept;

  /** The action for this policy step. Pops the chunk, holding the last one when it is empty. */
  Eigen::VectorXd next();

  bool inFlight() const noexcept
  {
    return inFlight_;
  }
  bool starved() const noexcept
  {
    return starved_;
  }
  std::size_t remaining() const noexcept
  {
    return chunk_.size();
  }
  /** Replies discarded for arriving after their episode ended, or out of order. */
  int discarded() const noexcept
  {
    return discarded_;
  }
  int actionDim() const noexcept
  {
    return actionDim_;
  }
  bool ready() const noexcept
  {
    return haveLastAction_ || !chunk_.empty();
  }

  /** Actions left before a request must go out. Configurable because the right value depends
   * on the server's latency, which depends on the machine. */
  void refillAt(std::size_t remaining) noexcept
  {
    refillAt_ = remaining;
  }

private:
  int fd_ = -1;
  int nObsSteps_ = 1;
  int stateDim_ = 0;
  int actionDim_ = 0;

  std::deque<Eigen::VectorXd> history_;
  std::deque<Eigen::VectorXd> chunk_;
  Eigen::VectorXd lastAction_;
  bool haveLastAction_ = false;

  std::uint32_t seq_ = 0;
  std::uint32_t inFlightSeq_ = 0;
  bool inFlight_ = false;
  bool starved_ = false;
  int discarded_ = 0;
  std::size_t refillAt_ = 4;

  //: The reply as it accumulates. A reply arrives in whatever pieces the kernel chose, and a
  //: non-blocking reader that could not resume mid-message would never finish one.
  std::vector<char> pending_;
};

} // namespace collab
