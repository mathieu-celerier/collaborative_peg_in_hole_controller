// Exercises RemotePolicy against a live scripts/serve_policy.py, which is the only way to
// test framing: a mock would test this file's idea of the protocol against itself.
#include "RemotePolicy.h"
#include <chrono>
#include <cstdio>
#include <thread>

int main(int argc, char ** argv)
{
  const int port = argc > 1 ? std::atoi(argv[1]) : 5556;
  collab::RemotePolicy policy("127.0.0.1", port);
  std::printf("hello: %s for %s, %dx%d in, %dx%d out, skip %d\n", policy.name().c_str(), policy.partner().c_str(),
              policy.nObsSteps(), policy.stateDim(), policy.nActionSteps(), policy.actionDim(), policy.skip());

  // Two observations fill the history; the first request should then go out.
  Eigen::VectorXd x = Eigen::VectorXd::Zero(policy.stateDim());
  x(14) = 0.45;
  x(15) = 0.00;
  x(16) = 0.43; // a plausible tool position
  x(17) = 1.0;
  x(21) = 1.0; // rows of an identity-ish rotation
  policy.observe(x);
  std::printf("after 1 obs: request() = %d (history not full yet)\n", policy.request());
  policy.observe(x);
  std::printf("after 2 obs: request() = %d, inFlight = %d\n", policy.request(), policy.inFlight());

  // Poll the way the controller does: never blocking, once per 1 ms tick.
  const auto started = std::chrono::steady_clock::now();
  int ticks = 0;
  while(!policy.poll() && ticks < 5000)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ++ticks;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  std::printf("chunk arrived after %.1f ms (%d non-blocking polls), %zu actions of %d\n", ms, ticks, policy.remaining(),
              policy.actionDim());

  Eigen::VectorXd first = policy.next();
  std::printf("first action: [%.4f %.4f %.4f] (translation), starved = %d\n", first(0), first(1), first(2),
              policy.starved());

  // Drain it, then check the hold-and-flag behaviour past the end.
  while(policy.remaining())
  {
    policy.next();
  }
  Eigen::VectorXd held = policy.next();
  std::printf("past the end: starved = %d, holds last = %d\n", policy.starved(), (held - policy.next()).norm() < 1e-12);

  // A reset must make an in-flight reply stale rather than usable.
  policy.observe(x);
  policy.observe(x);
  policy.request();
  policy.reset();
  ticks = 0;
  while(policy.discarded() == 0 && ticks < 5000)
  {
    policy.poll();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ++ticks;
  }
  std::printf("after reset: discarded = %d (a stale reply must not be applied)\n", policy.discarded());
  return 0;
}
