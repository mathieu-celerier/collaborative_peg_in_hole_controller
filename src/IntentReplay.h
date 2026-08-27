#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

namespace collab
{

/** A partner's recorded intent, played back by episode tick.
 *
 * This is not a policy and deliberately not a `PolicyBackend`: it observes nothing, and a
 * state-feedback signature would only invite the mistake of thinking it does. It exists to
 * answer one question that no policy result can answer on its own -- **is the path from an
 * intent to the arm moving correct at all?**
 *
 * The intent path has three stages: a predicted point, the replaced partner's impedance
 * turning that point into a force, and the force injected as `CompliantEndEffectorTask`'s
 * `refAccel_` rather than as a physical wrench in the simulator. When a policy diverges in
 * closed loop, any of the three could be responsible, and the argument cannot be settled by
 * staring at it. Replaying the *recorded* reference of a demonstration that is known to have
 * succeeded holds the first stage fixed at ground truth: if the dyad still inserts, stages
 * two and three are sound and the policy's predictions are the problem; if it does not, the
 * plumbing is, and every policy number measured through it is meaningless.
 *
 * ### Rebased, because the plan the reference came from was
 *
 * `SimHumanPartner` rebases each partner's plan onto its owner's hand at the start of every
 * episode, so a recorded reference already carries one episode's start pose inside it. Played
 * back verbatim into an episode that started somewhere else -- a different noise draw, a
 * different placement -- it would aim at the old episode's peg. The file therefore stores the
 * hand position the recording started from, and the controller shifts the whole trajectory by
 * the difference. Replaying into the very episode it came from makes that shift zero, which
 * is the case to prefer; the offset is logged so a large one cannot pass unnoticed.
 *
 * ### Holding past the end
 *
 * The reference stops when the recorded episode did, and a replay episode may run longer.
 * The last sample is held, which is what the demonstrating partner itself does: the min-jerk
 * plan ends with a dwell at the insertion, not with the hand letting go.
 */
class IntentReplay
{
public:
  /** Load the JSON written by the learning repo's `scripts/export_replay.py`.
   *
   * `timeStep` is the controller's, and decides how many control ticks each recorded sample
   * covers. Throws if the file is not a replay, or if its rate is not a whole multiple of the
   * control rate -- resampling silently would put the replay out of phase with the episode it
   * is being compared against, which is the one thing this must not do.
   */
  IntentReplay(const std::string & path, double timeStep);

  /** The intent for a tick of the episode, rebased. Clamped at both ends. */
  Eigen::Vector3d at(int episodeTick) const;

  /** Where the recording's hand started. `at()` shifts by (anchor - origin). */
  const Eigen::Vector3d & origin() const noexcept
  {
    return origin_;
  }

  /** Move the trajectory onto the hand this episode actually started from. */
  void anchor(const Eigen::Vector3d & position)
  {
    anchor_ = position;
  }
  const Eigen::Vector3d & anchor() const noexcept
  {
    return anchor_;
  }
  Eigen::Vector3d offset() const
  {
    return anchor_ - origin_;
  }

  /** Whether this tick is past the recording, and so holding the last sample. */
  bool exhausted(int episodeTick) const noexcept;

  /** The partner whose reference this is. Decides who stops pushing, as for a policy. */
  const std::string & partner() const noexcept
  {
    return partner_;
  }
  double rate() const noexcept
  {
    return rate_;
  }
  const std::string & source() const noexcept
  {
    return source_;
  }
  std::size_t size() const noexcept
  {
    return reference_.size();
  }

private:
  std::string partner_;
  std::string source_;
  double rate_ = 0.0;
  //: Control ticks per recorded sample.
  int stride_ = 1;
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d anchor_ = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> reference_;
};

} // namespace collab
