#include "SimHumanPartner.h"

#include "../../src/CollabPegInHoleController.h"

#include <mc_control/GlobalPluginMacros.h>
#include <mc_rtc/gui.h>
#include <mc_rtc/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace collab
{

namespace
{

/** Quintic minimum-jerk interpolation on [0, 1], with zero velocity and acceleration at
 * both ends -- the polynomial that minimises integrated squared jerk, and the reason human
 * point-to-point reaches show a symmetric bell-shaped speed profile (Flash & Hogan 1985). */
inline void minJerkUnit(double tau, double & s, double & ds)
{
  const double t2 = tau * tau;
  const double t3 = t2 * tau;
  const double t4 = t3 * tau;
  const double t5 = t4 * tau;
  s = 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
  ds = 30.0 * t2 - 60.0 * t3 + 30.0 * t4;
}

} // namespace

void SimHumanPartner::referenceAt(const Partner & partner, double t, Eigen::Vector3d & position, Eigen::Vector3d & velocity)
{
  for(const auto & seg : partner.segments)
  {
    if(t < seg.start + seg.duration)
    {
      if(t <= seg.start)
      {
        // Before the segment opens -- either the very start, or a dwell between segments.
        // Holding still is also what a human waiting for their partner does.
        position = seg.from;
        velocity.setZero();
        return;
      }
      double s = 0.0;
      double ds = 0.0;
      minJerkUnit((t - seg.start) / seg.duration, s, ds);
      const Eigen::Vector3d delta = seg.to - seg.from;
      position = seg.from + s * delta;
      velocity = (ds / seg.duration) * delta;
      return;
    }
  }
  // Past the last segment: hold the final target. An episode ends on the task condition,
  // not on the trajectory running out.
  position = partner.finalTarget;
  velocity.setZero();
}

double SimHumanPartner::phaseFromState(Partner & partner, const Eigen::Vector3d & handPos, double dt)
{
  if(partner.segments.empty())
  {
    return 0.0;
  }

  // Nearest point on the polyline. Segments are straight lines, so this is exact rather than
  // a sampling: the closest point of a segment is the clamped scalar projection onto it.
  double bestDistance = std::numeric_limits<double>::max();
  double bestTime = partner.segments.front().start;
  for(const auto & seg : partner.segments)
  {
    const Eigen::Vector3d delta = seg.to - seg.from;
    const double length2 = delta.squaredNorm();
    // A dwell: zero length, so every point of it projects to the same place. Its *time* still
    // matters -- holding still is part of the plan -- but it cannot win a distance contest
    // against the leg it sits between, so leave it to the neighbouring segments.
    const double u = length2 > 1e-12 ? std::clamp((handPos - seg.from).dot(delta) / length2, 0.0, 1.0) : 0.0;
    const double distance = (handPos - (seg.from + u * delta)).norm();
    if(distance >= bestDistance)
    {
      continue;
    }
    bestDistance = distance;
    // Invert the min-jerk profile: s(tau) is strictly increasing on [0,1], so bisection is
    // both correct and cheap. Solved rather than approximated because the profile spends most
    // of its time in the middle of the segment -- a linear guess would put the phase a long
    // way out near either end, which is exactly where the corner of the L is.
    double lo = 0.0;
    double hi = 1.0;
    for(int i = 0; i < 30; ++i)
    {
      const double mid = 0.5 * (lo + hi);
      double sMid = 0.0;
      double dsMid = 0.0;
      minJerkUnit(mid, sMid, dsMid);
      (sMid < u ? lo : hi) = mid;
    }
    bestTime = seg.start + 0.5 * (lo + hi) * seg.duration;
  }

  if(partner.labelClock < 0.0)
  {
    partner.labelClock = bestTime;
    return bestTime;
  }
  // Monotone, and rate-limited to what the plan itself covers in a tick times a slack factor.
  // The trajectory is an L: near the corner the transit and the descent pass close together in
  // space, and an unguarded nearest-point search snaps between them. A person does not
  // un-execute a plan, and does not skip half of one either.
  const double ceiling = partner.labelClock + std::max(dt, 0.0) * partner.feedback.maxAdvanceRate;
  partner.labelClock = std::clamp(bestTime, partner.labelClock, ceiling);
  return partner.labelClock;
}

void SimHumanPartner::aimedPoint(const mc_rbdyn::Robot & robot, const Partner & partner,
                                 Eigen::Vector3d & position, Eigen::Vector3d & velocity)
{
  const sva::PTransformd & X_0_b = robot.bodyPosW(partner.referenceBody);
  // sva stores the rotation as world->body, so the transpose maps a body-frame offset back
  // into the world.
  const Eigen::Vector3d offsetW = X_0_b.rotation().transpose() * partner.referencePoint;
  position = X_0_b.translation() + offsetW;
  const sva::MotionVecd & V_0_b = robot.bodyVelW(partner.referenceBody);
  // Velocity of the offset point, not of the body origin: they differ as soon as the body
  // rotates, and the hole sits well away from the wrist axis.
  velocity = V_0_b.linear() + V_0_b.angular().cross(offsetW);
}

void SimHumanPartner::rebase(Partner & partner, const Eigen::Vector3d & origin,
                             const Eigen::Vector3d & holeOrigin)
{
  if(partner.segments.empty())
  {
    return;
  }
  if(!partner.relative)
  {
    partner.segments.front().from = origin;
    return;
  }
  // The waypoints are absolute targets for the *hole*: they end on the peg, which is a fixed
  // world point. Move them into this hand's frame by the hand-to-hole offset measured now, so
  // the plan stays registered to where the hole really is. Rebuild from the configured copy so
  // that the second episode does not shift an already-shifted one.
  const Eigen::Vector3d delta = origin - holeOrigin;
  for(size_t i = 0; i < partner.segments.size(); ++i)
  {
    partner.segments[i].from = partner.episodeSegments[i].from + delta;
    partner.segments[i].to = partner.episodeSegments[i].to + delta;
  }
  partner.finalTarget = partner.episodeFinalTarget + delta;
  // The configured start is where the generator *assumed* the hole would be. It is wrong by
  // whatever the episode's actual pose is, so the first leg starts from the hand instead.
  partner.segments.front().from = origin;
}

const char * SimHumanPartner::phaseName(Phase phase)
{
  switch(phase)
  {
    case Phase::Idle:
      return "idle";
    case Phase::GoingCompliant:
      return "going compliant";
    case Phase::Settling:
      return "settling";
    case Phase::Recording:
      return "recording";
    case Phase::GoingHome:
      return "going home";
  }
  return "unknown";
}

void SimHumanPartner::parsePartners(const mc_rtc::Configuration & config)
{
  partners_.clear();
  std::vector<mc_rtc::Configuration> entries = config("partners", std::vector<mc_rtc::Configuration>{});
  if(entries.empty())
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[SimHumanPartner] no partners configured. This plugin exists only to apply them, so an "
        "empty list is a configuration mistake, not a valid no-op. Generate the config with "
        "scripts/make_sim_config.py in the learning repo.");
  }
  for(const auto & entry : entries)
  {
    Partner p;
    p.name = entry("name", std::string("partner"));
    if(!entry.has("body"))
    {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[SimHumanPartner] partner '{}' has no 'body'. Which body it pushes on decides whether "
          "the F/T sensor observes it, so there is no sensible default.",
          p.name);
    }
    p.body = entry("body").operator std::string();
    entry("local_point", p.localPoint);
    // Default to aiming the very point that is pushed, which is right for person B (they
    // hold the piece) and wrong for person A -- so the generated config always states it.
    p.referenceBody = p.body;
    p.referencePoint = p.localPoint;
    entry("reference_body", p.referenceBody);
    entry("reference_point", p.referencePoint);
    entry("relative", p.relative);
    if(p.referenceBody != p.body || !p.referencePoint.isApprox(p.localPoint))
    {
      // Not a style preference: see the Partner docs. An impedance anchored on a body other
      // than the one it pushes leaves the arm's internal motion unopposed, and the QP dies a
      // second or so after the compliance stiffness ramps down.
      mc_rtc::log::warning(
          "[SimHumanPartner] partner '{}' pushes '{}' but aims '{}'. An impedance that does not "
          "close on the point it acts through cannot see the motion it causes; expect the arm's "
          "internal configuration to run away once the end-effector stiffness ramps to zero. Use "
          "'relative: true' with the partner's own body instead.",
          p.name, p.body, p.referenceBody);
    }
    entry("stiffness", p.stiffness);
    entry("damping", p.damping);
    entry("max_force", p.maxForce);
    entry("delay", p.delay);
    entry("start_delay", p.startDelay);

    if((p.stiffness.array() < 0).any() || (p.damping.array() < 0).any())
    {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[SimHumanPartner] partner '{}' has a negative gain; that is a partner that pushes away "
          "from its own target.",
          p.name);
    }

    if(entry.has("feedback"))
    {
      auto f = entry("feedback");
      f("lookahead_time", p.feedback.lookaheadTime);
      f("max_advance_rate", p.feedback.maxAdvanceRate);
      if(p.feedback.lookaheadTime < 0.0)
      {
        mc_rtc::log::error_and_throw<std::runtime_error>(
            "[SimHumanPartner] partner '{}' has a negative feedback lookahead_time. Use 0 to "
            "read the plan from the clock, which is the behaviour without this block at all.",
            p.name);
      }
    }
    if(entry.has("label_stiffness"))
    {
      p.labelStiffness = entry("label_stiffness");
      if((p.labelStiffness.array() < 0).any())
      {
        mc_rtc::log::error_and_throw<std::runtime_error>(
            "[SimHumanPartner] partner '{}' has a negative label_stiffness. It is the stiffness "
            "this partner's *resistance* is measured with -- the gains it would have had if it "
            "were acting -- not a second spring.",
            p.name);
      }
    }

    if(entry.has("search"))
    {
      auto s = entry("search");
      s("amplitude", p.search.amplitude);
      s("frequency", p.search.frequency);
      s("contact_force", p.search.contactForce);
      s("stall_time", p.search.stallTime);
      s("hold_time", p.search.holdTime);
      s("stall_motion", p.search.stallMotion);
      s("spiral_time", p.search.spiralTime);
      s("ramp_time", p.search.rampTime);
      if(p.search.rampTime < 0.0)
      {
        mc_rtc::log::error_and_throw<std::runtime_error>(
            "[SimHumanPartner] partner '{}' has a negative search ramp_time. Use 0 to keep the "
            "rate coupled to frequency; a negative ramp inverts the envelope.",
            p.name);
      }
            if(p.search.amplitude < 0.0)
      {
        mc_rtc::log::error_and_throw<std::runtime_error>(
            "[SimHumanPartner] partner '{}' has a negative search amplitude. Use 0 to disable "
            "the search; a negative radius is a circle traced backwards, which is the same "
            "circle and almost certainly not what was meant.",
            p.name);
      }
      if(p.search.enabled() && p.search.frequency <= 0.0)
      {
        mc_rtc::log::error_and_throw<std::runtime_error>(
            "[SimHumanPartner] partner '{}' has a search amplitude but a non-positive frequency.",
            p.name);
      }
    }

    Eigen::Vector3d cursor = Eigen::Vector3d::Zero();
    entry("start", cursor);
    double clock = p.startDelay;
    std::vector<mc_rtc::Configuration> waypoints = entry("waypoints", std::vector<mc_rtc::Configuration>{});
    if(waypoints.empty())
    {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[SimHumanPartner] partner '{}' has no waypoints, so it has no intent at all.", p.name);
    }
    for(const auto & wp : waypoints)
    {
      Segment seg;
      seg.start = clock;
      seg.duration = wp("duration", 1.0);
      if(seg.duration <= 0.0)
      {
        mc_rtc::log::error_and_throw<std::runtime_error>(
            "[SimHumanPartner] partner '{}' has a waypoint of non-positive duration.", p.name);
      }
      seg.from = cursor;
      wp("position", seg.to);
      clock += seg.duration + wp("dwell", 0.0);
      cursor = seg.to;
      p.segments.push_back(seg);
    }
    p.finalTarget = cursor;
    p.configuredSegments = p.segments;
    p.configuredStart = p.segments.front().from;
    p.configuredFinalTarget = p.finalTarget;
    p.configuredStiffness = p.stiffness;
    p.configuredDamping = p.damping;
    p.configuredDelay = p.delay;
    p.configuredStartDelay = p.startDelay;
    // With no noise drawn yet -- episodes driven from the GUI never draw one -- the episode
    // plan has to be the configured plan, or `rebase` would register onto an empty one.
    p.episodeSegments = p.configuredSegments;
    p.episodeFinalTarget = p.configuredFinalTarget;
    partners_.push_back(p);

    mc_rtc::log::info("[SimHumanPartner] '{}' on body '{}': kp {:.1f} N/m, kd {:.1f} N.s/m, delay {:.0f} ms",
                      p.name, p.body, p.stiffness.x(), p.damping.x(), p.delay * 1000.0);
    if(p.search.enabled())
    {
      mc_rtc::log::info("[SimHumanPartner] '{}' searches when stuck: {:.1f} mm at {:.1f} Hz once it "
                        "feels {:.1f} N and the piece has not moved {:.1f} mm in {:.1f} s",
                        p.name, p.search.amplitude * 1000.0, p.search.frequency, p.search.contactForce,
                        p.search.stallMotion * 1000.0, p.search.stallTime);
      if(p.search.holds())
      {
        // Worth a line of its own: it is the difference between the configured amplitude and
        // the amplitude that actually gets drawn.
        const double rampSeconds =
            p.search.rampsExplicitly()
                ? p.search.rampTime
                : 1.0 / (std::max(p.search.frequency, 1e-6) * (p.search.spiral() ? 4.0 : 1.0));
        mc_rtc::log::info("[SimHumanPartner] '{}' holds the search {:.1f} s across a stall dropout "
                          "(the envelope needs {:.2f} s of continuous searching to reach {:.1f} mm, "
                          "and a full turn takes {:.2f} s, so a constant radius needs a burst "
                          "longer than {:.2f} s)",
                          p.name, p.search.holdTime, rampSeconds, p.search.amplitude * 1000.0,
                          1.0 / std::max(p.search.frequency, 1e-6), 2.0 * rampSeconds);
      }
    }
  }
}

void SimHumanPartner::parseNoise(const mc_rtc::Configuration & config)
{
  if(!config.has("noise"))
  {
    return;
  }
  auto entry = config("noise");
  unsigned int seed = noise_.seed;
  entry("seed", seed);
  noise_.seed = seed;
  entry("target_sigma", noise_.targetSigma);
  entry("gain_sigma", noise_.gainSigma);
  entry("timing_sigma", noise_.timingSigma);
  entry("delay_sigma", noise_.delaySigma);
  entry("start_sigma", noise_.startSigma);
  if(noise_.targetSigma < 0.0 || noise_.gainSigma < 0.0 || noise_.timingSigma < 0.0
     || noise_.delaySigma < 0.0 || noise_.startSigma < 0.0)
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[SimHumanPartner] a noise sigma is negative. These are standard deviations, not offsets.");
  }
  rng_.seed(noise_.seed);
  if(noise_.active())
  {
    mc_rtc::log::info("[SimHumanPartner] per-episode noise, seed {}: target {:.1f} mm, gain {:.0f}%, "
                      "timing {:.0f}%, delay {:.0f} ms, registration {:.1f} mm",
                      noise_.seed, noise_.targetSigma * 1000.0, noise_.gainSigma * 100.0,
                      noise_.timingSigma * 100.0, noise_.delaySigma * 1000.0, noise_.startSigma * 1000.0);
  }
}

void SimHumanPartner::drawEpisodeNoise()
{
  const bool active = noise_.active();
  std::normal_distribution<double> unit(0.0, 1.0);
  registrationOffset_.setZero();
  if(active && noise_.startSigma > 0.0)
  {
    for(int axis = 0; axis < 3; ++axis)
    {
      registrationOffset_[axis] = noise_.startSigma * unit(rng_);
    }
  }
  for(auto & p : partners_)
  {
    p.targetOffset.setZero();
    p.gainScale = 1.0;
    p.stiffness = p.configuredStiffness;
    p.damping = p.configuredDamping;
    p.delay = p.configuredDelay;
    p.startDelay = p.configuredStartDelay;
    p.episodeSegments = p.configuredSegments;
    p.episodeFinalTarget = p.configuredFinalTarget;
    if(!active)
    {
      continue;
    }
    if(noise_.gainSigma > 0.0)
    {
      // One scale for the partner, not one per axis: the gains it perturbs are isotropic and
      // come out of a single scalar game, so a per-axis draw would invent a directional
      // preference the model does not have. Floored well above zero -- a partner with no
      // stiffness is not a soft partner, it is an absent one.
      p.gainScale = std::max(0.2, 1.0 + noise_.gainSigma * unit(rng_));
      p.stiffness = p.configuredStiffness * p.gainScale;
      p.damping = p.configuredDamping * p.gainScale;
    }
    if(noise_.delaySigma > 0.0)
    {
      p.delay = std::max(0.0, p.configuredDelay + noise_.delaySigma * unit(rng_));
      p.startDelay = std::max(0.0, p.configuredStartDelay + noise_.delaySigma * unit(rng_));
    }
    // Rebuild the clock rather than nudging it: a segment's start is the sum of everything
    // before it, so scaling durations in place would leave the starts describing a different
    // plan than the durations do. The dwells are the gaps between the configured segments.
    double clock = p.startDelay;
    for(std::size_t i = 0; i < p.episodeSegments.size(); ++i)
    {
      const auto & cfg = p.configuredSegments[i];
      double dwell = 0.0;
      if(i + 1 < p.configuredSegments.size())
      {
        dwell = p.configuredSegments[i + 1].start - cfg.start - cfg.duration;
      }
      double scale = 1.0;
      if(noise_.timingSigma > 0.0)
      {
        scale = std::max(0.5, 1.0 + noise_.timingSigma * unit(rng_));
      }
      p.episodeSegments[i].duration = std::max(1e-3, cfg.duration * scale);
      p.episodeSegments[i].start = clock;
      clock += p.episodeSegments[i].duration + std::max(0.0, dwell * scale);
    }
  }
  applyTargetDisagreement();
  for(auto & p : partners_)
  {
    p.segments = p.episodeSegments;
  }
}

void SimHumanPartner::applyTargetDisagreement()
{
  if(!noise_.active() || noise_.targetSigma <= 0.0 || partners_.empty())
  {
    return;
  }
  // What makes a dyad negotiate is the DIFFERENCE between two beliefs about where the peg is.
  // What makes it miss is their MEAN: the equilibrium of two springs pulling on the same plate
  // sits between their targets, so a common component is aim error, not interaction, and it
  // does not have to be large to matter -- the clearance is 1.5 mm and the compliance floor
  // already spends most of it.
  //
  // Drawing independently gives both at once, and the mean of two 4 mm draws misses the hole
  // reliably: measured, every second episode ended 13-15 mm off the peg and never descended.
  // So the draws are centred here. Each partner keeps a belief of its own, the pair still
  // disagrees by the full sigma, and the point they settle on is the peg.
  // A partner with zero gains applies no force, so it is not one of the springs whose
  // equilibrium the mean describes. Centring over it would shift the real partners' beliefs
  // by half of its draw and make a DAgger episode physically unlike the evaluation episode it
  // is meant to label -- the one thing the ghost expert must not do. See `passive` in the
  // learning repo's PartnerRole.
  std::vector<Partner *> acting;
  for(auto & p : partners_)
  {
    if(p.stiffness.isZero() && p.damping.isZero())
    {
      p.targetOffset.setZero();
      continue;
    }
    acting.push_back(&p);
  }
  if(acting.empty())
  {
    return;
  }

  std::normal_distribution<double> unit(0.0, 1.0);
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for(auto * p : acting)
  {
    for(int axis = 0; axis < 3; ++axis)
    {
      p->targetOffset[axis] = noise_.targetSigma * unit(rng_);
    }
    mean += p->targetOffset;
  }
  mean /= static_cast<double>(acting.size());
  for(auto * p : acting)
  {
    p->targetOffset -= mean;
    // A person who is wrong about the target is wrong about it throughout the reach, not only
    // once they arrive, so the belief shifts the whole plan rather than only its end.
    for(auto & seg : p->episodeSegments)
    {
      seg.from += p->targetOffset;
      seg.to += p->targetOffset;
    }
    p->episodeFinalTarget += p->targetOffset;
  }
}

void SimHumanPartner::init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config)
{
  dt_ = controller.controller().solver().dt();
  robot_ = config("robot", controller.controller().robot().name());
  config("enabled", enabled_);
  config("auto_episodes", autoEpisodes_);
  config("compliant_episodes", compliantEpisodes_);
  config("settle_time", settleTime_);
  config("home_time", homeTime_);
  config("episode_duration", episodeDuration_);
  config("success_depth", successDepth_);
  config("success_radius", successRadius_);
  config("stop_on_success", stopOnSuccess_);
  haveTarget_ = config.has("target");
  config("target", target_);
  config("insertion_body", insertionBody_);
  config("insertion_point", insertionPoint_);
  config("episodes", episodesRequested_);
  parsePartners(config);
  parseNoise(config);

  if(!controller.controller().datastore().has(robot_ + "::ApplyWrenchOnBody"))
  {
    // Not fatal: the controller must still run on hardware, where there is nothing to push
    // with. But it means every partner is inert, which is worth saying loudly once.
    mc_rtc::log::warning(
        "[SimHumanPartner] '{}::ApplyWrenchOnBody' is not in the datastore. The simulated partners "
        "will do nothing. Expected when running on the real robot; in simulation it means mc_mujoco "
        "is too old -- the callback landed in commit c777ac4d of the fork.",
        robot_);
  }

  const auto & robot = controller.controller().robot(robot_);
  for(const auto & p : partners_)
  {
    if(!robot.hasBody(p.body))
    {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[SimHumanPartner] partner '{}' pushes on body '{}', which robot '{}' does not have.",
          p.name, p.body, robot_);
    }
    if(!robot.hasBody(p.referenceBody))
    {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[SimHumanPartner] partner '{}' aims at body '{}', which robot '{}' does not have.",
          p.name, p.referenceBody, robot_);
    }
  }

  // Resolved whatever the episode mode is: `rebase` needs the hole to register a relative
  // partner's plan, and episodes can be driven from the GUI without auto_episodes ever being
  // set in the configuration.
  if(insertionBody_.empty())
  {
    insertionBody_ = partners_.front().referenceBody;
    insertionPoint_ = partners_.front().referencePoint;
  }
  if(!robot.hasBody(insertionBody_))
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[SimHumanPartner] insertion_body '{}' is not a body of robot '{}'.", insertionBody_, robot_);
  }

  if(autoEpisodes_)
  {
    // Refusing here rather than defaulting to something: with no target the success test
    // can only ever return false, and a batch of uniformly failed episodes looks like a
    // hard task rather than a missing configuration line.
    if(!haveTarget_)
    {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[SimHumanPartner] auto_episodes needs 'target': the world position of the peg tip. "
          "The controller cannot supply it -- its insertion_frame is a frame on the tool, which "
          "moves with the arm. Generate the config with scripts/make_sim_config.py.");
    }
  }

  addLog(controller);
  addGui(controller);
  reset(controller);
}

void SimHumanPartner::reset(mc_control::MCGlobalController & controller)
{
  t_ = 0.0;
  episodeTime_ = 0.0;
  seatedDepth_ = 0.0;
  seatedLateral_ = 0.0;
  timeToSeat_ = -1.0;
  everInserted_ = false;
  phase_ = Phase::Idle;
  phaseTime_ = 0.0;
}

Eigen::Vector3d SimHumanPartner::insertionWorld(const mc_rbdyn::Robot & robot) const
{
  const sva::PTransformd & X_0_b = robot.bodyPosW(insertionBody_);
  // sva stores the rotation as world->body, so the transpose maps a body-frame offset back
  // into the world.
  return X_0_b.translation() + X_0_b.rotation().transpose() * insertionPoint_;
}

double SimHumanPartner::insertionDepth(const mc_rbdyn::Robot & robot, double & lateral)
{
  const Eigen::Vector3d hole = insertionWorld(robot);
  lateral = (hole.head<2>() - target_.head<2>()).norm();
  // Positive once the hole has descended past the peg tip, i.e. swallowed that much peg.
  const double depth = target_.z() - hole.z();
  insertionLateralLog_ = lateral;
  insertionDepthLog_ = depth;
  return depth;
}

void SimHumanPartner::before(mc_control::MCGlobalController & controller)
{
  t_ += dt_;
  updateEpisode(controller);

  if(!enabled_)
  {
    for(auto & p : partners_)
    {
      p.force.setZero();
      p.delivered = false;
    }
    return;
  }

  auto & ctl = controller.controller();
  const auto & robot = ctl.robot(robot_);
  // Logged every tick, not only while recording: where the hole *starts* is what decides
  // whether the peg was placed somewhere the arm can reach it by descending, and that is
  // read off the log before any episode runs.
  insertionWorldLog_ = insertionWorld(robot);
  // Before the partner loop, so every partner reads the same verdict from a pose measured
  // this tick.
  updateStuck(insertionWorldLog_);
  const std::string key = robot_ + "::ApplyWrenchOnBody";
  const bool haveCallback = ctl.datastore().has(key);

  // With episodes driven automatically the partners act only while recording: pushing on a
  // stiff, position-controlled arm during homing achieves nothing and pollutes the log.
  // Driven by hand they act continuously, which is what you want while tuning.
  const bool acting = !autoEpisodes_ || phase_ == Phase::Recording;
  const double clock = autoEpisodes_ ? episodeTime_ : t_;

  for(auto & p : partners_)
  {
    // The error is measured at the point the person is aiming, which is the point they push
    // through -- anchoring the spring anywhere else makes it blind to the motion it causes.
    // See the Partner docs. Read before the reference because a state-fed partner's reference
    // is a function of it; `aimedPoint` only reads the robot, so the order is free.
    Eigen::Vector3d handPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d handVel = Eigen::Vector3d::Zero();
    aimedPoint(robot, p, handPos, handVel);

    Eigen::Vector3d xRef = Eigen::Vector3d::Zero();
    Eigen::Vector3d vRef = Eigen::Vector3d::Zero();
    if(p.feedback.enabled())
    {
      // Where the plan says to go next *from where this hand actually is*. No delay term: a
      // reaction time is a property of executing an answer, not of the answer, and a label
      // that lags the state it is labelling is the defect this exists to remove.
      referenceAt(p, phaseFromState(p, handPos, dt_) + p.feedback.lookaheadTime, xRef, vRef);
    }
    else
    {
      // Evaluating the intent in the past is the whole of the sensorimotor delay model: the
      // partner acts on where it meant to be a reaction time ago, not on where it is now.
      referenceAt(p, clock - p.delay, xRef, vRef);
    }

    // What this person's hand feels, from the plan alone. Read before the dither is added so
    // that the search cannot hold itself on through its own displacement of the reference.
    // `feltStiffness` rather than `stiffness`: a zero-gain DAgger expert would otherwise feel
    // nothing, never register contact, and silently label without the corrective search.
    p.resistance = p.feltStiffness().cwiseProduct(xRef - handPos).norm();

    // Added to the intent, deliberately: `reference` is what is logged, extracted and cloned,
    // so a correction applied to the force instead would be invisible to a policy that
    // imitates intent.
    if(acting && phase_ == Phase::Recording)
    {
      xRef += searchOffset(p, p.resistance);
    }

    if(!acting)
    {
      p.reference = xRef;
      p.handPosition = handPos;
      p.force.setZero();
      p.saturated = false;
      p.delivered = false;
      continue;
    }

    Eigen::Vector3d force = p.stiffness.cwiseProduct(xRef - handPos) + p.damping.cwiseProduct(vRef - handVel);
    const double norm = force.norm();
    p.saturated = norm > p.maxForce;
    if(p.saturated && norm > 1e-9)
    {
      // A human is not an ideal spring. Without this, an initial reference offset produces
      // a transient no arm could apply and the contact goes unstable.
      force *= p.maxForce / norm;
    }

    p.reference = xRef;
    p.handPosition = handPos;
    p.force = force;
    p.delivered = false;

    if(haveCallback)
    {
      // World-frame wrench, body-frame application point, every tick: mc_mujoco clears its
      // queue at the start of each control cycle. A point contact contributes no couple of
      // its own -- the moment about the body origin is what the offset point produces, and
      // mc_mujoco derives that from localPoint.
      const sva::ForceVecd wrench(Eigen::Vector3d::Zero(), force);
      p.delivered = ctl.datastore().call<bool, const std::string &, const sva::ForceVecd &, const Eigen::Vector3d &>(
          key, p.body, wrench, p.localPoint);
      if(!p.delivered && !warnedMissingCallback_)
      {
        mc_rtc::log::warning("[SimHumanPartner] mc_mujoco rejected body '{}' for partner '{}'", p.body, p.name);
        warnedMissingCallback_ = true;
      }
    }
  }
}

void SimHumanPartner::updateEpisode(mc_control::MCGlobalController & controller)
{
  if(!autoEpisodes_)
  {
    return;
  }
  auto * collab = dynamic_cast<CollabPegInHoleController *>(&controller.controller());
  if(collab == nullptr)
  {
    // Driving episodes means calling the controller's own start/mark/stop and its own state
    // requests, so that the markers, the log entries and the extractor stay the single ones
    // that exist. Against any other controller there is nothing to drive.
    mc_rtc::log::error("[SimHumanPartner] auto_episodes needs CollabPegInHoleController; disabling it");
    autoEpisodes_ = false;
    return;
  }

  auto & ctl = controller.controller();
  const auto & robot = ctl.robot(robot_);
  phaseTime_ += dt_;

  // "Compliant" is a *request*: requestState only records it, the FSM transitions on a later
  // tick, and Compliant::start() is what finally assigns ControlMode. So the control mode --
  // not the request -- is what says whether the arm is actually torque controlled.
  const auto controlMode = ctl.datastore().has("ControlMode")
                               ? ctl.datastore().get<std::string>("ControlMode")
                               : std::string("Position");
  const bool compliant = controlMode == "Torque";

  auto enter = [this](Phase phase)
  {
    phase_ = phase;
    phaseTime_ = 0.0;
  };

  switch(phase_)
  {
    case Phase::Idle:
    {
      if(episodesRequested_ == 0)
      {
        return;
      }
      if(!compliantEpisodes_)
      {
        startRecording(*collab, robot);
        return;
      }
      collab->requestState("Compliant");
      enter(Phase::GoingCompliant);
      return;
    }
    case Phase::GoingCompliant:
    {
      if(compliant)
      {
        enter(Phase::Settling);
        return;
      }
      // Re-assert every tick rather than asking once. `requestedState_` is a single slot
      // that the controller writes too: entering Initial calls switchToInitialState(),
      // which sets it back to "Initial". On the very first tick the plugin asks before the
      // FSM has started its initial state, so a single request is overwritten moments later
      // and the transition never happens -- silently, because nothing is in error.
      collab->requestState("Compliant");
      // The FSM only leaves Initial when that state's run() sees the request, so a couple of
      // ticks are normal. Seconds are not, and mean the request is not being consumed.
      if(phaseTime_ > 5.0)
      {
        mc_rtc::log::error("[SimHumanPartner] still not compliant {:.1f} s after asking; giving up "
                           "on driving episodes. Control mode is '{}', "
                           "requested state is '{}'.",
                           phaseTime_, controlMode, collab->requestedState());
        autoEpisodes_ = false;
        enter(Phase::Idle);
      }
      return;
    }
    case Phase::Settling:
    {
      // The compliance gains ramp in over the controller's gain_transition_time; recording
      // through that ramp would put a partly-stiff arm into the demonstrations.
      if(phaseTime_ >= settleTime_)
      {
        startRecording(*collab, robot);
      }
      return;
    }
    case Phase::Recording:
    {
      episodeTime_ += dt_;

      // Success is a task condition, not a timeout: the tool's hole has to have come far
      // enough down over the peg. Judged from the controller's own insertion frame so that
      // simulated and real episodes are scored the same way.
      // Both conditions matter. Depth alone would score "the tool sank past the peg tip
      // somewhere else in the workspace" as an insertion, which is how the first version of
      // this test managed to succeed in 0.00 s.
      // It is a task condition but no longer necessarily a *stopping* condition -- see
      // stopOnSuccess_ and seatedDepth_ in the header for why the two were separated.
      double lateral = 0.0;
      const double depth = insertionDepth(robot, lateral);
      const bool inserted = lateral < successRadius_ && depth > successDepth_;

      // The graded outcome, accumulated every tick rather than sampled at the end. The
      // deepest sample is not the last one: the dyad can seat the peg and be pushed back
      // out again by the tail of its own plan, and an episode that stops at the threshold
      // reports the threshold whatever it was about to do.
      if(lateral < successRadius_ && depth > seatedDepth_)
      {
        seatedDepth_ = depth;
        seatedLateral_ = lateral;
      }
      if(inserted && !everInserted_)
      {
        everInserted_ = true;
        timeToSeat_ = episodeTime_;
      }

      // With stop_on_success false the episode always runs its full clock, so the outcome
      // is a threshold on what was recorded rather than the reason the recording stopped.
      const bool done = (stopOnSuccess_ && inserted) || episodeTime_ >= episodeDuration_;
      if(!done)
      {
        return;
      }

      collab->markEpisode(everInserted_ ? "success" : "failure");
      collab->stopEpisode();
      ++episodesCompleted_;
      if(episodesRequested_ > 0)
      {
        --episodesRequested_;
      }
      mc_rtc::log::info(
          "[SimHumanPartner] episode {} {} after {:.2f} s, seated {:.2f} mm at {:.2f} mm off axis ({} left)",
          episodesCompleted_, everInserted_ ? "succeeded" : "failed", episodeTime_, seatedDepth_ * 1e3,
          seatedLateral_ * 1e3, episodesRequested_ < 0 ? -1 : episodesRequested_);

      if(!compliantEpisodes_)
      {
        enter(Phase::Idle);
        return;
      }
      // Home between episodes so every one starts from the same configuration. Without this
      // each episode begins wherever the last one happened to end, and the dataset quietly
      // encodes that drift as if it were part of the task.
      collab->requestState("Initial");
      enter(Phase::GoingHome);
      return;
    }
    case Phase::GoingHome:
    {
      if(phaseTime_ >= homeTime_)
      {
        enter(Phase::Idle);
      }
      return;
    }
  }
}

size_t SimHumanPartner::stallWindowTicks() const
{
  double period = 0.0;
  for(const auto & p : partners_)
  {
    if(p.search.enabled() && p.search.frequency > 0.0)
    {
      period = std::max(period, 1.0 / p.search.frequency);
    }
  }
  return period > 0.0 ? static_cast<size_t>(std::lround(period / dt_)) : 1;
}

void SimHumanPartner::updateStuck(const Eigen::Vector3d & piece)
{
  // Average over one search period before deciding anything, so that the dither this plugin
  // is itself adding cannot register as the piece going somewhere. See stuckWindow_.
  const size_t window = stallWindowTicks();
  stuckWindow_.push_back(piece);
  stuckWindowSum_ += piece;
  while(stuckWindow_.size() > window)
  {
    stuckWindowSum_ -= stuckWindow_.front();
    stuckWindow_.pop_front();
  }
  const Eigen::Vector3d smooth = stuckWindowSum_ / static_cast<double>(stuckWindow_.size());

  // Seeded here, from a pose handed in by the caller. Never from a member updated later in
  // the tick: on the first tick of an episode those still hold the previous episode's values.
  if(!stuckSeeded_)
  {
    stuckOrigin_ = smooth;
    lastMotionTime_ = episodeTime_;
    stuckSeeded_ = true;
  }
  // Displacement from where it last was, not depth: a piece grinding sideways against a
  // chamfer is going somewhere, and a piece that has stopped has stopped whatever the
  // direction of travel was. Depth also cannot be used as the progress measure while the
  // hole is still above the peg tip, which is most of the episode.
  // One piece, so one threshold: take the first partner's rather than inventing a plugin-level
  // duplicate of a number that already has a home in the config.
  if(partners_.empty())
  {
    pieceStuck_ = false;
    return;
  }
  const Partner::Search & spec = partners_.front().search;
  if((smooth - stuckOrigin_).norm() > spec.stallMotion)
  {
    stuckOrigin_ = smooth;
    lastMotionTime_ = episodeTime_;
  }
  const double quiet = spec.stallTime;
  pieceStuck_ = episodeTime_ - lastMotionTime_ > quiet;
}

Eigen::Vector3d SimHumanPartner::searchOffset(Partner & p, double resistance)
{
  // By default the spiral ramps four times faster than the fixed circle, and can afford to:
  // its radius is amplitude * spiralProgress, which starts near zero, so its early bursts are
  // gentle however quickly the envelope moves. It also had to be faster back when the median
  // burst lasted 0.48-0.56 s and a full-period ramp left every one of them at roughly a
  // seventh of the radius it asked for. `hold_time` has since made bursts 6.2 s, so that
  // argument no longer carries on its own -- see Search::rampTime, which is how the ramp gets
  // set without also moving the orbit speed.
  // `ramp_time`, when set, is the whole of the rate: the envelope crosses its range in that
  // many seconds whatever the orbit speed or the shape. Left at 0 this is arithmetically the
  // legacy expression, which is what keeps every dataset generated before it reproducible.
  const double rampSeconds =
      p.search.rampsExplicitly()
          ? p.search.rampTime
          : 1.0 / (std::max(p.search.frequency, 1e-6) * (p.search.spiral() ? 4.0 : 1.0));
  const double ramp = p.search.enabled() ? dt_ / std::max(rampSeconds, dt_) : 0.0;
  if(!p.search.enabled())
  {
    p.searching = false;
    p.searchEnvelope = 0.0;
    p.searchOffset.setZero();
    return p.searchOffset;
  }

  // Stuck means all three: this person is pushing against something, the piece has stopped
  // moving, and it has never gone in. Dropping the contact term makes the dither fire in free
  // space -- which it did, from the moment the transit arrived over the peg with the hole
  // still 240 mm up, because the old gate was proximity plus a stall detector that a stale
  // read had latched permanently.
  //
  // `!everInserted_` is load-bearing, and not merely tidiness about stopping the wiggle once
  // the peg is in. It is what fixes *what this mechanism claims to model*. Because the search
  // can never fire again after the piece has entered, it is incapable of representing a jam:
  // 94.8 % of its life is spent with the piece resting on the plate about 8.5 mm short of the
  // tip, and the 3.6 % past the tip is all at the 99th percentile and above -- the envelope
  // decaying while a successful insertion drops in. So the modelled behaviour is a person who
  // has set a plate down on a surface and slides it around until it drops, not one vibrating a
  // stuck peg free. That is what makes a 17 mm sweep defensible: it is scaled to the
  // registration uncertainty being hunted through, not to a jammed contact.
  //
  // Relaxing this term to handle partial insertions would take the justification with it.
  const bool inContact = resistance > p.search.contactForce;

  // `hold_time` bridges a dropout of the stall term only. The search displaces the piece, so
  // it clears its own stuck test and cuts itself off after a median 0.77 s -- every observed
  // interruption is this term, and each gap is exactly `stallTime` long. Without the bridge
  // the envelope never gets the continuous 1/frequency seconds it needs to reach the
  // commanded radius, and the configured shape is simply not what gets drawn.
  //
  // Contact and `everInserted_` are deliberately outside the hold: they still cut the search
  // on the tick they go false. See Search::holdTime.
  const bool stuckNow = pieceStuck_ && inContact && !everInserted_;
  if(stuckNow && p.search.holds())
  {
    p.searchHoldUntil = episodeTime_ + p.search.holdTime;
  }
  const bool held = p.search.holds() && p.searchHoldUntil >= 0.0 && episodeTime_ < p.searchHoldUntil;
  p.searching = (pieceStuck_ || held) && inContact && !everInserted_;

  // Ramp over one period rather than switching, so the reference is continuous in both
  // directions: a step of a whole amplitude is a kick the impedance turns into a force spike,
  // and it would be the largest transient in the episode.
  p.searchEnvelope = std::clamp(p.searchEnvelope + (p.searching ? ramp : -ramp), 0.0, 1.0);

  // Progress up the spiral accrues with searching time and is never given back inside an
  // episode, which is the whole point of keeping it separate from the envelope: the search
  // arrives in short bursts, and a radius that restarted with each one would never get out of
  // the middle of the circle. See Partner::spiralProgress.
  if(p.searching && p.search.spiral())
  {
    p.spiralProgress = std::min(1.0, p.spiralProgress + dt_ / p.search.spiralTime);
  }

  if(p.searchEnvelope <= 0.0)
  {
    p.searchPhase = 0.0;
    p.searchOffset.setZero();
    return p.searchOffset;
  }

  // A circle, not a line: which way to move to find the hole is exactly what is unknown, so
  // the search has to cover every lateral direction. Phase is integrated rather than computed
  // from the clock so that starting, stopping and restarting cannot jump it.
  p.searchPhase += 2.0 * M_PI * p.search.frequency * dt_;
  // For the fixed circle the amplitude *is* the radius, reached once and orbited. For the
  // spiral it is the outer radius and the search works out to it, so every radius below it is
  // swept on the way -- which is what lets it cross a hole whose distance it does not know.
  const double reach = p.search.spiral() ? p.spiralProgress : 1.0;
  const double radius = p.search.amplitude * reach * p.searchEnvelope;
  p.searchOffset = Eigen::Vector3d(radius * std::cos(p.searchPhase), radius * std::sin(p.searchPhase), 0.0);
  return p.searchOffset;
}

void SimHumanPartner::startRecording(CollabPegInHoleController & collab, const mc_rbdyn::Robot & robot)
{
  // Start each partner's reach from where its aimed point actually is, rather than from a
  // fixed world point: otherwise the first tick asks for a force no arm could apply and the
  // saturation limit is all that survives of the model. Relative partners are additionally
  // registered to the hole's real position now -- it varies by more than 100 mm between
  // episodes, and the plan was written assuming one place.
  drawEpisodeNoise();
  const Eigen::Vector3d holeOrigin = insertionWorld(robot) + registrationOffset_;
  for(auto & p : partners_)
  {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    aimedPoint(robot, p, position, velocity);
    rebase(p, position, holeOrigin);
  }
  for(auto & p : partners_)
  {
    p.resistance = 0.0;
    p.searching = false;
    p.searchPhase = 0.0;
    p.searchEnvelope = 0.0;
    p.spiralProgress = 0.0;
    p.searchOffset.setZero();
    p.searchHoldUntil = -1.0;
    // The plan is rebased onto this episode's hand, so last episode's phase means nothing.
    p.labelClock = -1.0;
  }
  // Re-seeded on the first tick of the episode from the pose that tick measures.
  stuckSeeded_ = false;
  pieceStuck_ = false;
  stuckWindow_.clear();
  stuckWindowSum_.setZero();
  collab.startEpisode();
  episodeTime_ = 0.0;
  seatedDepth_ = 0.0;
  seatedLateral_ = 0.0;
  timeToSeat_ = -1.0;
  everInserted_ = false;
  phase_ = Phase::Recording;
  phaseTime_ = 0.0;
}

void SimHumanPartner::after(mc_control::MCGlobalController &) {}

void SimHumanPartner::addLog(mc_control::MCGlobalController & controller)
{
  auto & logger = controller.controller().logger();
  logger.addLogEntry("sim_human_enabled", [this]() { return enabled_; });
  logger.addLogEntry("sim_human_episode_time", [this]() { return episodeTime_; });
  logger.addLogEntry("sim_human_episodes_done", [this]() { return episodesCompleted_; });
  logger.addLogEntry("sim_human_phase", [this]() { return std::string(phaseName(phase_)); });
  // The two numbers the success test is actually made of. Without them a logged 'success' is
  // an assertion, not a measurement, and there is no way to tell a real insertion from a
  // mis-specified target after the fact.
  // Where the peg is, world frame. Constant for a run, and scene knowledge the controller
  // does not have -- which is exactly why it has to be in the log. A policy trained to
  // replace a partner has to see the error that partner is closing, and that error is
  // (hole - peg). Without this the peg's position survives only in the generated yaml beside
  // the run, so an .npz stops being self-contained and a dataset pooled across placements
  // cannot be built at all.
  logger.addLogEntry("sim_human_target", [this]() -> const Eigen::Vector3d & { return target_; });
  logger.addLogEntry("sim_human_insertion_depth", [this]() { return insertionDepthLog_; });
  logger.addLogEntry("sim_human_insertion_lateral", [this]() { return insertionLateralLog_; });
  logger.addLogEntry("sim_human_piece_stuck", [this]() { return pieceStuck_; });
  // The graded outcome and its timing. Logged per tick like everything else, so the value at
  // the episode's last tick is the episode's value -- which is what extract_episodes.py and
  // run_batch.py both read. Held flat after the peg is seated, deliberately: an episode that
  // runs on past the insertion must not lose what it achieved.
  logger.addLogEntry("sim_human_seated_depth", [this]() { return seatedDepth_; });
  logger.addLogEntry("sim_human_seated_lateral", [this]() { return seatedLateral_; });
  logger.addLogEntry("sim_human_time_to_seat", [this]() { return timeToSeat_; });
  // The thresholds this run scored against. Constant for a run, but an episode whose .npz
  // does not carry them cannot be re-scored at a different depth later, which is the whole
  // point of recording the seated depth in the first place.
  logger.addLogEntry("sim_human_success_depth", [this]() { return successDepth_; });
  logger.addLogEntry("sim_human_success_radius", [this]() { return successRadius_; });
  logger.addLogEntry("sim_human_noise_seed", [this]() { return static_cast<double>(noise_.seed); });
  logger.addLogEntry("sim_human_noise_registration",
                     [this]() -> const Eigen::Vector3d & { return registrationOffset_; });
  logger.addLogEntry("sim_human_insertion_world", [this]() -> const Eigen::Vector3d & { return insertionWorldLog_; });
  for(std::size_t i = 0; i < partners_.size(); ++i)
  {
    const std::string prefix = "sim_human_" + partners_[i].name;
    // Captured by index: `partners_` must not be reallocated after this point, which is
    // why it is only ever filled in init().
    logger.addLogEntry(prefix + "_force", [this, i]() -> const Eigen::Vector3d & { return partners_[i].force; });
    logger.addLogEntry(prefix + "_reference", [this, i]() -> const Eigen::Vector3d & { return partners_[i].reference; });
    logger.addLogEntry(prefix + "_hand", [this, i]() -> const Eigen::Vector3d & { return partners_[i].handPosition; });
    logger.addLogEntry(prefix + "_error", [this, i]() { return (partners_[i].reference - partners_[i].handPosition).norm(); });
    logger.addLogEntry(prefix + "_saturated", [this, i]() { return partners_[i].saturated; });
    // The episode's draw. Constant across an episode by construction, and the only record of
    // what this partner actually was: the generated yaml holds the distribution, not the
    // sample.
    logger.addLogEntry(prefix + "_target_offset",
                       [this, i]() -> const Eigen::Vector3d & { return partners_[i].targetOffset; });
    logger.addLogEntry(prefix + "_gain_scale", [this, i]() { return partners_[i].gainScale; });
    logger.addLogEntry(prefix + "_delay", [this, i]() { return partners_[i].delay; });
    logger.addLogEntry(prefix + "_delivered", [this, i]() { return partners_[i].delivered; });
    // The search, logged whether or not it is enabled: an episode where nobody wiggled and one
    // where the feature was compiled out have to be distinguishable in the dataset.
    logger.addLogEntry(prefix + "_searching", [this, i]() { return partners_[i].searching; });
    // The trigger's own input, so "why did it wiggle there" is answerable from the log rather
    // than by reconstructing a spring from poses afterwards.
    logger.addLogEntry(prefix + "_resistance", [this, i]() { return partners_[i].resistance; });
    // Where on its own plan a state-fed partner thinks it is. Held at -1 on the clock path, so
    // a log says which kind of expert produced it rather than leaving that to the config file.
    logger.addLogEntry(prefix + "_label_clock", [this, i]() { return partners_[i].labelClock; });
    logger.addLogEntry(prefix + "_search_offset",
                       [this, i]() -> const Eigen::Vector3d & { return partners_[i].searchOffset; });
  }
}

void SimHumanPartner::addGui(mc_control::MCGlobalController & controller)
{
  auto & gui = *controller.controller().gui();
  const std::vector<std::string> category = {"Collab", "Sim humans"};
  gui.addElement(category, mc_rtc::gui::Checkbox("Partners active", [this]() { return enabled_; },
                                                 [this]() { enabled_ = !enabled_; }));
  gui.addElement(category, mc_rtc::gui::Checkbox("Drive episodes", [this]() { return autoEpisodes_; },
                                                 [this]() { autoEpisodes_ = !autoEpisodes_; }));
  gui.addElement(category, mc_rtc::gui::NumberInput("Episodes left",
                                                    [this]() { return static_cast<double>(episodesRequested_); },
                                                    [this](double v) { episodesRequested_ = static_cast<int>(v); }));
  gui.addElement(category, mc_rtc::gui::Label("Episodes done", [this]() { return episodesCompleted_; }));
  gui.addElement(category, mc_rtc::gui::Label("Episode time [s]", [this]() { return episodeTime_; }));
  gui.addElement(category, mc_rtc::gui::Label("Phase", [this]() { return std::string(phaseName(phase_)); }));
  gui.addElement(category, mc_rtc::gui::Checkbox("Drive the FSM too", [this]() { return compliantEpisodes_; },
                                                 [this]() { compliantEpisodes_ = !compliantEpisodes_; }));

  for(std::size_t i = 0; i < partners_.size(); ++i)
  {
    const std::vector<std::string> sub = {"Collab", "Sim humans", partners_[i].name};
    gui.addElement(sub, mc_rtc::gui::Label("Body", [this, i]() { return partners_[i].body; }));
    gui.addElement(sub, mc_rtc::gui::Label("Force norm [N]", [this, i]() { return partners_[i].force.norm(); }));
    gui.addElement(sub, mc_rtc::gui::Label("Tracking error [m]",
                                           [this, i]() { return (partners_[i].reference - partners_[i].handPosition).norm(); }));
    gui.addElement(sub, mc_rtc::gui::Label("Saturated", [this, i]() { return partners_[i].saturated; }));
    // Seeing where a partner *wants* the hand to be is the fastest way to spot a
    // trajectory expressed in the wrong frame.
    gui.addElement(sub, mc_rtc::gui::Point3D("Intent", [this, i]() -> const Eigen::Vector3d & { return partners_[i].reference; }));
    gui.addElement(sub, mc_rtc::gui::Arrow("Applied force",
                                           [this, i]() -> const Eigen::Vector3d & { return partners_[i].handPosition; },
                                           [this, i]() { return Eigen::Vector3d(partners_[i].handPosition + 0.01 * partners_[i].force); }));
  }
}

void SimHumanPartner::removeGui(mc_control::MCGlobalController & controller)
{
  controller.controller().gui()->removeCategory({"Collab", "Sim humans"});
}

mc_control::GlobalPlugin::GlobalPluginConfiguration SimHumanPartner::configuration()
{
  mc_control::GlobalPlugin::GlobalPluginConfiguration out;
  out.should_run_before = true; // the wrench must be queued before mc_mujoco steps
  out.should_run_after = false;
  out.should_always_run = false;
  return out;
}

} // namespace collab

EXPORT_MC_RTC_PLUGIN("SimHumanPartner", collab::SimHumanPartner)
