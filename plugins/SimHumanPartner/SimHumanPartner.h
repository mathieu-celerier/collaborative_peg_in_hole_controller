#pragma once

#include <mc_control/GlobalPlugin.h>

struct CollabPegInHoleController;

#include <SpaceVecAlg/SpaceVecAlg>

#include <deque>
#include <random>
#include <string>
#include <vector>

namespace collab
{

/** Simulated human partners, applied as external wrenches in mc_mujoco.
 *
 * The real experiment has *two* people acting on one arm, on opposite sides of the F/T
 * sensor (see the controller's CLAUDE.md). Neither can be recorded at the moment, so both
 * are simulated here and the demonstrations are generated rather than performed.
 *
 * Each simulated partner is a **point-of-contact force source**: it has an intent -- a
 * via-point minimum-jerk trajectory for its hand -- and a task-space impedance that turns
 * the difference between that intent and where the robot actually is into a force. That is
 * deliberately all a partner is: an F/T sensor and a joint-torque residual never observe a
 * human body, only a wrench at a contact point.
 *
 * Where the wrench lands decides which channel observes it, and this is the whole basis of
 * the dataset design:
 *
 * - **person A**, the demonstrator, pushes on an arm link *above* the F/T sensor. Invisible
 *   to the sensor, observable only through the joint-torque residual.
 * - **person B**, the collaborator, pushes on the tool *below* the sensor, and is therefore
 *   measured by it.
 *
 * Nothing here enforces that split -- it is a property of where MuJoCo's force sensor sits
 * in the kinematic chain, verified in the learning repo's `test_ft_sensor_separates_the_two_partners`.
 * Point a partner at the wrong body and the channels silently swap.
 *
 * ### Where the numbers come from
 *
 * The gains are **not** tuned here. They are generated, together with the trajectories, by
 * `cpih.sim.plugin_config` in the learning repo: for an M3 partner they are the Nash
 * equilibrium of a two-player differential game, whose weights are recorded as comments in
 * the generated YAML. This plugin only evaluates the runtime form both models reduce to --
 * a via-point polynomial plus constant gains -- which is why it needs no Riccati solver.
 *
 * Consequently: regenerate the config, do not hand-edit it.
 *
 * ### Delivery
 *
 * Wrenches reach MuJoCo through mc_mujoco's `{robot}::ApplyWrenchOnBody` datastore
 * callback. It expects the wrench in the **world** frame and the application point in the
 * **body** frame, and it clears its queue every control cycle -- hence the call happens on
 * every tick, from `before()`, ahead of the controller's own run.
 */
struct SimHumanPartner : public mc_control::GlobalPlugin
{
  void init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config) override;
  void reset(mc_control::MCGlobalController & controller) override;
  void before(mc_control::MCGlobalController & controller) override;
  void after(mc_control::MCGlobalController & controller) override;
  mc_control::GlobalPlugin::GlobalPluginConfiguration configuration() override;

  ~SimHumanPartner() override = default;

private:
  /** One leg of a via-point minimum-jerk trajectory. */
  struct Segment
  {
    double start = 0.0;    // when this segment begins, s
    double duration = 1.0; // how long it takes, s
    Eigen::Vector3d from = Eigen::Vector3d::Zero();
    Eigen::Vector3d to = Eigen::Vector3d::Zero();
  };

  /** One simulated person.
   *
   * Two frames matter and they are not the same one:
   *
   * - **where the force is applied** (`body`, `localPoint`) -- where the person's hands are;
   * - **what the person is trying to position** (`referenceBody`, `referencePoint`) -- the
   *   feature whose motion the impedance closes on.
   *
   * These must be the *same* point, and `relative` is what makes that possible for person A.
   *
   * The tempting arrangement -- person A pushes `forearm_link` while their error is measured
   * at the tool, because the tool is what they are watching -- is unstable, and not subtly:
   * it aborts the QP within two seconds of the compliance stiffness ramping down. An
   * unanchored spring cannot see the motion it causes. The arm has internal configurations
   * that move the forearm a long way while barely moving the tool (joints 1 and 3 counter-
   * rotating: q0 + q2 stays put to within 0.05 rad while each runs at 1.8 rad/s). Person A's
   * force excites that motion directly, their error does not register it, so nothing opposes
   * it; it winds up until it hits the joint velocity bounds and QLD reports the constraint
   * set inconsistent. Person B never triggers it because they push the very body they aim.
   *
   * So the impedance is always anchored where the hand is. Person A's tool-space *intent*
   * survives as `relative`: their waypoints are displacements, applied to wherever their own
   * hand starts, so "bring the tool 28 cm down" becomes "move your hand 28 cm down". Exact
   * for a rigid translation; the mismatch left by the arm's kinematics shows up as a steady
   * push, which is what guiding a robot that cannot go quite where you want actually feels
   * like.
   */
  struct Partner
  {
    std::string name;
    std::string body;                                          // body the wrench is applied on
    Eigen::Vector3d localPoint = Eigen::Vector3d::Zero();       // application point, body frame
    std::string referenceBody;                                 // body whose motion is being aimed
    Eigen::Vector3d referencePoint = Eigen::Vector3d::Zero();   // aimed point, reference frame
    Eigen::Vector3d stiffness = Eigen::Vector3d::Zero();        // N/m, per world axis
    Eigen::Vector3d damping = Eigen::Vector3d::Zero();          // N.s/m, per world axis
    double maxForce = 80.0;                                     // saturation on the force norm, N
    double delay = 0.0;                                         // sensorimotor delay, s
    double startDelay = 0.0;                                    // wait before moving, s
    /** The waypoints describe the *hole's* path, and this hand carries it out.
     *
     * Both people need it: the demonstrator's hand is on the flange and the collaborator's is
     * on the handle, so neither is on the hole they are both aiming at. A world-frame target
     * belonging to the hole would have them haul their own hand onto the peg.
     *
     * `rebase` therefore converts the plan into this hand's frame by the hand-to-hole offset
     * measured at episode start (`origin - holeOrigin`), which is exact while the tool
     * translates. What it must *not* do is shift by the hand's offset from a configured start
     * pose: the plan's waypoints are absolute and true -- they end at the peg, a known world
     * point -- and anchoring them on an assumed start throws that registration away. The hole
     * really starts up to 130 mm from where the generator assumes, and the dyad then runs
     * open-loop from wherever the arm happened to be. */
    bool relative = false;
    std::vector<Segment> segments;
    Eigen::Vector3d finalTarget = Eigen::Vector3d::Zero();
    // The configured trajectory, kept so that rebasing an episode is idempotent: each episode
    // rebases from `configured*`, never from the previous episode's already-shifted copy.
    std::vector<Segment> configuredSegments;
    Eigen::Vector3d configuredStart = Eigen::Vector3d::Zero();
    Eigen::Vector3d configuredFinalTarget = Eigen::Vector3d::Zero();
    // Everything the per-episode noise touches has the same discipline: the configured value
    // is the one that is drawn FROM, every episode, so a draw never compounds on the last
    // one. With noise off these are copied through unchanged and nothing below can be told
    // from the configuration it came from.
    Eigen::Vector3d configuredStiffness = Eigen::Vector3d::Zero();
    Eigen::Vector3d configuredDamping = Eigen::Vector3d::Zero();
    double configuredDelay = 0.0;
    double configuredStartDelay = 0.0;
    /** The plan this episode flies: the configured one after this episode's draw.
     *
     * `rebase` builds `segments` from THIS rather than from `configuredSegments`, so the
     * noise survives the registration and the registration still starts from a clean plan. */
    std::vector<Segment> episodeSegments;
    Eigen::Vector3d episodeFinalTarget = Eigen::Vector3d::Zero();
    // The realised draw, kept for the log. An episode whose perturbation is not recorded
    // cannot be reproduced from the dataset, which is most of the point of randomising.
    Eigen::Vector3d targetOffset = Eigen::Vector3d::Zero();
    double gainScale = 1.0;

    // Live state, for logging and the GUI.
    Eigen::Vector3d reference = Eigen::Vector3d::Zero();
    Eigen::Vector3d handPosition = Eigen::Vector3d::Zero(); // the aimed point, world frame
    Eigen::Vector3d force = Eigen::Vector3d::Zero();
    bool saturated = false;
    bool delivered = false; // did ApplyWrenchOnBody accept the last call?
  };

  /** Per-episode randomisation of the partners.
   *
   * Without it every episode of a run is the same episode -- the plans, gains and delays are
   * fixed at generation time -- so the dyad reaches the same place by the same route and there
   * is nothing in the recording about interaction. It lives here rather than in the generator
   * because a simulator launch costs ~20 s and drives many episodes: randomising upstream
   * would mean one launch per episode.
   *
   * Every sigma defaults to zero, and an absent `noise` block leaves the plugin behaving
   * exactly as it did before this existed. */
  struct Noise
  {
    unsigned int seed = 0;
    double targetSigma = 0.0; // m, on a partner's belief about where the peg is
    double gainSigma = 0.0;   // fraction of stiffness and damping
    double timingSigma = 0.0; // fraction of each segment's duration and dwell
    double delaySigma = 0.0;  // s, on the sensorimotor and start delays
    double startSigma = 0.0;  // m, on the registration the plan is rebased onto

    bool active() const
    {
      return targetSigma > 0.0 || gainSigma > 0.0 || timingSigma > 0.0 || delaySigma > 0.0
             || startSigma > 0.0;
    }
  };

  void parseNoise(const mc_rtc::Configuration & config);
  /** Draw this episode's perturbation into every partner's `episode*` fields.
   *
   * Called once, at the top of `startRecording`, before anything is rebased. With noise off
   * it is a copy of the configured plan. */
  void drawEpisodeNoise();
  /** Give each partner its own belief about the peg, with the pair's mean left on it.
   *
   * Separated from the rest of the draw because it is the one channel that couples the
   * partners: it is a property of the dyad, not of either person. */
  void applyTargetDisagreement();

  /** Minimum-jerk position and velocity of one partner's intent at time `t`. */
  static void referenceAt(const Partner & partner, double t, Eigen::Vector3d & position, Eigen::Vector3d & velocity);

  /** Where a partner's aimed point is now, and how fast it is moving. World frame. */
  static void aimedPoint(const mc_rbdyn::Robot & robot, const Partner & partner, Eigen::Vector3d & position,
                         Eigen::Vector3d & velocity);

  /** Anchor a partner's trajectory on this episode's actual geometry.
   *
   * `origin` is where this partner's own hand is, `holeOrigin` where the hole it is aiming
   * at is. Both are world frame, both measured at episode start.
   *
   * Every partner starts its first segment from `origin`, because otherwise it begins by
   * yanking: a trajectory that starts at a fixed world point has nothing to do with where the
   * robot happens to be, so the very first tick asks for hundreds of newtons and the
   * saturation limit is all that is left of the model. A person reaches from where their hand
   * is.
   *
   * A `relative` partner additionally has its whole plan translated by `origin - holeOrigin`,
   * turning "bring the hole down onto the peg" into "move your hand by that much". */
  static void rebase(Partner & partner, const Eigen::Vector3d & origin, const Eigen::Vector3d & holeOrigin);

  void parsePartners(const mc_rtc::Configuration & config);
  void addGui(mc_control::MCGlobalController & controller);
  void addLog(mc_control::MCGlobalController & controller);
  void removeGui(mc_control::MCGlobalController & controller);
  /** Drive the controller's own episode markers, for unattended batch generation. */
  void updateEpisode(mc_control::MCGlobalController & controller);
  /** Rebase every partner's reach and open an episode. */
  void startRecording(CollabPegInHoleController & collab, const mc_rbdyn::Robot & robot);

  // Last evaluated insertion measures, kept only so the success decision is auditable in the
  // log rather than being a verdict with no evidence behind it.
  double insertionDepthLog_ = 0.0;
  double insertionLateralLog_ = 0.0;
  Eigen::Vector3d insertionWorldLog_ = Eigen::Vector3d::Zero();

  std::vector<Partner> partners_;
  Noise noise_;
  std::mt19937 rng_{0};
  /** This episode's jitter on the hole's measured position, shared by every partner.
   *
   * The plugin cannot move the robot, so "the episode started from somewhere slightly
   * different" is expressed as the plan being registered onto a slightly different hole.
   * From the dyad's point of view those are the same thing: what they feel is the plan
   * landing a few millimetres off where the tool actually is. */
  Eigen::Vector3d registrationOffset_ = Eigen::Vector3d::Zero();
  std::string robot_;
  bool enabled_ = true;
  bool warnedMissingCallback_ = false;

  /** How an automatically driven episode progresses.
   *
   * Recording has to wait for the arm to actually *be* compliant: `requestState` only sets a
   * request, the FSM transitions a tick or more later, and the compliance gains then ramp in
   * over `gain_transition_time`. Recording through that ramp would put a stiff, position-
   * controlled arm into the demonstrations. */
  enum class Phase
  {
    Idle,           // nothing to do, or waiting for the next episode to be requested
    GoingCompliant, // asked for Compliant, waiting for the controller to get there
    Settling,       // compliant; letting the gain ramp finish before recording
    Recording,      // the episode proper -- the only phase the partners act in
    GoingHome,      // asked for Initial, letting the arm re-home before the next episode
  };

  static const char * phaseName(Phase phase);

  /** Where the tool's hole is right now, in the world. */
  Eigen::Vector3d insertionWorld(const mc_rbdyn::Robot & robot) const;

  /** How far the tool's hole has descended past the peg tip, and how far off-axis it is. */
  double insertionDepth(const mc_rbdyn::Robot & robot, double & lateral);

  // Episode driving.
  bool autoEpisodes_ = false;
  bool compliantEpisodes_ = true; // drive the FSM, rather than recording in whatever state
  double settleTime_ = 1.0;       // s, after reaching Compliant
  double homeTime_ = 3.0;         // s, after asking for Initial
  Phase phase_ = Phase::Idle;
  double phaseTime_ = 0.0;
  double episodeDuration_ = 10.0;
  // Where the peg is, in the world. Scene knowledge: the controller has no idea the peg
  // exists, and the frame it calls "insertion_frame" is a frame on the *tool*, so it moves
  // with the arm and cannot serve as a target.
  Eigen::Vector3d target_ = Eigen::Vector3d::Zero();
  bool haveTarget_ = false;
  double successDepth_ = 0.02;
  double successRadius_ = 0.01; // the hole must be over the peg, not merely below its tip
  /** Whether meeting the success test ends the episode.
   *
   * True is the interactive default, and what every episode used to do. It also censors the
   * outcome: an episode stopped the instant it is inside the threshold can never report how
   * much deeper it would have gone, so `success_depth` acts as a stopping rule rather than a
   * measurement and a batch saturates at whatever fraction of the dyads clear it. With this
   * false the episode runs to `episode_duration` whatever happens, `seatedDepth_` below is
   * the graded outcome, and the threshold can be re-chosen afterwards from the recording
   * instead of being fixed before it. */
  bool stopOnSuccess_ = true;
  /** The deepest the hole got *while it was over the peg*, this episode.
   *
   * Gated on `lateral < successRadius_` on purpose: a plate that sinks beside the peg
   * reports a large depth and has inserted nothing, which is the failure the success test's
   * radius term exists to reject. Stays 0 while the hole has never been over the peg. */
  double seatedDepth_ = 0.0;
  double seatedLateral_ = 0.0; // lateral at the tick seatedDepth_ was taken
  double timeToSeat_ = -1.0;   // s into the episode when the threshold was first met; -1 = never
  bool everInserted_ = false;  // whether the success predicate ever held this episode
  std::string insertionBody_;   // body carrying the feature being inserted (the tool)
  Eigen::Vector3d insertionPoint_ = Eigen::Vector3d::Zero(); // the hole centre, body frame
  int episodesRequested_ = 0;   // how many still to run; negative means "keep going"
  int episodesCompleted_ = 0;
  double episodeTime_ = 0.0;    // seconds since the current episode started
  double t_ = 0.0;              // plugin clock, seconds since reset
  double dt_ = 0.001;
};

} // namespace collab
