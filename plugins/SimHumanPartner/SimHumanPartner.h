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

    /** A corrective wiggle, for when the piece will not go in.
     *
     * A person whose peg has stopped going into the hole does not keep pushing harder along
     * the same line -- they jiggle it. Without this, a dyad that arrives slightly off-axis is
     * simply stuck: the plan runs to its end, the springs settle, and the episode records a
     * static standoff. Nothing in the demonstrations then shows a policy what to do about
     * being stuck, which is exactly the state a cloned policy spends its time in.
     *
     * The dither is added to the partner's *reference*, not to its force, so it lands in
     * `reference`, is logged, is extracted, and is therefore **clonable**. A correction
     * applied to the force would be invisible to a policy that clones intent.
     *
     * `amplitude` 0 disables it, so a config written before this existed is unchanged. */
    struct Search
    {
      double amplitude = 0.0; // m, radius of the lateral circle; 0 = off
      double frequency = 1.5; // Hz
      /** N of resistance this person must feel before searching at all.
       *
       * A person wiggles a piece because it is *against* something. Free-space motion is not
       * a reason to wiggle however long it lasts, and the piece being near the peg is not
       * either -- the corner of the L sits directly over the peg by construction, so a
       * proximity test opens 240 mm too high.
       *
       * The resistance is this partner's own spring stretch against the plan,
       * `stiffness * (planned reference - aimed point)`, which is what the person's hand
       * feels. Deliberately computed from the *undithered* plan: the dither displaces the
       * reference by a whole amplitude, so including it would let the search hold itself on. */
      double contactForce = 2.0;
      double stallTime = 0.5;      // s without the piece moving before deciding it is stuck
      double stallMotion = 0.001; // m the piece must travel to count as still going in
      /** Seconds of accumulated searching to grow the circle from nothing to `amplitude`.
       *
       * 0 keeps the original fixed-radius circle. Anything else makes it a spiral, which is a
       * different search and a better-posed one: a circle only finds the hole when its radius
       * happens to match how far off the piece landed, and that distance varies episode to
       * episode (p10 6.3 mm, p90 18.2 mm), so no single radius is right for most of them.
       * Measured on the fixed circle, the median failed episode never came within the 10 mm
       * lip at all -- only 22-27 % of them ever did -- because the piece was orbiting at a
       * radius of about 3 mm around a point 13.6 mm out. A spiral crosses every radius from 0
       * up to `amplitude`, so it passes over the hole whenever the amplitude exceeds the
       * offset instead of depending on the two matching.
       *
       * Sized against the search budget, not chosen freely: a failed episode gets a median
       * 6.4 s of searching in total, so a spiral much slower than that never reaches its
       * outer radius before the episode ends. */
      double spiralTime = 0.0;
      /** Seconds the search keeps running across a dropout of the *stall* term alone.
       *
       * The search moves the piece, so it clears its own stuck test and stops itself. The
       * smoothing window on `stuckWindow_` was meant to absorb that and only half does:
       * measured over 193 gaps on the 24 mm arm, **100 % of the interruptions are the stall
       * term clearing** -- not the contact term, which never once dropped -- and every gap is
       * exactly `stallTime` long, because that is how long re-arming takes.
       *
       * The cost is not the lost time, it is the lost amplitude. `searchEnvelope` ramps at
       * `frequency`, so a fixed circle needs 1/0.25 = 4 s of continuous searching to reach the
       * radius it was asked for, and the flag holds for a median 0.77 s before dropping for
       * 0.50 s. The envelope ratchets up across the short gaps rather than resetting, but only
       * to a median 12 mm of a commanded 24 mm, with 21 % of bursts reaching 90 %. So neither
       * the circle nor the spiral is delivered at the width it was configured with.
       *
       * This bridges only the stall term. `contactForce` and `everInserted_` still cut the
       * search immediately, which matters: the second of those is what limits this mechanism
       * to modelling a piece being slid around on a surface rather than a jam, and holding the
       * search through an insertion would take that justification with it.
       *
       * 0 disables the hold and reproduces every dataset generated before it existed. */
      double holdTime = 0.0;
      /** Seconds for `searchEnvelope` to travel its full range, in or out. 0 keeps the legacy
       * coupling, which is `1 / frequency` for the circle and a quarter of that for the spiral.
       *
       * Splitting this off matters because the two are not the same knob and the legacy rate
       * conflates them. The envelope ramping over exactly one period means the phase advances
       * exactly one full turn while the radius grows, so the "fixed circle" traces a spiral out
       * and a spiral back in, and holds a constant radius only in whatever burst time is left
       * over. Measured across 800 episodes at 0.25 Hz that is 16 % of search time: a plateau
       * needs a burst longer than 2/frequency = 8 s, and the median burst is 6.2 s.
       *
       * Raising `frequency` shortens the ramp *and* speeds the orbit together, so it cannot say
       * which of the two an effect came from. `rampTime` holds the orbit speed fixed and moves
       * only the ramp.
       *
       * It also symmetrises the spiral. The legacy 4x applies to the decay as well as the
       * climb, so the spiral climbs over `spiralTime` and is cut off in a quarter period; that
       * asymmetry does not cancel, and leaves a mean offset of 1.7 mm per burst against a
       * 1.5 mm seating radius, where the circle's symmetric ramps cancel to 0.8 mm. */
      double rampTime = 0.0;
      bool enabled() const { return amplitude > 0.0; }
      bool spiral() const { return spiralTime > 0.0; }
      bool holds() const { return holdTime > 0.0; }
      bool rampsExplicitly() const { return rampTime > 0.0; }
    };
    Search search;

    /** Reading the plan from the robot's state instead of from the clock.
     *
     * `referenceAt` is a function of `t` alone, which makes this partner a trajectory and not
     * a policy. That is fine for a *demonstrator* -- it is executing a plan -- and fatal for
     * a **DAgger expert**, which has to answer "what should be aimed at from the state the
     * learner has actually reached". From a drifted state the clock-driven answer is "be
     * where the plan says at time t", i.e. a jump proportional to how far the learner has
     * strayed. Training on that pair teaches exactly the oversized command that empties the
     * QP's constraint set, and it teaches it hardest in the states that drift most -- which
     * is the pattern the 2026-09-04 DAgger iteration produced (-5.7 pp, p = 3.3e-05).
     *
     * With `lookaheadTime > 0` the phase comes from where the aimed point actually *is*:
     * project it onto the plan's polyline, and read the reference that far ahead of the
     * projection. The command is then bounded by (path speed x lookahead) however far the
     * learner has drifted, and it points along the path rather than back at a stale clock --
     * which is what makes it a recovery.
     *
     * 0 is the clock, exactly as before this existed. */
    struct Feedback
    {
      /** Seconds ahead of the projected phase to aim. 0 disables the whole mechanism.
       *
       * This is the one parameter that sets how large a correction can be. Too small and the
       * expert aims at its own feet and never advances; too large and it recovers the jump
       * this exists to remove. */
      double lookaheadTime = 0.0;
      /** Seconds of plan the projected phase may advance per second of wall clock.
       *
       * The trajectory is an L, so the transit and the descent pass close to each other near
       * the corner and a nearest-point search can jump between legs. The phase is kept
       * monotone and rate-limited for that reason: a person does not un-execute a plan, and
       * does not skip half of one either.
       *
       * 1.0 would pin the expert to the plan's own pace, which would make a learner that is
       * *ahead* of the plan drag it backwards. The default is enough slack for a learner
       * moving faster than the plan and far too little to cross a segment in a tick. */
      double maxAdvanceRate = 3.0;
      /** Re-plan from the hand instead of aiming at the plan.
       *
       * The projection expert still aims *at* a fixed polyline, so its command carries the
       * **cross-track** error whatever the lookahead does: `|ref - hand|` is bounded below by
       * the distance from the hand to the path, and no choice of phase touches that term.
       * Measured on drifted leader-A-stiff states, both experts commanded ~100 mm and implied
       * ~50 N against a demonstrator that never exceeded 37.3 N.
       *
       * With this set, the reference is the plan's own displacement over the lookahead applied
       * **from the hand**, plus a correction back onto the path clamped to `maxCorrection`. So
       *
       *     |ref - hand| <= (path speed x lookaheadTime) + maxCorrection
       *
       * holds by construction rather than by hope, and on-plan the two experts agree: with the
       * hand at the projection the displacement term reproduces the projection expert exactly
       * and the correction is zero. What changes is only what happens once a learner has
       * strayed, which is the only place a DAgger label differs from a demonstration anyway. */
      bool replan = false;
      /** Metres of cross-track pull the re-planning expert may add. 0 leaves the label purely
       * along-track, which is bounded but never steers a drifted learner back to the hole.
       *
       * Sized against the demonstrated stretch, not against the drift: person A's own
       * `reference - hand` runs to 41.8 mm at the 95th percentile in the stiff cells, so a
       * correction of that order keeps the label inside the distribution the policy is being
       * asked to reproduce. */
      double maxCorrection = 0.0;
      bool enabled() const { return lookaheadTime > 0.0; }
    };
    Feedback feedback;

    /** The stiffness `resistance` is computed with, when it differs from `stiffness`.
     *
     * A DAgger expert is configured with **zero** gains so that it generates a reference and
     * applies no force (`--keep-expert`). But `resistance` is `stiffness * (reference - hand)`,
     * so a zero-gain partner feels identically nothing, `inContact` is never true, and its
     * search can never fire -- so its labels silently omit the corrective search that every
     * demonstration contains. That is not the `acting` gate, which is a phase flag and lets
     * the ghost through; it is the zero stiffness.
     *
     * Set this to the gains the partner *would* have had and the ghost feels what its
     * acting twin feels. Zero means "use `stiffness`", which is every ordinary partner. */
    Eigen::Vector3d labelStiffness = Eigen::Vector3d::Zero();

    /** The stiffness to measure resistance with: `labelStiffness` when set, else `stiffness`. */
    Eigen::Vector3d feltStiffness() const
    {
      return labelStiffness.isZero() ? stiffness : labelStiffness;
    }

    // Live search state, reset every episode by startRecording.
    double resistance = 0.0; // N this person is feeling against the plan, logged
    bool searching = false;  // is the stuck condition true right now
    double searchPhase = 0.0;      // rad, integrated so a frequency change cannot step the output
    /** 0..1, ramped in and out over one period.
     *
     * Without it the circle would start at full radius on its first tick, which is a step in
     * the reference of a whole amplitude -- a kick, not a wiggle, and one the impedance turns
     * into a force spike. */
    double searchEnvelope = 0.0;
    /** 0..1, how far along the spiral this partner has got, in accumulated searching time.
     *
     * Kept apart from `searchEnvelope` because the two answer different questions, and
     * conflating them is what makes a spiral impossible here: the envelope says whether the
     * offset is being applied *now* and must fall to zero the moment the search releases, but
     * the progress must survive that, or the spiral restarts from nothing every time.
     *
     * It has to survive it because the search does not run in one stretch. Measured over the
     * failed episodes of two arms, it fires in a median 5-6.5 bursts of a median 0.48-0.56 s
     * -- 50 degrees of arc at 0.25 Hz -- and fewer than 3 % of bursts last a whole turn. A
     * radius that resets per burst never leaves the middle of the circle. */
    double spiralProgress = 0.0;
    Eigen::Vector3d searchOffset = Eigen::Vector3d::Zero(); // what was added this tick, m
    /** Episode time until which the stall term is held true, or -1 when nothing is held.
     *
     * Set when the search *starts*, and extended for as long as the stuck test keeps
     * confirming itself; what it bridges is the gap after the test stops confirming. */
    double searchHoldUntil = -1.0;
    /** Episode time the state-fed phase has reached, kept monotone. -1 before the first tick.
     *
     * Monotone because the projection is a nearest-point search on a path that doubles back
     * on itself in space; letting it fall would make the expert re-execute the transit
     * whenever the descent brought the piece back past the corner. */
    double labelClock = -1.0;

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

  /** Episode time of the point on `partner`'s plan nearest `handPos`, kept monotone.
   *
   * The plan is a polyline of straight segments with min-jerk timing, so the projection is
   * exact: the nearest point of each segment is a clamped scalar projection onto its line,
   * and the time that reaches it is the min-jerk inverse of that fraction. Returned as a time
   * so that the answer can go straight back through `referenceAt` -- the state-fed expert and
   * the clock-driven one are then the same function of a phase, differing only in where the
   * phase comes from.
   *
   * `partner.labelClock` carries the monotone floor and is advanced by this call. */
  static double phaseFromState(Partner & partner, const Eigen::Vector3d & handPos, double dt);

  /** The re-planning expert's reference: a plan re-anchored on the hand.
   *
   * See `Feedback::replan`. Not `const`, because it advances the partner's monotone phase
   * exactly as the projection expert does.
   */
  static void replannedReference(Partner & partner, const Eigen::Vector3d & handPos, double dt,
                                 Eigen::Vector3d & position, Eigen::Vector3d & velocity);

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
  /** Whether the piece itself has stopped moving. One piece, so one verdict for both people.
   *
   * Seeded from the pose passed in rather than from a member, because the member a previous
   * version read (`insertionDepthLog_`) is not written until later in the first tick of an
   * episode: it still held the *previous* episode's final depth, which latched the stall
   * detector permanently and left the wiggle gated on proximity alone. */
  void updateStuck(const Eigen::Vector3d & piece);

  /** Length of the boxcar the stall test averages the piece pose over, in ticks.
   *
   * One full search period of the widest circle anyone is drawing. Nobody searching means
   * there is nothing to cancel, and the average collapses to the raw sample. */
  size_t stallWindowTicks() const;
  /** The dither to add to one partner's reference, given what that person is feeling.
   *
   * `resistance` is the partner's own spring stretch against the undithered plan, in N. */
  Eigen::Vector3d searchOffset(Partner & partner, double resistance);

  // Is the piece sitting still? Shared, because it is a property of the piece and not of
  // either person. Reset by startRecording, which is the only thing that may seed them.
  Eigen::Vector3d stuckOrigin_ = Eigen::Vector3d::Zero();
  double lastMotionTime_ = 0.0;
  bool stuckSeeded_ = false;
  bool pieceStuck_ = false;

  // The stall test reads the piece pose averaged over one search period, never the raw pose.
  // The dither is a circle, so a boxcar of exactly one period integrates it to zero while
  // genuine drift survives. Without this the search cancels its own gate: a circle wide enough
  // to move the piece past stallMotion clears pieceStuck_, which collapses the envelope before
  // it reaches full amplitude. Measured, that capped the achieved radius near 10 mm however
  // much was asked for -- 8 mm arrived at 97 % of command, 16 mm at 59 %, 24 mm at 41 % -- and
  // made a stronger wiggle un-testable.
  std::deque<Eigen::Vector3d> stuckWindow_;
  Eigen::Vector3d stuckWindowSum_ = Eigen::Vector3d::Zero();

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
  /** Episodes an earlier launch of this cell already ran, when this one resumes it.
   *
   * A solver failure kills the simulator and every episode of the cell that had not run yet,
   * because one process drives all of them. Resuming means launching again for the remainder,
   * and the remainder has to be *the* remainder: the RNG is seeded once and consumed one draw
   * per episode, so a fresh launch would replay the episodes that already exist -- including
   * the one that killed it. This burns that many draws before the first episode, so episode
   * k+1 gets exactly the noise it would have had in an uninterrupted run, and tells the
   * controller to number from k+1.
   *
   * `k` is the count of *completed* episodes, so the episode the failure happened in is
   * re-run rather than skipped, with the draw it already had. That is deliberate: episode k
   * of a cell has to be the same seeded episode in every arm, and skipping one would shift
   * every later draw against the arm being compared. Verified end to end -- a cell that died
   * in episode 2 of 3 was relaunched and its episode 2 came back with the same gain scale,
   * 1.0119089535461756, that the truncated one carried.
   *
   * Consuming draws rather than re-seeding is deliberate: seed-matching is what makes every
   * comparison in this project paired, and a resumed cell has to stay matched to the same cell
   * in the arm it is being compared against.
   *
   * One caveat that is a property of resuming and not of this counter: episodes within a
   * launch start from wherever the previous one left the arm, while a resumed launch starts
   * from the home pose. A resumed episode is therefore not bit-identical to the one an
   * uninterrupted run would have produced -- only identically *seeded*.
   */
  int resumeAfter_ = 0;
  bool resumeApplied_ = false;
  double episodeTime_ = 0.0;    // seconds since the current episode started
  double t_ = 0.0;              // plugin clock, seconds since reset
  double dt_ = 0.001;
};

} // namespace collab
