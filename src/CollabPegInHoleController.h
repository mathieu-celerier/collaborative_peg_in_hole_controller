#pragma once

#include <mc_control/fsm/Controller.h>
#include <mc_tasks/CompliantEndEffectorTask.h>
#include <mc_tasks/CompliantPostureTask.h>
#include <mc_tasks/VectorOrientationTask.h>

#include <RBDyn/Jacobian.h>

#include "IntentReplay.h"
#include "PolicyPartner.h"
#include "RemotePolicy.h"
#include "api.h"

#include <array>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Residual gains fed to the ExternalForcesObserver. The residual is the
// only observable channel for the person manipulating the arm *behind* the F/T
// sensor (person A); a high gain makes hand-guiding responsive.
#define COLLAB_HIGH_RESIDUAL_GAIN 20.0
#define COLLAB_LOW_RESIDUAL_GAIN 0.5

/** Collaborative peg-in-hole data-collection controller.
 *
 * Two-state FSM built on mc_rtc's explicit compliance & safety framework:
 *  - Initial:   stiff, position-controlled homing / setup (estimator feedback off).
 *  - Compliant: fully compliant torque control so two people can hand-guide the
 *               robot simultaneously for a peg-in-hole insertion demonstration.
 *
 * The controller separates the two human interaction channels so both can be
 * recorded for imitation learning:
 *  - Person A (arm, before the F/T sensor): observed via the joint-torque
 *    residual of the ExternalForcesObserver.
 *  - Person B (peg piece, after the F/T sensor): observed via the F/T sensor
 *    wrench (EEForceSensor).
 */
struct CollabPegInHoleController_DLLAPI CollabPegInHoleController : public mc_control::fsm::Controller
{
  CollabPegInHoleController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config);

  bool run() override;

  void reset(const mc_control::ControllerResetData & reset_data) override;

  /** Enter the stiff, position-controlled homing state. */
  void switchToInitialState();
  /** Enter the fully-compliant, torque-controlled collaboration state. */
  void switchToCompliantState();

  /** FSM coordination: which mode the operator has requested. */
  void requestState(const std::string & state);
  const std::string & requestedState() const;

  /** Live posture (null-space) compliance toggle for person A hand-guiding. */
  void postureCompliance(bool compliant);
  bool postureCompliance() const;

  /** Dual-compliance manager: while in the Compliant state, watch the F/T wrench
   * and drop the end-effector stiffness to ~0 when a force is applied (so the
   * plate can be moved freely to perform the insertion), restoring stiffness to
   * hold position once the force is released (hysteresis). Modelled on the
   * monodzukuri / box demo dual-compliance behaviour. */
  void updateDualCompliance();
  /** Reset the end-effector task to its stiff "hold" configuration. */
  void holdEndEffector();
  /** Angle between the tool's up axis and world vertical, in radians. */
  double toolTilt() const;
  /** Apply the end-effector compliance for the current free/hold state.
   *
   * Compliance is per-axis, and translation and orientation are not treated alike: see
   * `eeOrientationCompliant_`. */
  void applyEECompliance(bool free);

  /** Episode markers for dataset segmentation.
   *
   * An episode is one recorded demonstration, delimited inside the session log by the
   * `collab_recording` marker. `startEpisode()` opens it, `markEpisode()` attaches a
   * *sticky* outcome that covers the whole episode, `stopEpisode()` closes it. The extractor
   * (`scripts/extract_episodes.py`) segments on the `collab_recording` edges and reads
   * `collab_episode_outcome` to filter successful demonstrations. */
  /** Continue numbering episodes after `id`, for a run that resumes a cell.
   *
   * A solver failure ends the simulator, and with it every episode of that cell that had not
   * run yet -- one process drives all of them. Resuming means launching again for the
   * remainder, and the episodes it records must not reuse the ids the first launch already
   * wrote: `extract_episodes.py` names files from `collab_episode_id`, so a second launch
   * starting at 1 would overwrite the episodes that survived.
   *
   * Called by SimHumanPartner from its `resume_after`, before the first `startEpisode()`.
   */
  void resumeEpisodesAt(int id);

  void startEpisode();
  void markEpisode(const std::string & label);
  void stopEpisode();

  /** Evaluate the behaviour-cloned partner and inject its force at the task level.
   *
   * Called once per tick from run(), ahead of the FSM, so `policyWrenchNorm_` is fresh when
   * updateDualCompliance() reads it in the same tick.
   *
   * The force does NOT act on the robot. It is converted to a task-frame acceleration and
   * handed to compEETask->refAccel(), which is where the demonstrating partner's force
   * arrived from: CompliantEndEffectorTask::update() computes
   *
   *     disturbedAccel_ = refAccel_ + Gamma * (J * H^-1 * tau_ext)
   *
   * so `refAccel_` is the free feed-forward slot beside the estimator's disturbance term.
   * Two consequences that are easy to get wrong and silent when you do:
   *
   * - **Gamma is not applied to refAccel_.** It scales the disturbance term only, so this
   *   applies it by hand. Skip that and the policy drives the angular axes even when
   *   `ee_orientation_compliant` is false, quietly undoing the choice to keep the plate
   *   level while the translation floats.
   * - **The route is through joint space.** J_A^T at person A's contact point, then H^-1,
   *   then the tool Jacobian. That is what makes a pure force at an offset point produce
   *   the right angular term at the tool frame; a direct task-space conversion loses it.
   */
  void updatePolicyPartner();

  /** The observation the policy takes, in cpih.policy.features order.
   *
   * Mirrors `cpih.policy.features.FEATURES` exactly, and `policyFeatureNames()` is checked
   * against the names in the policy file at load time. Note that `q`/`dq` are the *encoder*
   * channels, because the dataset's obs/q and obs/dq are qIn and alphaIn -- feeding the
   * control robot's mbc here would be a different signal wearing the same name.
   */
  Eigen::VectorXd policyFeatures() const;
  static std::vector<std::string> policyFeatureNames(bool withPhase);

  /** The observation a RoboManipBaselines policy takes: 29 values, RMB's own layout.
   *
   * `measured_joint_pos(7) | measured_joint_vel(7) | measured_eef_pose9(9) |
   * measured_eef_wrench(6)`, which is what `scripts/export_rmb.py` wrote into the files it
   * was trained on. Two details carry the whole risk, and neither would raise:
   *
   * - the pose is **translation first**, then the first two **rows** of the rotation matrix
   *   (RMB's `PoseUtils._get_rotation_6d_from_matrix` slices `rot[..., :2, :]`). Columns
   *   instead produce a valid-looking vector that is wrong in six of its entries.
   * - the wrench is **force first**, where mc_rtc's `ForceVecd` is couple first.
   *
   * `scripts/export_onnx.py:pose9_from_pose7` in the learning repo is the tested reference
   * for the first, and it is the same quantity as `collab_ft_wrench_world` for the second.
   */
  Eigen::VectorXd policyObservation() const;

  /** Six numbers describing the partner the policy is currently working with.
   *
   * MUST stay numerically identical to `cpih.policy.context.partner_context` in the learning
   * repo, which computes the same thing at export time; the constants below are that module's,
   * and `scripts/check_context_agreement.py` compares the two on a logged episode. A mismatch
   * here is a silent train/serve skew in the input the policy leans on hardest, and nothing
   * would raise -- the widths would still agree.
   *
   * Why it exists: the six leader x effort regimes differ only in the partners' GAINS, and the
   * 29-value observation names neither leader nor effort. Measured on train9, the leader is
   * recoverable from a 1 s window of wrench and tool motion at 40 %, from 10 s at 83 %, and
   * from this vector at a single instant at up to 90 % -- but `n_obs_steps: 2` is 20 ms of
   * history, so without this the policy cannot tell the regimes apart and learns an average
   * of all six.
   *
   * Layout: f_mean [N] | f_dev [N] | v_mean [m/s] | k [kN/m] | c [kNs/m] | age [s, capped].
   */
  Eigen::Matrix<double, 6, 1> partnerContext() const { return partnerContext_; }
  void updatePartnerContext();
  void resetPartnerContext();

  /** Position and velocity of a point rigidly attached to a body, world frame.
   *
   * Mirrors `SimHumanPartner::aimedPoint`. The angular cross term is not optional: the hole
   * sits well away from the wrist axis, so the offset point and the body origin have
   * different velocities as soon as the wrist turns.
   */
  static void aimedPoint(const mc_rbdyn::Robot & robot,
                         const std::string & body,
                         const Eigen::Vector3d & point,
                         Eigen::Vector3d & position,
                         Eigen::Vector3d & velocity);

  /** The force an intent-predicting policy produces, through the impedance of the partner it
   * replaces.
   *
   * The policy emits where person A *meant* their hand to be. The force follows from the same
   * spring the simulated partner used -- `f = kp (xRef - x) + kd (vRef - v)`, saturated at
   * that partner's `max_force` -- because reproducing the law is what makes the policy a
   * stand-in for the partner rather than a different actor wearing its name.
   *
   * `vRef` is not predicted: the reference velocity was never an action key. It is a filtered
   * finite difference of the intent instead. Dropping it would leave pure damping, which is a
   * materially different partner during transit, where the demonstrator's `kd vRef` is a
   * large forward term.
   *
   * Split in two because they run at different rates. `updateIntentReference` is called at
   * the policy rate, `intentForce` every control tick: an impedance evaluated only when a new
   * prediction arrives is a constant push between predictions, not a spring, and would not
   * react to the arm moving under it in the 9 ms in between.
   *
   * `policy.max_intent_speed` is applied here, on the way in: what the impedance closes on is
   * the prediction, or the furthest point towards it a hand could have reached in one policy
   * step. See `policyMaxIntentSpeed_`.
   */
  void updateIntentReference(const Eigen::Vector3d & predicted);
  Eigen::Vector3d intentForce();

  // Frames / sensors (configurable per robot module).
  std::string tool_frame;
  std::string ft_sensor;
  std::string insertion_frame;

  // --- Behaviour-cloned partner (replaces person A) ---
  /** Either a `PolicyPartner` reading the YAML the learning repo writes, or an `OnnxBackend`
   * running a RoboManipBaselines checkpoint. Chosen by the file's extension at load time;
   * everything below this line is shared between them. */
  std::unique_ptr<collab::PolicyBackend> policy_;
  /** What the policy's output means: `force` (a world-frame force, applied directly) or
   * `intent` (a point person A's plan aimed at, turned into a force by `intentForce`). Read
   * out of the policy file, never configured -- the two need completely different handling
   * and a config that can disagree with the file will one day disagree with it. */
  std::string policyOutputKind_ = "force";
  bool policyEnabled_ = false;
  std::string policyPath_;
  /** Where the peg is, world frame. Scene knowledge the controller has no other source for.
   * In simulation the plugin knows it and logs it; here it is configured, which is also how
   * it would arrive on the real robot (from calibration or perception). */
  Eigen::Vector3d pegTarget_ = Eigen::Vector3d::Zero();
  bool havePegTarget_ = false;
  /** Person A's contact point: the body they pushed on and where. Must match the partner
   * entry the demonstrations were generated with, or the Jacobian is the wrong one. */
  std::string policyBody_ = "end_effector_link";
  Eigen::Vector3d policyPoint_ = Eigen::Vector3d::Zero();
  /** The policy is trained at the dataset rate (100 Hz), not the control rate. Evaluate it
   * every `policyDecimation_` ticks and hold the force between: it matches the rate the map
   * was fitted at, and costs a tenth of the inference. */
  int policyDecimation_ = 10;
  int policyTick_ = 0;
  Eigen::Vector3d policyForce_ = Eigen::Vector3d::Zero();
  Eigen::Vector6d policyAccel_ = Eigen::Vector6d::Zero();
  /** The policy's contribution to the free/hold trigger. The policy stays out of
   * externalTorques() -- that is the estimate of what the world does, and the policy is not
   * the world -- but the trigger has to see it, or the end-effector re-latches earlier than
   * it did in the demonstrations, where person A's real force fed the same trigger. */
  double policyWrenchNorm_ = 0.0;
  /** Hole position one policy step ago, for the finite difference. The features use a
   * difference over the policy interval, not frame().velocity(): the dataset's velocity is a
   * difference at 100 Hz, and a feed-forward term is not the place to spend that mismatch. */
  Eigen::Vector3d lastHole_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d holeVelocity_ = Eigen::Vector3d::Zero();
  bool haveLastHole_ = false;

  // --- The intent path: policy -> impedance -> force (policyOutputKind_ == "intent") ---
  /** Person A's impedance, copied from the partner this policy replaces. The *nominal* gains,
   * not a per-episode `gain_sigma` draw: there is no person A at evaluation time to draw for,
   * and the demonstrations' perturbations were of the partner, not of the plan. */
  Eigen::Vector3d policyStiffness_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d policyDamping_ = Eigen::Vector3d::Zero();
  double policyMaxForce_ = 0.0;
  /** The point the impedance closes on -- where the replaced partner aimed, which is the
   * point it pushed through. A spring anchored anywhere else is blind to the motion it
   * causes; see SimHumanPartner's Partner docs. */
  /** Whether `measured_partner_context` is appended to the observation, making it 35 wide.
   *
   * Off by default, because every checkpoint trained before this expects exactly 29 and the
   * remote server rejects a mismatched width with rows=0 -- which the controller treats as a
   * dead policy. Turn it on only for a checkpoint trained with the key in --state_keys. */
  bool policyPartnerContext_ = false;
  Eigen::Matrix<double, 6, 1> partnerContext_ = Eigen::Matrix<double, 6, 1>::Zero();
  /** Estimator state. Reset at every episode start, exactly as the exporter starts each
   * episode's sequence from its own first sample. */
  Eigen::Vector3d ctxPosBar_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d ctxForceBar_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d ctxPrevPos_ = Eigen::Vector3d::Zero();
  double ctxFMagBar_ = 0.0;
  double ctxFDevBar_ = 0.0;
  double ctxVMagBar_ = 0.0;
  double ctxSxx_ = 0.0, ctxSxv_ = 0.0, ctxSvv_ = 0.0, ctxSxf_ = 0.0, ctxSvf_ = 0.0;
  bool ctxSeeded_ = false;
  double ctxAge_ = 0.0;
  /** The estimator advances at the DATASET's rate (100 Hz), not the controller's (1 kHz).
   *
   * It is a discrete recursion, so its output depends on the step it is iterated at: run at
   * 1 kHz against a 100 Hz training set it produces a different sequence from the same
   * physics, which is a train/serve skew that no width check would catch. Measured before
   * this: f_dev differed by 0.17 N and k by 0.08 kN/m on the same episode.
   *
   * The force is accumulated and averaged over the ticks in between, because that is what the
   * extractor does to it (`decimate(..., smooth=True)` is a moving average, then a stride).
   * The extractor's average is CENTRED and this one cannot be -- a controller has no future
   * samples -- so the two differ by that half-window of lag, and nothing can remove it. That
   * is the residual the agreement check tolerates; it is a 5 ms phase shift on a 1 Hz signal. */
  /** Raw ticks since the episode began. The estimator steps on a schedule derived from it. */
  int ctxTick_ = 0;
  /** The last 11 forces and 6 positions, so that a step can reproduce the extractor's window
   * exactly. 11 is the extractor's box width (it rounds 10 up to odd); 6 is how far back the
   * strided position sample sits when the window closes. */
  std::array<Eigen::Vector3d, 11> ctxForceRing_;
  std::array<Eigen::Vector3d, 6> ctxPosRing_;
  std::string policyRefBody_;
  Eigen::Vector3d policyRefPoint_ = Eigen::Vector3d::Zero();
  /** First-order filter on the differentiated intent, as a per-policy-step blend factor.
   * 1.0 is the raw difference. The policy's prediction error lands directly in this
   * derivative, so some smoothing is not optional. */
  double policyVelocityFilter_ = 0.2;
  /** Ceiling on how fast the predicted intent point may travel, in m/s. 0 disables it, which
   * is what an unset config gets, so this changes nothing until it is asked for.
   *
   * The demonstrating partner's own intent is a min-jerk plan and moves at most 0.181 m/s
   * (measured over dataset-train8's 540 episodes, p99 0.124). A prediction has no such
   * bound: a chunk that lands far from the last one moves the spring's anchor by that whole
   * distance in one policy step, and the impedance answers with a force whose only limit is
   * `max_force`. Limiting the step keeps the direction the policy asked for and spends more
   * ticks getting there -- which is what the hand it is imitating would have done. */
  double policyMaxIntentSpeed_ = 0.0;
  /** Whether the policy predicts a *displacement from its own hand* rather than a world
   * point. False is the absolute action space every policy before this was trained in.
   *
   * The impedance consumes `reference - hand`, never the reference alone, so that difference
   * is what the demonstrations actually contain: in the cells where person A leads -- the ones
   * the policy fails in -- it is 6.2 mm median and 60.9 mm at its largest, where a drifted
   * policy commanded 92.9 mm median and 307 mm, ~50 N against a hand that never exceeded 37.3.
   *
   * The gain is distributional, not a hard bound, and the difference matters: the two `limits`
   * normalisation boxes have near-identical diagonals (415 mm absolute, 408 relative), so
   * `clip_sample` bounds neither more tightly than the other. What changes is what a typical
   * output means -- in this space commanding 300 mm is far out in the policy's own output
   * distribution, where an absolute world point 300 mm from the hand is unremarkable. The
   * reference is rebuilt here as `hand + prediction`. Everything below that -- the rate limit, the
   * impedance, the saturation -- is unchanged, so the two action spaces differ in exactly one
   * line and stay comparable.
   *
   * `max_intent_speed` is applied to the prediction, above the reconstruction, so that it
   * bounds what the policy asked for rather than the arm's own motion -- see
   * updateIntentReference() for what limiting the rebuilt point did instead. */
  bool policyRelativeIntent_ = false;
  Eigen::Vector3d policyIntent_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d lastIntent_ = Eigen::Vector3d::Zero();
  /** The previous prediction, in the policy's own action space -- what `max_intent_speed` is
   * measured against. Equal to `lastIntent_` for an absolute policy; a displacement for a
   * relative one, so the limiter bounds the prediction rather than the arm's own motion. */
  Eigen::Vector3d lastCommand_ = Eigen::Vector3d::Zero();
  bool haveLastCommand_ = false;
  Eigen::Vector3d policyRefVelocity_ = Eigen::Vector3d::Zero();
  bool haveLastIntent_ = false;
  /** Whether the impedance hit `max_force` this tick. Logged: a policy that saturates
   * constantly is aiming somewhere the arm cannot follow, which reads as a control problem
   * and is a prediction problem. */
  bool policySaturated_ = false;
  /** Whether the rate limit above shortened the intent's step this tick. Logged for the same
   * reason as `policySaturated_`: a limiter that never fires explains nothing about a run,
   * and one that fires on every tick has replaced the policy with a ramp. */
  bool policyIntentLimited_ = false;
  /** The recorded intent replayed in place of a prediction, when `policy.replay` is set.
   *
   * Not a `PolicyBackend`: it observes nothing. It substitutes for exactly one thing -- the
   * predicted point -- so that the impedance and the `refAccel_` injection below it can be
   * exercised against a reference already known to have inserted. See IntentReplay.h. */
  std::unique_ptr<collab::IntentReplay> replay_;
  std::string policyReplayPath_;
  /** A chunked policy served by another process, when `policy.remote_port` is set.
   *
   * The third and last way an intent is produced, beside `policy_` (a forward pass here) and
   * `replay_` (a recording). Exactly one of the three is ever non-null. It exists because a
   * diffusion policy costs 22.6 ms per answer and the loop ticks every 1 ms; see
   * RemotePolicy.h. */
  std::unique_ptr<collab::RemotePolicy> remote_;
  std::string policyRemoteHost_ = "127.0.0.1";
  int policyRemotePort_ = 0;
  /** Whether this episode's replay anchor has been taken yet. Taken once, at the first
   * recording tick, because that is the moment SimHumanPartner rebases its own plan. */
  bool haveReplayAnchor_ = false;
  /** Log the 29-value RMB observation every tick. Off by default: it is log volume no run
   * needs, and it exists for one job -- recording a *demonstration* with it set, so the
   * vector the controller assembles can be compared frame for frame against the one
   * `scripts/export_rmb.py` wrote from the same episode. */
  bool logPolicyObservation_ = false;
  /** Jacobian at person A's contact point, built once. Constructing an rbd::Jacobian
   * allocates, and this runs on the control thread. */
  std::unique_ptr<rbd::Jacobian> policyJac_;

  // Explicit-compliance tasks.
  std::shared_ptr<mc_tasks::CompliantPostureTask> compPostureTask;
  std::shared_ptr<mc_tasks::CompliantEndEffectorTask> compEETask;

  /** Keeps the tool pointing up while leaving rotation about the vertical free.
   *
   * A VectorOrientationTask constrains two rotational degrees of freedom, not three: it
   * aligns one frame-fixed vector with a world direction and says nothing about the spin
   * around it. That is exactly the split this task needs -- the plate must stay level for
   * the peg to enter the hole, but which way it faces around the vertical is the dyad's
   * business. Defending the full orientation instead is what made the QP report its
   * constraints inconsistent; see `eeOrientationCompliant_`. Modelled on the up-vector task
   * in box_carying_demo_controller's Compliance state. */
  std::shared_ptr<mc_tasks::VectorOrientationTask> eeUpVectorTask;

  /** Still riding the homing trajectory. The Initial state must not hand over while true:
   * the episode driver asks for Compliant as soon as it can, and leaving mid-ride puts the
   * arm into the compliant state from a configuration that is neither home nor the one the
   * gains were checked at. */
  bool homing() const { return homing_; }

  // Home posture (per-robot, from config).
  std::map<std::string, std::vector<double>> postureHome;

private:
  void addGui();
  void addLog();
  /** Swap in mc_kinova's simulation F/T calibration when running under mc_mujoco.
   *
   * The robot's calibration folds in the Bota's force gain and its home-pose tare; the
   * simulated sensor is exact and untared, so applied there it compensates almost nothing.
   * No-op on hardware. Called from reset(), which is late enough for mc_mujoco's datastore
   * calls to exist. */
  void applySimForceSensorCalibration();

  /** Shape the estimator's external joint torques on the way into the solver: low-pass, then
   * saturate, on both robot() and realRobot(). Logged and recorded signals are read before this
   * runs, so it changes what the QP acts on and nothing else. */
  void conditionExternalTorques();
  // Toggle the ExternalForcesObserver feedback on/off if the observer is in the pipeline.
  void setEstimatorActive(bool active);
  bool estimatorAvailable() const;
  // Read an Eigen vector from the datastore if present, else a zero vector.
  Eigen::VectorXd datastoreVector(const std::string & key, Eigen::Index size) const;
  // Running-average helper (bounded window), cf. box_carying_demo_controller.
  static void updateSlidingAverage(std::deque<double> & history, double & sum, double value, std::size_t maxSamples);
  /** Task-space wrench at tool_frame corresponding to a vector of external joint torques,
   * by least-squares inversion of tau = J^T w, expressed in world. */
  sva::ForceVecd wrenchFromTorques(const Eigen::VectorXd & tau) const;
  /** Task-space external wrench at tool_frame, from the observer's full external joint
   * torques (residual + F/T projection). This is the only signal that sees BOTH people:
   * person A pushing the arm and person B pushing the plate. It drives the free/hold
   * trigger and the compliant task's feed-forward. */
  sva::ForceVecd computeExternalWrench();
  /** Same conversion applied to the momentum residual alone - person A, with person B already
   * subtracted by the observer's known-input fusion. Logged only: it is what tells you which
   * of the two people an angular term that runs away belongs to. */
  sva::ForceVecd computeResidualWrench() const;

  mc_rtc::Configuration config_;

  std::string requestedState_ = "Initial";

  bool postureCompliant_ = true;
  bool eeInSolver_ = false;

  // Tuning parameters (all overridable from the per-robot config block).
  double residualGain_ = COLLAB_HIGH_RESIDUAL_GAIN; // ExternalForcesObserver residual gain
  double eeTaskWeight_ = 10000.0; // end-effector task weight
  double postureCompliantDamping_ = 2.0; // posture damping in the compliant state
  /** Posture stiffness in the Compliant state. Zero leaves the arm's internal configuration
   * held by damping alone, which is enough while nothing else acts on the rotation but not
   * once a task defends the plate's tilt: two angular objectives on a limp arm is what makes
   * the QP report its constraints inconsistent. box_carying_demo_controller anchors its null
   * space the same way, with posture stiffness 10 against an end-effector task of the same
   * weight as ours. */
  double postureCompliantStiffness_ = 0.0;
  double initialPostureStiffness_ = 0.5; // posture stiffness if homing is disabled

  /** Homing: a joint-space minimum-jerk ride to `postureHome`, not a step target.
   *
   * Homing used to be one step -- set the target to the home posture and let a soft posture
   * task walk there -- and the arm did not arrive: at a stiffness low enough that the step
   * does not fling it, the residual error never closes inside the time the episode driver
   * waits, so episode 2 started 100 mm below where episode 1 did and the dataset recorded the
   * drift as if it were part of the task. Raising the stiffness instead trades that for a
   * lunge, because a step target asks for infinite velocity and only the gain limits it.
   *
   * Interpolating the target removes the choice: the reference itself has bounded velocity
   * and acceleration (zero at both ends), so the stiffness only has to *track* it. That is
   * what lets it be an order of magnitude higher than the step version could afford. */
  double homingDuration_ = 2.5;          // s; <= 0 restores the old step behaviour
  double homingPostureStiffness_ = 10.0; // stiffness while riding the trajectory
  bool homing_ = false;
  double homingTime_ = 0.0;
  std::map<std::string, std::vector<double>> homingStart_;
  /** Latch the current posture and start the ride to `postureHome`. */
  void startHoming();
  /** Advance the ride by one control cycle. No-op unless homing. */
  void updateHoming();
  /** Drop the homing feed-forward, so no state after homing inherits a reference velocity. */
  void clearHomingRefs();

  // Dual-compliance (force-triggered EE freeing) parameters and state.
  double dualForceEnter_ = 5.0; // wrench norm (N / Nm) above which the EE frees up
  double dualForceExit_ = 2.0; // wrench norm below which the EE re-stiffens
  double eeHoldStiffness_ = 400.0; // stiffness when holding position
  double eeHoldDamping_ = 40.0; // damping when holding position (ramp target)
  double eeFreeDamping_ = 15.0; // damping when free (stiffness 0)
  /** Whether freeing the end-effector also frees the plate's orientation in the 6d task.
   *
   * The default matches the shipped yaml, which is false. It used to be true, justified by
   * `eeUpVectorTask` holding the tool's up axis vertical so the 6d task could hand the whole
   * rotation to the partners and the plate would still stay level -- but that task is off
   * (`ee_up_vector_enabled: false`, and the measurements behind that are tabulated in
   * etc/CollabPegInHoleController.in.yaml), so the justification no longer applies.
   *
   * False is the setting to run, and it needs conditionExternalTorques()'s low-pass to be
   * stable. Holding the orientation means commanding torque against tau_ext; that torque
   * accelerates the plate; and the plate's inertial reaction is most of what the F/T sensor
   * reports, so the loop closes inside one tick. Measured (19:30 run, 2026-08-24): a steady
   * ~12 N from the operator, flat for seconds, then at the free edge joints 1, 5 and 7 -- the
   * rotations about the vertical, i.e. exactly the defended axes -- ring at ~90 Hz growing ~3x
   * per cycle while the tool travels 1.3 mm and the sensor swings +45 to -77 N; QP infeasible
   * 26 ms later. tauMomentumObserver stayed at ~0.02 N.m throughout, so none of it was person A
   * and none of it was the operator -- which is why this only ever appeared with
   * use_forces_from_ft_sensors: true and looked like the calibration bug.
   *
   * What is settled: defending a pose in the 6d task does not work here either. The full
   * orientation is three constraints, and pinning all three at a posture where joints 1 and 3
   * are nearly parallel makes the QP report its constraints inconsistent. */
  bool eeOrientationCompliant_ = false;
  // The up-vector task. Off by default: every strength that actually holds the tilt takes the
  // QP with it, on an arm whose end-effector is ramped to zero stiffness while free. The
  // measurements are tabulated in etc/CollabPegInHoleController.in.yaml.
  bool eeUpVectorEnabled_ = false;
  /** Weight of the 6d task's orientation sub-task while the up-vector task is active.
   *
   * Two tasks pulling on the same frame's rotation at comparable weights is not a hierarchy,
   * it is a fight, and the QP loses it. Dropping the 6d task's angular weight leaves the
   * up-vector task as the only angular objective; translation keeps the full weight. */
  double eeOrientationWeight_ = -1.0; // < 0 : keep eeTaskWeight_
  double eeUpVectorStiffness_ = 100.0;
  double eeUpVectorWeight_ = 20000.0;
  Eigen::Vector3d eeUpVectorInTool_ = Eigen::Vector3d::UnitZ(); // latched on entering Compliant
  bool dualFree_ = false; // currently in the free (low-stiffness) mode
  double currentWrenchNorm_ = 0.0; // raw trigger input (norm of the selected source)

  // Trigger source: "external" (full external wrench, both people) or "ft" (F/T sensor only).
  std::string triggerSource_ = "external";
  std::unique_ptr<rbd::Jacobian> extWrenchJac_; // Jacobian at tool_frame, built once
  sva::ForceVecd externalWrench_ = sva::ForceVecd::Zero(); // last reconstructed external wrench
  mutable bool warnedExtTorqueSize_ = false; // set from the const wrenchFromTorques()

  // Anti-chatter smoothing of the free/hold trigger, cf. box_carying_demo_controller.
  double wrenchFilterTime_ = 0.1; // first-order low-pass time constant on the wrench norm
  double triggerAveragingWindow_ = 0.0; // moving-average window (s); 0 -> off
  double gainTransitionTime_ = 0.2; // exponential ramp time constant for stiffness/damping
  double filteredWrenchNorm_ = 0.0; // low-pass filtered wrench norm
  double triggerWrenchNorm_ = 0.0; // moving-averaged (fully smoothed) trigger signal
  bool wrenchFilterInit_ = false; // filter seeded with the first sample
  std::deque<double> triggerHistory_; // moving-average ring buffer
  double triggerSum_ = 0.0; // running sum of triggerHistory_
  double currentEEStiffness_ = 400.0; // ramped end-effector stiffness
  double currentEEDamping_ = 40.0; // ramped end-effector damping
  // Warn-once flags so diagnostics don't spam every control tick.
  bool warnedNoFtSensor_ = false;
  bool warnedNoEstimator_ = false;
  bool warnedZeroWrench_ = false;
  int zeroWrenchTicks_ = 0; // consecutive ticks with an exactly-zero wrench

  // Velocity-damper (safety) parameters, cf. monodzukuri_controller.
  // External-torque saturation (see conditionExternalTorques()): the estimate is fed to the QP as a
  // known term of the dynamics equality, so one larger than the motors can oppose empties the
  // constraint set. Bound it to this fraction of each joint's torque limit; <= 0 disables.
  double extTorqueRatio_ = 0.7;
  Eigen::VectorXd extTorqueLimit_; // per-dof min(|tl|, |tu|), built once
  // Cutoff of the first-order low-pass on tau_ext into the solver. The sensor->estimate->
  // command->motion loop closes within one tick through the payload's inertia and rings at
  // ~90 Hz; a human guiding the arm lives below ~5 Hz. <= 0 disables (and brings the ring back).
  double extTorqueFilterHz_ = 10.0;
  Eigen::VectorXd extTorqueFiltered_;
  int extTorqueClampedDofs_ = 0;
  bool warnedExtTorqueClamp_ = false;

  double m_ = 1.8;
  double lambda_ = 70.0;
  double xsiOff_ = 0.0;

  // Episode bookkeeping.
  int episodeId_ = 0;
  std::string episodeLabel_ = "idle"; // live label ("idle" / "recording")
  // Outcome of the current episode ("none" until marked). Sticky: only startEpisode() clears it,
  // so the label covers every sample of the episode and survives stopEpisode().
  std::string episodeOutcome_ = "none";
  bool recording_ = false;
  int episodeTick_ = -1; // samples since the episode started, -1 while idle
};
