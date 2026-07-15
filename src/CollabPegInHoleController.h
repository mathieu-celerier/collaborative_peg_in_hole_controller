#pragma once

#include <mc_control/fsm/Controller.h>
#include <mc_tasks/CompliantEndEffectorTask.h>
#include <mc_tasks/CompliantPostureTask.h>

#include <RBDyn/Jacobian.h>

#include "api.h"

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Residual gains fed to the ExternalForcesEstimator plugin. The residual is the
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
 *    residual of the ExternalForcesEstimator plugin.
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

  /** Episode markers for dataset segmentation. */
  void startEpisode();
  void markEpisode(const std::string & label);
  void stopEpisode();

  // Frames / sensors (configurable per robot module).
  std::string tool_frame;
  std::string ft_sensor;
  std::string insertion_frame;

  // Explicit-compliance tasks.
  std::shared_ptr<mc_tasks::CompliantPostureTask> compPostureTask;
  std::shared_ptr<mc_tasks::CompliantEndEffectorTask> compEETask;

  // Home posture (per-robot, from config).
  std::map<std::string, std::vector<double>> postureHome;

private:
  void addGui();
  void addLog();
  // Toggle the ExternalForcesEstimator feedback on/off if the plugin is loaded.
  void setEstimatorActive(bool active);
  bool estimatorAvailable() const;
  // Read an Eigen vector from the datastore if present, else a zero vector.
  Eigen::VectorXd datastoreVector(const std::string & key, Eigen::Index size) const;
  // Running-average helper (bounded window), cf. box_carying_demo_controller.
  static void updateSlidingAverage(std::deque<double> & history, double & sum, double value, std::size_t maxSamples);
  /** Task-space external wrench at tool_frame, reconstructed from the estimator's full
   * external joint torques (residual fused with the F/T sensor). This is the only signal
   * that sees BOTH people: person A pushing the arm (residual only) and person B pushing
   * the plate (F/T). Mirrors the plugin's own J^T pseudo-inverse conversion, cf.
   * mc_residual_estimation FixedBaseEstimatorBackend::computeFixedBaseForceFusion. */
  sva::ForceVecd computeExternalWrench();

  mc_rtc::Configuration config_;

  std::string requestedState_ = "Initial";

  bool postureCompliant_ = true;
  bool eeInSolver_ = false;

  // Tuning parameters (all overridable from the per-robot config block).
  double residualGain_ = COLLAB_HIGH_RESIDUAL_GAIN; // ExternalForcesEstimator residual gain
  double eeTaskWeight_ = 10000.0; // end-effector task weight
  double postureCompliantDamping_ = 2.0; // posture damping in the compliant state
  double initialPostureStiffness_ = 0.5; // posture stiffness in the initial state

  // Dual-compliance (force-triggered EE freeing) parameters and state.
  double dualForceEnter_ = 5.0; // wrench norm (N / Nm) above which the EE frees up
  double dualForceExit_ = 2.0; // wrench norm below which the EE re-stiffens
  double eeHoldStiffness_ = 400.0; // stiffness when holding position
  double eeHoldDamping_ = 40.0; // damping when holding position (ramp target)
  double eeFreeDamping_ = 15.0; // damping when free (stiffness 0)
  bool dualFree_ = false; // currently in the free (low-stiffness) mode
  double currentWrenchNorm_ = 0.0; // raw trigger input (norm of the selected source)

  // Trigger source: "external" (full external wrench, both people) or "ft" (F/T sensor only).
  std::string triggerSource_ = "external";
  std::unique_ptr<rbd::Jacobian> extWrenchJac_; // Jacobian at tool_frame, built once
  sva::ForceVecd externalWrench_ = sva::ForceVecd::Zero(); // last reconstructed external wrench
  bool warnedExtTorqueSize_ = false;

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
  double m_ = 1.8;
  double lambda_ = 70.0;
  double xsiOff_ = 0.0;

  // Episode bookkeeping.
  int episodeId_ = 0;
  std::string episodeLabel_ = "idle";
  bool recording_ = false;
};
