#include "CollabPegInHoleController.h"

#include <mc_rtc/gui/Button.h>
#include <mc_rtc/gui/Checkbox.h>
#include <mc_rtc/gui/Label.h>
#include <mc_solver/DynamicsConstraint.h>

#include <algorithm>
#include <cmath>

namespace
{
// Build a posture-target map {joint_1: [q1], ...} from a flat vector of joint angles.
std::map<std::string, std::vector<double>> makePostureMap(const std::vector<double> & q)
{
  std::map<std::string, std::vector<double>> target;
  for(size_t i = 0; i < q.size(); ++i)
  {
    target["joint_" + std::to_string(i + 1)] = {q[i]};
  }
  return target;
}
} // namespace

CollabPegInHoleController::CollabPegInHoleController(mc_rbdyn::RobotModulePtr rm,
                                                     double dt,
                                                     const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  config_ = config;

  // Frames / sensor names, overridable per robot module.
  tool_frame = "FT_sensor_wrench";
  ft_sensor = "EEForceSensor";
  insertion_frame = "tool";
  std::vector<double> initial_posture = {0.0, 0.262, 3.14, -2.269, 0.0, 0.96, 1.57};
  if(config.has(robot().module().name))
  {
    auto rconfig = config(robot().module().name);
    rconfig("tool_frame", tool_frame);
    rconfig("ft_sensor", ft_sensor);
    rconfig("insertion_frame", insertion_frame);
    rconfig("initial_posture", initial_posture);
    rconfig("dual_force_enter", dualForceEnter_);
    rconfig("dual_force_exit", dualForceExit_);
    rconfig("ee_hold_stiffness", eeHoldStiffness_);
    rconfig("ee_hold_damping", eeHoldDamping_);
    rconfig("ee_free_damping", eeFreeDamping_);
    rconfig("ee_task_weight", eeTaskWeight_);
    rconfig("wrench_filter_time", wrenchFilterTime_);
    rconfig("trigger_averaging_window", triggerAveragingWindow_);
    rconfig("gain_transition_time", gainTransitionTime_);
    rconfig("trigger_source", triggerSource_);
    rconfig("residual_gain", residualGain_);
    rconfig("posture_compliant_damping", postureCompliantDamping_);
    rconfig("initial_posture_stiffness", initialPostureStiffness_);
    rconfig("vel_damper_m", m_);
    rconfig("vel_damper_lambda", lambda_);
  }
  postureHome = makePostureMap(initial_posture);
  mc_rtc::log::info("[CollabPegInHoleController] tool_frame={}, ft_sensor={}, insertion_frame={}", tool_frame,
                    ft_sensor, insertion_frame);

  // Diagnostics: surface configuration / hardware mismatches at load time so a
  // silently non-working dual compliance is easy to spot.
  if(!robot().hasForceSensor(ft_sensor))
  {
    std::string available;
    for(const auto & fs : robot().forceSensors())
    {
      available += " " + fs.name();
    }
    mc_rtc::log::warning("[CollabPegInHoleController] F/T sensor '{}' not found on robot '{}' - dual compliance will "
                         "be disabled. Available force sensors:{}",
                         ft_sensor, robot().name(), available.empty() ? " <none>" : available);
  }
  if(!robot().hasFrame(tool_frame))
  {
    mc_rtc::log::warning("[CollabPegInHoleController] tool_frame '{}' not found on robot '{}'", tool_frame,
                         robot().name());
  }
  if(!robot().hasFrame(insertion_frame))
  {
    mc_rtc::log::warning("[CollabPegInHoleController] insertion_frame '{}' not found on robot '{}'", insertion_frame,
                         robot().name());
  }
  if(triggerSource_ != "external" && triggerSource_ != "ft")
  {
    mc_rtc::log::warning("[CollabPegInHoleController] unknown trigger_source '{}' (expected 'external' or 'ft'), "
                         "falling back to 'external'",
                         triggerSource_);
    triggerSource_ = "external";
  }
  mc_rtc::log::info("[CollabPegInHoleController] dual-compliance trigger_source={}", triggerSource_);

  // Jacobian at the tool frame, used to map the estimator's external joint torques back to a
  // task-space wrench. Built once: the kinematic structure never changes. Needs a body (not
  // just a frame), so guard rather than let RBDyn throw.
  if(robot().hasBody(tool_frame))
  {
    extWrenchJac_ = std::make_unique<rbd::Jacobian>(robot().mb(), tool_frame);
  }
  else
  {
    mc_rtc::log::warning("[CollabPegInHoleController] tool_frame '{}' is not a body: cannot reconstruct the external "
                         "wrench, dual compliance will fall back to the F/T sensor.",
                         tool_frame);
    triggerSource_ = "ft";
  }

  // Safety framework: closed-loop velocity damper on the dynamics + collisions.
  selfCollisionConstraint->setCollisionsDampers(solver(), {m_, lambda_});
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(new mc_solver::DynamicsConstraint(
      robots(), robot().robotIndex(), solver().dt(), std::array<double, 3>{0.1, 0.01, xsiOff_},
      std::array<double, 2>{m_, lambda_}, 0.9, false, true));
  solver().addConstraintSet(dynamicsConstraint);

  // Replace the default posture task with the compliant one, and expose it so the
  // ExternalForcesEstimator plugin uses it as the posture task.
  solver().removeTask(getPostureTask(robot().name()));
  compPostureTask = std::make_shared<mc_tasks::CompliantPostureTask>(solver(), robot().robotIndex(), 1, 1);
  compPostureTask->reset();
  compPostureTask->stiffness(initialPostureStiffness_);
  compPostureTask->target(postureHome);
  compPostureTask->makeCompliant(false);
  solver().addTask(compPostureTask);
  datastore().make_call("getPostureTask", [this]() -> mc_tasks::PostureTaskPtr { return compPostureTask; });

  // End-effector compliance task (kept out of the solver until the compliant state).
  compEETask =
      std::make_shared<mc_tasks::CompliantEndEffectorTask>(tool_frame, robots(), robot().robotIndex(), 1.0, 10000.0);

  // Control-mode + episode markers on the datastore.
  datastore().make<std::string>("ControlMode", "Position");
  datastore().make<int>("EpisodeId", episodeId_);
  datastore().make<std::string>("EpisodeLabel", episodeLabel_);
  datastore().make<bool>("Recording", recording_);

  addGui();
  addLog();

  mc_rtc::log::success("[CollabPegInHoleController] init done");
}

bool CollabPegInHoleController::run()
{
  // Feedback type follows the control mode: open-loop in position, closed-loop
  // (real robot integration) in torque so the compliant tasks react to contact.
  auto ctrl_mode = datastore().get<std::string>("ControlMode");
  if(ctrl_mode == "Position")
  {
    return mc_control::fsm::Controller::run(mc_solver::FeedbackType::OpenLoop);
  }
  return mc_control::fsm::Controller::run(mc_solver::FeedbackType::ClosedLoopIntegrateReal);
}

void CollabPegInHoleController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);
  requestState("Initial");
}

void CollabPegInHoleController::switchToInitialState()
{
  setEstimatorActive(false);

  if(eeInSolver_)
  {
    solver().removeTask(compEETask);
    eeInSolver_ = false;
  }
  if(!compPostureTask->inSolver())
  {
    solver().addTask(compPostureTask);
  }
  compPostureTask->reset();
  compPostureTask->stiffness(initialPostureStiffness_);
  compPostureTask->target(postureHome);
  compPostureTask->makeCompliant(false);

  datastore().assign<std::string>("ControlMode", "Position");
  requestState("Initial");
  mc_rtc::log::success("[CollabPegInHoleController] Initial state - position controlled");
}

void CollabPegInHoleController::switchToCompliantState()
{
  setEstimatorActive(true);

  // Without the residual estimator the compliant tasks get no external-force
  // feedback, so the arm will not move when pushed even once the EE is "freed".
  // (setEstimatorActive already warns once if the plugin is missing.)
  if(estimatorAvailable() && !datastore().call<bool>("EF_Estimator::isActive"))
  {
    mc_rtc::log::warning("[CollabPegInHoleController] ExternalForcesEstimator is loaded but not active - compliance "
                         "feedback is off.");
  }
  if(!robot().hasForceSensor(ft_sensor))
  {
    mc_rtc::log::warning("[CollabPegInHoleController] F/T sensor '{}' missing - the plate cannot be freed on contact.",
                         ft_sensor);
  }
  else
  {
    // The raw wrench includes the plate's own weight; if that alone already trips
    // the enter threshold the EE will be permanently free (looks like "it doesn't
    // hold"). Flag it so the threshold can be raised / gravity handled.
    const double restNorm = robot().forceSensor(ft_sensor).wrench().vector().norm();
    const double restNormNoGrav = robot().forceSensor(ft_sensor).wrenchWithoutGravity(robot()).vector().norm();
    mc_rtc::log::info("[CollabPegInHoleController] Resting F/T wrench norm: raw={:.2f}, gravity-free={:.2f} "
                      "(enter={:.2f}, exit={:.2f})",
                      restNorm, restNormNoGrav, dualForceEnter_, dualForceExit_);
    if(restNorm >= dualForceEnter_)
    {
      mc_rtc::log::warning("[CollabPegInHoleController] Resting raw F/T wrench norm {:.2f} already exceeds "
                           "dual_force_enter {:.2f}: the plate weight/bias will keep the EE permanently free. Raise "
                           "the threshold or use gravity-free triggering.",
                           restNorm, dualForceEnter_);
    }
  }

  // Compliant posture so the arm can be reconfigured in the null space
  // (person A, observed through the residual).
  compPostureTask->reset();
  compPostureTask->stiffness(0.0);
  compPostureTask->damping(postureCompliantDamping_);
  compPostureTask->weight(1);
  compPostureTask->makeCompliant(postureCompliant_);

  // End-effector starts stiff (holds position). The dual-compliance manager frees
  // it up on contact so person B can move the plate to perform the insertion.
  if(!eeInSolver_)
  {
    solver().addTask(compEETask);
    eeInSolver_ = true;
  }
  // Reset the anti-chatter smoothing state and start from the (stiff) hold gains.
  dualFree_ = false;
  wrenchFilterInit_ = false;
  triggerHistory_.clear();
  triggerSum_ = 0.0;
  currentEEStiffness_ = eeHoldStiffness_;
  currentEEDamping_ = eeHoldDamping_;
  holdEndEffector();

  datastore().assign<std::string>("ControlMode", "Torque");
  requestState("Compliant");
  mc_rtc::log::success("[CollabPegInHoleController] Compliant state - torque controlled, hand-guidable");
}

void CollabPegInHoleController::holdEndEffector()
{
  compEETask->reset();
  compEETask->positionTask->reset();
  compEETask->positionTask->weight(eeTaskWeight_);
  compEETask->positionTask->setGains(currentEEStiffness_, currentEEDamping_);
  compEETask->orientationTask->reset();
  compEETask->orientationTask->weight(eeTaskWeight_);
  compEETask->orientationTask->setGains(currentEEStiffness_, currentEEDamping_);
  compEETask->makeCompliant(false);
}

void CollabPegInHoleController::updateSlidingAverage(std::deque<double> & history,
                                                    double & sum,
                                                    double value,
                                                    std::size_t maxSamples)
{
  history.push_back(value);
  sum += value;
  while(history.size() > maxSamples)
  {
    sum -= history.front();
    history.pop_front();
  }
}

sva::ForceVecd CollabPegInHoleController::computeExternalWrench()
{
  if(!extWrenchJac_)
  {
    return sva::ForceVecd::Zero();
  }
  // Full external joint torques written by the ExternalForcesEstimator: the momentum-observer
  // residual (sees person A pushing the arm) fused with the F/T sensor (person B pushing the
  // plate). Already gravity-free.
  const Eigen::VectorXd & tau = robot().externalTorques();
  const auto nrDof = robot().mb().nrDof();
  if(tau.size() != nrDof)
  {
    if(!warnedExtTorqueSize_)
    {
      mc_rtc::log::warning("[CollabPegInHoleController] externalTorques size {} != nrDof {}; cannot reconstruct the "
                           "external wrench.",
                           tau.size(), nrDof);
      warnedExtTorqueSize_ = true;
    }
    return sva::ForceVecd::Zero();
  }

  // tau = J^T * w  ->  solve for the 6D wrench w = [couple; force] (least squares).
  Eigen::MatrixXd fullJ = Eigen::MatrixXd::Zero(6, nrDof);
  extWrenchJac_->fullJacobian(robot().mb(), extWrenchJac_->jacobian(robot().mb(), robot().mbc()), fullJ);
  const Eigen::Vector6d w = fullJ.transpose().completeOrthogonalDecomposition().solve(tau);
  sva::ForceVecd extW(w);
  // Express in world, as the estimator does for its own fusedWrench.
  const auto R = robot().bodyPosW(tool_frame).rotation();
  extW.force() = R * extW.force();
  extW.couple() = R * extW.couple();
  return extW;
}

void CollabPegInHoleController::updateDualCompliance()
{
  if(!eeInSolver_)
  {
    mc_rtc::log::warning("[CollabPegInHoleController] updateDualCompliance called but the end-effector task is not in "
                         "the solver (not in Compliant state?).");
    return;
  }
  const bool useFtTrigger = (triggerSource_ == "ft");
  const bool hasFt = robot().hasForceSensor(ft_sensor);
  if(useFtTrigger && !hasFt)
  {
    if(!warnedNoFtSensor_)
    {
      mc_rtc::log::warning("[CollabPegInHoleController] Dual compliance inactive: trigger_source='ft' but F/T sensor "
                           "'{}' was not found on robot '{}'.",
                           ft_sensor, robot().name());
      warnedNoFtSensor_ = true;
    }
    return;
  }

  // The full external wrench is the only signal that sees BOTH people: person A guiding the
  // arm (visible only in the residual) and person B moving the plate (F/T, fused in by the
  // estimator). Always reconstruct it - it is logged even when 'ft' drives the trigger.
  externalWrench_ = computeExternalWrench();
  const double ftNorm = hasFt ? robot().forceSensor(ft_sensor).wrenchWithoutGravity(robot()).vector().norm() : 0.0;
  currentWrenchNorm_ = useFtTrigger ? ftNorm : externalWrench_.vector().norm();

  // A sustained exact zero means the trigger source is not being fed, not a real measurement.
  if(currentWrenchNorm_ == 0.0)
  {
    if(++zeroWrenchTicks_ > 500 && !warnedZeroWrench_)
    {
      if(useFtTrigger)
      {
        mc_rtc::log::warning("[CollabPegInHoleController] F/T sensor '{}' has read exactly zero for >500 ticks - is "
                             "the RosForceSensor plugin loaded and the Bota topic publishing? Dual compliance cannot "
                             "trigger without a wrench.",
                             ft_sensor);
      }
      else
      {
        mc_rtc::log::warning("[CollabPegInHoleController] External wrench has been exactly zero for >500 ticks - is "
                             "the ExternalForcesEstimator plugin loaded and active? Dual compliance cannot trigger "
                             "without it.");
      }
      warnedZeroWrench_ = true;
    }
  }
  else
  {
    zeroWrenchTicks_ = 0;
  }

  // --- Anti-chatter smoothing of the trigger signal (cf. box controller) ---
  // Layer 1: first-order low-pass on the wrench norm.
  if(!wrenchFilterInit_)
  {
    filteredWrenchNorm_ = currentWrenchNorm_;
    wrenchFilterInit_ = true;
  }
  else
  {
    const double a = std::exp(-timeStep / std::max(1e-3, wrenchFilterTime_));
    filteredWrenchNorm_ = a * filteredWrenchNorm_ + (1.0 - a) * currentWrenchNorm_;
  }
  // Layer 2: moving-average window (window 0 -> 1 sample -> off).
  const std::size_t windowSamples =
      std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(std::max(0.0, triggerAveragingWindow_) / timeStep)));
  updateSlidingAverage(triggerHistory_, triggerSum_, filteredWrenchNorm_, windowSamples);
  triggerWrenchNorm_ = triggerSum_ / static_cast<double>(triggerHistory_.size());

  // Layer 3: two-threshold hysteresis on the smoothed signal, with edge detection.
  if(!dualFree_ && triggerWrenchNorm_ >= dualForceEnter_)
  {
    // Rising edge (hold -> free): enable compliance so the applied force moves the plate.
    dualFree_ = true;
    compEETask->makeCompliant(true);
    mc_rtc::log::info("[CollabPegInHoleController] EE freeing (wrench {:.1f})", triggerWrenchNorm_);
  }
  else if(dualFree_ && triggerWrenchNorm_ < dualForceExit_)
  {
    // Falling edge (free -> hold): latch the target at the current pose, then stiffen.
    dualFree_ = false;
    compEETask->reset();
    compEETask->makeCompliant(false);
    mc_rtc::log::info("[CollabPegInHoleController] EE holding (wrench {:.1f})", triggerWrenchNorm_);
  }

  // While free, keep the target on the current pose so there is no restoring force
  // pulling the plate back - it floats with the applied force + damping.
  if(dualFree_)
  {
    compEETask->reset();
  }

  // Layer 4: exponentially ramp the gains toward the target selected by dualFree_,
  // so stiffness/damping never jump instantaneously.
  const double targetStiffness = dualFree_ ? 0.0 : eeHoldStiffness_;
  const double targetDamping = dualFree_ ? eeFreeDamping_ : eeHoldDamping_;
  const double ga = std::exp(-timeStep / std::max(1e-3, gainTransitionTime_));
  currentEEStiffness_ = ga * currentEEStiffness_ + (1.0 - ga) * targetStiffness;
  currentEEDamping_ = ga * currentEEDamping_ + (1.0 - ga) * targetDamping;
  compEETask->positionTask->setGains(currentEEStiffness_, currentEEDamping_);
  compEETask->orientationTask->setGains(currentEEStiffness_, currentEEDamping_);
}

void CollabPegInHoleController::requestState(const std::string & state)
{
  requestedState_ = state;
}

const std::string & CollabPegInHoleController::requestedState() const
{
  return requestedState_;
}

void CollabPegInHoleController::postureCompliance(bool compliant)
{
  postureCompliant_ = compliant;
  if(compPostureTask)
  {
    compPostureTask->makeCompliant(compliant);
  }
}

bool CollabPegInHoleController::postureCompliance() const
{
  return postureCompliant_;
}

void CollabPegInHoleController::startEpisode()
{
  ++episodeId_;
  recording_ = true;
  episodeLabel_ = "recording";
  datastore().assign<int>("EpisodeId", episodeId_);
  datastore().assign<bool>("Recording", recording_);
  datastore().assign<std::string>("EpisodeLabel", episodeLabel_);
  mc_rtc::log::info("[CollabPegInHoleController] Episode {} started", episodeId_);
}

void CollabPegInHoleController::markEpisode(const std::string & label)
{
  episodeLabel_ = label;
  datastore().assign<std::string>("EpisodeLabel", episodeLabel_);
  mc_rtc::log::info("[CollabPegInHoleController] Episode {} labelled '{}'", episodeId_, label);
}

void CollabPegInHoleController::stopEpisode()
{
  recording_ = false;
  episodeLabel_ = "idle";
  datastore().assign<bool>("Recording", recording_);
  datastore().assign<std::string>("EpisodeLabel", episodeLabel_);
  mc_rtc::log::info("[CollabPegInHoleController] Episode {} stopped", episodeId_);
}

bool CollabPegInHoleController::estimatorAvailable() const
{
  return datastore().has("EF_Estimator::isActive");
}

void CollabPegInHoleController::setEstimatorActive(bool active)
{
  if(!estimatorAvailable())
  {
    if(!warnedNoEstimator_)
    {
      mc_rtc::log::warning("[CollabPegInHoleController] ExternalForcesEstimator datastore calls unavailable - the "
                           "plugin is not loaded; residual-based compliance is disabled.");
      warnedNoEstimator_ = true;
    }
    return;
  }
  if(datastore().call<bool>("EF_Estimator::isActive") != active)
  {
    datastore().call("EF_Estimator::toggleActive");
  }
  if(active)
  {
    if(datastore().has("EF_Estimator::useForceSensor") && !datastore().call<bool>("EF_Estimator::useForceSensor"))
    {
      datastore().call("EF_Estimator::toggleForceSensor");
    }
    if(datastore().has("EF_Estimator::setGain"))
    {
      datastore().call<void, double>("EF_Estimator::setGain", residualGain_);
    }
  }
}

Eigen::VectorXd CollabPegInHoleController::datastoreVector(const std::string & key, Eigen::Index size) const
{
  if(datastore().has(key))
  {
    return datastore().get<Eigen::VectorXd>(key);
  }
  return Eigen::VectorXd::Zero(size);
}

void CollabPegInHoleController::addGui()
{
  gui()->addElement(
      {"Collab", "Mode"},
      mc_rtc::gui::Label("Current mode", [this]() { return datastore().get<std::string>("ControlMode"); }),
      mc_rtc::gui::Button("Go to Initial", [this]() { requestState("Initial"); }),
      mc_rtc::gui::Button("Go to Compliant", [this]() { requestState("Compliant"); }));

  gui()->addElement(
      {"Collab", "Compliance"},
      mc_rtc::gui::Checkbox(
          "Posture compliant", [this]() { return postureCompliance(); },
          [this]() { postureCompliance(!postureCompliance()); }),
      mc_rtc::gui::Label("EE free (dual compliance)", [this]() { return dualFree_ ? "yes" : "no"; }),
      mc_rtc::gui::Label("Trigger source", [this]() { return triggerSource_; }),
      mc_rtc::gui::Label("Trigger norm (raw)", [this]() { return currentWrenchNorm_; }),
      mc_rtc::gui::Label("Trigger norm (smoothed)", [this]() { return triggerWrenchNorm_; }),
      mc_rtc::gui::Label("External wrench norm (A+B)",
                         [this]() { return externalWrench_.vector().norm(); }),
      mc_rtc::gui::Label("F/T wrench norm (B only)",
                         [this]()
                         {
                           return robot().hasForceSensor(ft_sensor)
                                      ? robot().forceSensor(ft_sensor).wrenchWithoutGravity(robot()).vector().norm()
                                      : 0.0;
                         }),
      mc_rtc::gui::Label("EE stiffness (ramped)", [this]() { return currentEEStiffness_; }),
      mc_rtc::gui::NumberInput(
          "Free above (wrench)", [this]() { return dualForceEnter_; }, [this](double v) { dualForceEnter_ = v; }),
      mc_rtc::gui::NumberInput(
          "Hold below (wrench)", [this]() { return dualForceExit_; }, [this](double v) { dualForceExit_ = v; }),
      mc_rtc::gui::NumberInput(
          "Wrench filter time (s)", [this]() { return wrenchFilterTime_; }, [this](double v) { wrenchFilterTime_ = v; }),
      mc_rtc::gui::NumberInput(
          "Trigger averaging window (s)", [this]() { return triggerAveragingWindow_; },
          [this](double v) { triggerAveragingWindow_ = v; }),
      mc_rtc::gui::NumberInput(
          "Gain transition time (s)", [this]() { return gainTransitionTime_; },
          [this](double v) { gainTransitionTime_ = v; }));

  gui()->addElement({"Collab", "Episode"},
                    mc_rtc::gui::Label("Episode id", [this]() { return std::to_string(episodeId_); }),
                    mc_rtc::gui::Label("Label", [this]() { return episodeLabel_; }),
                    mc_rtc::gui::Button("Start episode", [this]() { startEpisode(); }),
                    mc_rtc::gui::Button("Mark success", [this]() { markEpisode("success"); }),
                    mc_rtc::gui::Button("Mark failure", [this]() { markEpisode("failure"); }),
                    mc_rtc::gui::Button("Stop episode", [this]() { stopEpisode(); }));
}

void CollabPegInHoleController::addLog()
{
  const auto nrDof = robot().mb().nrDof();

  // --- Person A channel: joint-torque residual / external torques ---
  logger().addLogEntry("collab_tau_ext", this, [this]() -> Eigen::VectorXd { return robot().externalTorques(); });
  logger().addLogEntry(
      "collab_tau_comp", this, [this, nrDof]() -> Eigen::VectorXd
      { return robot().compensationTorques() ? robot().compensationTorques().value() : Eigen::VectorXd::Zero(nrDof); });
  logger().addLogEntry("collab_alphaD_ext", this, [this]() -> Eigen::VectorXd { return robot().externalTorquesAcc(); });
  logger().addLogEntry("collab_residual", this, [this, nrDof]() -> Eigen::VectorXd
                       { return datastoreVector("EF_Estimator::getResidualOnly", nrDof); });
  logger().addLogEntry("collab_speed_residual", this,
                       [this, nrDof]() -> Eigen::VectorXd { return datastoreVector("speed_residual", nrDof); });

  // --- Person B channel: F/T sensor wrench on the peg piece ---
  logger().addLogEntry("collab_ft_wrench", this,
                       [this]() -> sva::ForceVecd
                       {
                         if(robot().hasForceSensor(ft_sensor))
                         {
                           return robot().forceSensor(ft_sensor).wrench();
                         }
                         return sva::ForceVecd::Zero();
                       });
  logger().addLogEntry("collab_ft_wrench_world", this,
                       [this]() -> sva::ForceVecd
                       {
                         if(robot().hasForceSensor(ft_sensor))
                         {
                           return robot().forceSensor(ft_sensor).worldWrenchWithoutGravity(robot());
                         }
                         return sva::ForceVecd::Zero();
                       });

  // --- Robot / task state ---
  logger().addLogEntry("collab_tool_pose", this,
                       [this]() -> sva::PTransformd
                       {
                         if(robot().hasFrame(tool_frame))
                         {
                           return robot().frame(tool_frame).position();
                         }
                         return sva::PTransformd::Identity();
                       });
  logger().addLogEntry("collab_insertion_pose", this,
                       [this]() -> sva::PTransformd
                       {
                         if(robot().hasFrame(insertion_frame))
                         {
                           return robot().frame(insertion_frame).position();
                         }
                         return sva::PTransformd::Identity();
                       });

  // --- Episode markers ---
  logger().addLogEntry("collab_episode_id", this, [this]() -> double { return static_cast<double>(episodeId_); });
  logger().addLogEntry("collab_recording", this, [this]() -> bool { return recording_; });
  logger().addLogEntry("collab_episode_label", this, [this]() -> std::string { return episodeLabel_; });
  logger().addLogEntry("collab_posture_compliant", this, [this]() -> bool { return postureCompliant_; });

  // --- Dual-compliance state ---
  // --- Dual-compliance trigger (full external wrench: person A + person B) ---
  logger().addLogEntry("collab_ext_wrench", this, [this]() -> sva::ForceVecd { return externalWrench_; });
  logger().addLogEntry("collab_ext_wrench_norm", this, [this]() -> double { return externalWrench_.vector().norm(); });

  logger().addLogEntry("collab_ee_free", this, [this]() -> bool { return dualFree_; });
  logger().addLogEntry("collab_trigger_norm", this, [this]() -> double { return currentWrenchNorm_; });
  logger().addLogEntry("collab_trigger_norm_filtered", this, [this]() -> double { return filteredWrenchNorm_; });
  logger().addLogEntry("collab_trigger_norm_smoothed", this, [this]() -> double { return triggerWrenchNorm_; });
  logger().addLogEntry("collab_ee_stiffness", this, [this]() -> double { return currentEEStiffness_; });
  logger().addLogEntry("collab_ft_wrench_norm_nograv", this,
                       [this]() -> double
                       {
                         if(robot().hasForceSensor(ft_sensor))
                         {
                           return robot().forceSensor(ft_sensor).wrenchWithoutGravity(robot()).vector().norm();
                         }
                         return 0.0;
                       });
}
