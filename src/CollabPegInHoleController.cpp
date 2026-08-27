#include "CollabPegInHoleController.h"

// Only here, not in the header: onnxruntime's C++ API is a large include, and nothing in the
// controller's interface mentions it.
#include "OnnxBackend.h"

#include <mc_rbdyn/ForceSensor.h>
#include <mc_rtc/gui/Button.h>
#include <mc_rtc/gui/Checkbox.h>
#include <mc_rtc/gui/Label.h>
#include <mc_solver/DynamicsConstraint.h>

#include <RBDyn/FD.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

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
    rconfig("ee_orientation_compliant", eeOrientationCompliant_);
    rconfig("ee_up_vector_enabled", eeUpVectorEnabled_);
    rconfig("ee_orientation_weight", eeOrientationWeight_);
    rconfig("ee_up_vector_stiffness", eeUpVectorStiffness_);
    rconfig("ee_up_vector_weight", eeUpVectorWeight_);
    rconfig("ee_task_weight", eeTaskWeight_);
    rconfig("wrench_filter_time", wrenchFilterTime_);
    rconfig("trigger_averaging_window", triggerAveragingWindow_);
    rconfig("gain_transition_time", gainTransitionTime_);
    rconfig("trigger_source", triggerSource_);
    rconfig("residual_gain", residualGain_);
    rconfig("posture_compliant_damping", postureCompliantDamping_);
    rconfig("posture_compliant_stiffness", postureCompliantStiffness_);
    rconfig("initial_posture_stiffness", initialPostureStiffness_);
    rconfig("homing_duration", homingDuration_);
    rconfig("homing_posture_stiffness", homingPostureStiffness_);
    rconfig("vel_damper_m", m_);
    rconfig("vel_damper_lambda", lambda_);
    rconfig("max_external_torque_ratio", extTorqueRatio_);
    rconfig("external_torque_filter_hz", extTorqueFilterHz_);
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

  // Per-dof bound used to saturate the estimator's external torque before it reaches the QP
  // (see conditionExternalTorques()). tl()/tu() are indexed by joint, each entry sized by that
  // joint's dof, so walk them in dof order to line the vector up with externalTorques().
  {
    const auto & mb = robot().mb();
    const auto & tl = robot().tl();
    const auto & tu = robot().tu();
    extTorqueLimit_ = Eigen::VectorXd::Constant(mb.nrDof(), std::numeric_limits<double>::infinity());
    int dof = 0;
    for(std::size_t j = 0; j < mb.joints().size(); ++j)
    {
      const int jdof = mb.joint(static_cast<int>(j)).dof();
      for(int k = 0; k < jdof; ++k, ++dof)
      {
        const double lo = (j < tl.size() && static_cast<std::size_t>(k) < tl[j].size())
                              ? std::abs(tl[j][static_cast<std::size_t>(k)])
                              : std::numeric_limits<double>::infinity();
        const double hi = (j < tu.size() && static_cast<std::size_t>(k) < tu[j].size())
                              ? std::abs(tu[j][static_cast<std::size_t>(k)])
                              : std::numeric_limits<double>::infinity();
        extTorqueLimit_(dof) = std::min(lo, hi);
      }
    }
    std::string limits;
    for(int i = 0; i < dof; ++i)
    {
      limits += (i ? ", " : "") + fmt::format("{:.1f}", extTorqueLimit_(i));
    }
    mc_rtc::log::info("[CollabPegInHoleController] External-torque saturation at {:.0f}% of the joint torque limits "
                      "[{}] N.m. Set max_external_torque_ratio <= 0 to disable.",
                      100.0 * extTorqueRatio_, limits);
  }

  // --- Behaviour-cloned partner, off unless configured ---
  if(config.has("policy"))
  {
    auto pconfig = config("policy");
    policyEnabled_ = pconfig("enabled", false);
    pconfig("path", policyPath_);
    pconfig("body", policyBody_);
    pconfig("point", policyPoint_);
    pconfig("decimation", policyDecimation_);
    if(pconfig.has("peg_target"))
    {
      pegTarget_ = pconfig("peg_target");
      havePegTarget_ = true;
    }
    // The intent path's impedance: the gains of the partner being replaced. Read whether or
    // not the policy turns out to need them, so a mismatch is reported below rather than
    // silently producing a zero-stiffness partner, which is not a soft partner but an absent
    // one.
    pconfig("stiffness", policyStiffness_);
    pconfig("damping", policyDamping_);
    pconfig("max_force", policyMaxForce_);
    // Default to the contact itself: a partner aims through the point it pushes, and only a
    // configuration that deliberately separates them says otherwise.
    policyRefBody_ = policyBody_;
    policyRefPoint_ = policyPoint_;
    pconfig("reference_body", policyRefBody_);
    pconfig("reference_point", policyRefPoint_);
    pconfig("velocity_filter", policyVelocityFilter_);
    pconfig("log_observation", logPolicyObservation_);
    pconfig("replay", policyReplayPath_);
    if(policyEnabled_)
    {
      if(policyDecimation_ < 1)
      {
        mc_rtc::log::error_and_throw("[CollabPegInHoleController] policy.decimation must be >= 1, got {}",
                                     policyDecimation_);
      }
      // A replay stands in for the prediction, not for the controller: everything below the
      // intent -- the impedance, the saturation, the Jacobian, the refAccel_ injection -- is
      // the code a policy runs through, unchanged. That is the entire point of it, so it is
      // wired in here rather than given a path of its own.
      if(!policyReplayPath_.empty())
      {
        if(!policyPath_.empty())
        {
          mc_rtc::log::error_and_throw("[CollabPegInHoleController] policy.replay and policy.path are both set. A "
                                       "replay substitutes for the prediction, so exactly one of the two produces "
                                       "the intent; running both would silently use one and look like the other.");
        }
        replay_ = std::make_unique<collab::IntentReplay>(policyReplayPath_, timeStep);
        // A replay is a recorded intent by construction. There is no force variant: the
        // recorded force belongs to the episode's own arm states and re-applying it open-loop
        // would not be a partner at all.
        policyOutputKind_ = "intent";
      }
      // The extension decides the backend, and the backend decides what its output means.
      // Neither is configurable: a `policy.output` key that could disagree with the file is a
      // way to run an intent through the force path, which produces motion and no error.
      else if(policyPath_.size() >= 5 && policyPath_.compare(policyPath_.size() - 5, 5, ".onnx") == 0)
      {
        auto backend = std::make_unique<collab::OnnxBackend>(policyPath_);
        policyOutputKind_ = backend->outputKind();
        policy_ = std::move(backend);
      }
      else
      {
        auto backend = std::make_unique<collab::PolicyPartner>();
        backend->load(policyPath_);
        backend->checkFeatureOrder(policyFeatureNames(backend->withPhase()));
        if(backend->withPhase())
        {
          mc_rtc::log::error_and_throw("[CollabPegInHoleController] this policy was trained with the episode clock "
                                       "in its observation, which this controller does not provide: it has no "
                                       "episode phase outside a plugin-driven batch. Retrain without --with-phase.");
        }
        policyOutputKind_ = "force";
        policy_ = std::move(backend);
      }

      if(policyOutputKind_ == "force")
      {
        // Only the force policy observes (peg - hole). An intent policy's observation is the
        // robot's own state, so requiring the peg there would be asking for a number that
        // changes nothing.
        if(!havePegTarget_)
        {
          mc_rtc::log::error_and_throw("[CollabPegInHoleController] policy.enabled is set with a force-predicting "
                                       "policy but policy.peg_target is not. The error that policy closes is "
                                       "(peg - hole), and the controller has no other source for where the peg is.");
        }
        if(policy_->nOut() != 3)
        {
          mc_rtc::log::error_and_throw("[CollabPegInHoleController] a force policy must emit 3 values, this one "
                                       "emits {}",
                                       policy_->nOut());
        }
      }
      else
      {
        // A replay has no forward pass, so nothing to size-check: its file is validated by
        // IntentReplay's own constructor.
        if(policy_ && policy_->nOut() < 3)
        {
          mc_rtc::log::error_and_throw("[CollabPegInHoleController] an intent policy must emit at least 3 values "
                                       "(the translation of a pose), this one emits {}",
                                       policy_->nOut());
        }
        if((policyStiffness_.array() <= 0).any() || (policyDamping_.array() < 0).any() || policyMaxForce_ <= 0)
        {
          mc_rtc::log::error_and_throw(
              "[CollabPegInHoleController] this policy predicts an intent, which only becomes a force through the "
              "impedance of the partner it replaces -- so policy.stiffness, policy.damping and policy.max_force are "
              "required and must be positive. Got kp [{}], kd [{}], max {}.",
              policyStiffness_.transpose(), policyDamping_.transpose(), policyMaxForce_);
        }
        if(!(policyVelocityFilter_ > 0.0 && policyVelocityFilter_ <= 1.0))
        {
          mc_rtc::log::error_and_throw("[CollabPegInHoleController] policy.velocity_filter must be in (0, 1], got {}",
                                       policyVelocityFilter_);
        }
      }
      mc_rtc::log::info("[CollabPegInHoleController] policy partner '{}' active ({} output), evaluated every {} "
                        "ticks ({:.0f} Hz)",
                        policy_ ? policy_->partner() : replay_->partner(), policyOutputKind_, policyDecimation_,
                        1.0 / (timeStep * policyDecimation_));
      if(replay_)
      {
        mc_rtc::log::warning("[CollabPegInHoleController]   REPLAY, not a policy: {} recorded samples are being "
                             "played back by episode tick. This measures the impedance and the refAccel injection, "
                             "NOT a learned partner -- its success rate is not a policy result.",
                             replay_->size());
      }
      if(policyOutputKind_ == "intent")
      {
        mc_rtc::log::info("[CollabPegInHoleController]   impedance kp [{:.1f}] kd [{:.1f}] max {:.1f} N about "
                          "'{}', reference velocity filtered at {:.2f}",
                          policyStiffness_.x(), policyDamping_.x(), policyMaxForce_, policyRefBody_,
                          policyVelocityFilter_);
      }
    }
  }

  // Safety framework: closed-loop velocity damper on the dynamics + collisions.
  // It's not possible to use the same values as m_ and lambda_ for self-collisions this was taken from the monodzukuri
  // controller on rss2025-noVirtualTorqueSensor
  selfCollisionConstraint->setCollisionsDampers(solver(), {1.1, 9.0});
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(new mc_solver::DynamicsConstraint(
      robots(), robot().robotIndex(), solver().dt(), std::array<double, 3>{0.1, 0.01, xsiOff_},
      std::array<double, 2>{m_, lambda_}, 0.9, false, true));
  solver().addConstraintSet(dynamicsConstraint);

  // Replace the default posture task with the compliant one, and expose it through the
  // getPostureTask datastore call so anything looking for the posture task finds this one.
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
  datastore().make<std::string>("EpisodeOutcome", episodeOutcome_);
  datastore().make<bool>("Recording", recording_);

  addGui();
  addLog();

  mc_rtc::log::success("[CollabPegInHoleController] init done");
}

void CollabPegInHoleController::startHoming()
{
  if(homingDuration_ <= 0.0)
  {
    homing_ = false;
    compPostureTask->stiffness(initialPostureStiffness_);
    compPostureTask->target(postureHome);
    return;
  }
  homingStart_.clear();
  const auto & q = robot().mbc().q;
  for(const auto & [joint, home] : postureHome)
  {
    if(!robot().hasJoint(joint))
    {
      continue;
    }
    const auto & current = q[robot().jointIndexByName(joint)];
    if(current.size() == home.size())
    {
      homingStart_[joint] = current;
    }
  }
  homingTime_ = 0.0;
  homing_ = !homingStart_.empty();
  compPostureTask->stiffness(homingPostureStiffness_);
  // Start from where the arm is: the whole point is that the reference never steps.
  compPostureTask->target(homingStart_);
  clearHomingRefs();
}

void CollabPegInHoleController::clearHomingRefs()
{
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(robot().mb().nrDof());
  compPostureTask->refVel(zero);
  compPostureTask->refAccel(zero);
}

void CollabPegInHoleController::updateHoming()
{
  if(!homing_)
  {
    return;
  }
  homingTime_ += solver().dt();
  const double u = std::min(homingTime_ / homingDuration_, 1.0);
  // Quintic minimum-jerk: zero velocity and acceleration at both ends, so the posture task
  // never has to invent a velocity the gains would have to bound.
  const double T = homingDuration_;
  const double s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
  const double sd = u * u * (30.0 + u * (-60.0 + 30.0 * u)) / T;
  const double sdd = u * (60.0 + u * (-180.0 + 120.0 * u)) / (T * T);
  std::map<std::string, std::vector<double>> target;
  // The same profile handed to the task as a feed-forward: position alone leaves the tracking
  // error to be paid for by the gain, which is the very trade this homing exists to avoid. With
  // the velocity and acceleration of the reference supplied, the stiffness only has to correct
  // what is left over.
  Eigen::VectorXd refVel = Eigen::VectorXd::Zero(robot().mb().nrDof());
  Eigen::VectorXd refAccel = Eigen::VectorXd::Zero(robot().mb().nrDof());
  for(const auto & [joint, home] : postureHome)
  {
    const auto it = homingStart_.find(joint);
    if(it == homingStart_.end())
    {
      target[joint] = home;
      continue;
    }
    std::vector<double> q(home.size());
    for(std::size_t i = 0; i < home.size(); ++i)
    {
      q[i] = it->second[i] + s * (home[i] - it->second[i]);
    }
    target[joint] = q;
    // refVel/refAccel are indexed by *degree of freedom*, not by joint, and only line up with
    // the q entries one-to-one for the 1-dof joints this arm is made of -- hence the size check
    // rather than an assumption.
    const auto jIndex = robot().jointIndexByName(joint);
    const auto & j = robot().mb().joint(jIndex);
    if(j.dof() != static_cast<int>(home.size()))
    {
      continue;
    }
    const int dofStart = robot().mb().jointPosInDof(static_cast<int>(jIndex));
    for(std::size_t i = 0; i < home.size(); ++i)
    {
      const double delta = home[i] - it->second[i];
      refVel(dofStart + static_cast<int>(i)) = sd * delta;
      refAccel(dofStart + static_cast<int>(i)) = sdd * delta;
    }
  }
  compPostureTask->target(target);
  compPostureTask->refVel(refVel);
  compPostureTask->refAccel(refAccel);
  if(u >= 1.0)
  {
    homing_ = false;
    // Both are zero at u = 1 by construction, but a stale feed-forward left on the task is the
    // kind of thing that only shows up as a drift in some later state.
    clearHomingRefs();
    mc_rtc::log::info("[CollabPegInHoleController] Homed in {:.2f} s", homingTime_);
  }
}

bool CollabPegInHoleController::run()
{
  updateHoming();

  // Refresh the reconstructed external wrench every tick, not only from updateDualCompliance()
  // (which runs in the Compliant state only): it is a logged observation of both people and must
  // not carry a stale value into the samples recorded in any other state.
  //
  // Deliberately BEFORE conditionExternalTorques(): this is an observation of what the two people
  // did and belongs in the demonstrations unaltered. The saturation below exists to keep the QP
  // solvable, and must not rewrite the dataset.
  externalWrench_ = computeExternalWrench();

  // Now bound what the solver is told. The dynamics constraint takes tau_ext as a KNOWN term of
  // H qdd + C = tau + tau_ext with |tau| bounded, so an estimate larger than what the motors can
  // oppose makes the constraint set EMPTY -- QLD returns "constraints inconsistent", the QP
  // produces no command at all, and the simulator integrates garbage. The wrist makes this cheap
  // to hit: joints 5-7 are limited to 9 N.m against 39 N.m for joints 1-4, so ~30 N on the plate
  // at ~0.3 m already outruns joint_5.
  conditionExternalTorques();

  // Ahead of the FSM, so the trigger term this writes is from this tick and not the last
  // one: updateDualCompliance() runs inside the Compliant state, below.
  updatePolicyPartner();

  if(recording_)
  {
    ++episodeTick_;
  }

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
  applySimForceSensorCalibration();
  requestState("Initial");
}

void CollabPegInHoleController::applySimForceSensorCalibration()
{
  namespace fs = std::filesystem;

  // The F/T calibration is not interchangeable between the robot and the simulator, and the
  // robot module cannot tell which one it is in: it is built before any interface exists, and
  // mc_mujoco loads the same non-MuJoCo variant we run on hardware. So decide it here.
  //
  // mc_mujoco registers "<robot>::SetPDGains" from MjSimImpl::makeDatastoreCalls(), which runs
  // before controller->init() and therefore before this reset() -- mc_kortex registers nothing
  // of the sort. That makes it a reliable probe, and an available one by now.
  if(!datastore().has(robot().name() + "::SetPDGains"))
  {
    return; // on hardware: the shipped calibration is the right one, leave it alone
  }
  if(!robot().hasForceSensor(ft_sensor))
  {
    return;
  }

  // mc_kinova lays these out as <...>/calib/<variant> for hardware and <...>/calib/sim/<variant>
  // for simulation, so derive one from the other. If MC_KINOVA_CALIB=sim already selected the
  // simulation one, the parent directory is "sim" and there is nothing to do.
  const fs::path calibDir = robot().module().calib_dir;
  if(calibDir.parent_path().filename() == "sim")
  {
    mc_rtc::log::info("[CollabPegInHoleController] Running under mc_mujoco with the simulation F/T calibration "
                      "already selected ({}).",
                      calibDir.string());
    return;
  }
  const fs::path simFile = calibDir.parent_path() / "sim" / calibDir.filename() / ("calib_data." + ft_sensor);
  if(!fs::exists(simFile))
  {
    mc_rtc::log::warning(
        "[CollabPegInHoleController] Running under mc_mujoco, but no simulation F/T calibration exists at '{}'. The "
        "robot's calibration carries the sensor's force gain and its home-pose tare, neither of which applies in "
        "simulation: the plate's own weight will stay in the wrench, the observer will project it in as a phantom "
        "external force, and entering Compliant will kick the arm with it. Check collab_ft_wrench_norm_nograv.",
        simFile.string());
    return;
  }

  // Both robots: the controller reads the wrench off robot(), while ExternalForcesObserver
  // projects the sensors of realRobot(). Calibrating only one leaves the two disagreeing.
  for(auto * r : {&robot(), &realRobot()})
  {
    r->device<mc_rbdyn::ForceSensor>(ft_sensor).loadCalibrator(simFile.string(), r->mbc().gravity);
  }
  mc_rtc::log::success("[CollabPegInHoleController] mc_mujoco detected: loaded the simulation F/T calibration for '{}' "
                       "from '{}', overriding the robot calibration the module had loaded.",
                       ft_sensor, simFile.string());
}

void CollabPegInHoleController::conditionExternalTorques()
{
  // --- Low-pass, first. This is the one that keeps the freed end-effector stable. ---
  //
  // tau_ext and the robot's own motion are coupled through the payload: the F/T sensor reads
  // the reaction of everything distal to it, and wrenchWithoutGravity() removes gravity, not
  // inertia. The plate's CoM sits 139 mm off the sensor, so an angular acceleration alpha of
  // the tool shows up as a force m*|alpha x r| ~ 0.095*alpha at the sensor -- 100 rad/s^2 is
  // already ~10 N of pure artefact. The compliant task then turns tau_ext straight back into
  // commanded acceleration (refAccel = J H^-1 tau_ext), so sensor -> estimate -> command ->
  // motion -> sensor closes ALGEBRAICALLY, within one tick. Once the end-effector stiffness
  // ramps down there is nothing left to dissipate it and it rings up.
  //
  // Measured on the 19:30 run: the tool travelled 1.3 mm in the 26 ms it took to die, while the
  // F/T force swung +45 to -77 N. None of that was the operator, who was holding a steady 12 N.
  //
  // A human guiding an arm has a bandwidth of a few Hz; the ring was at ~90 Hz. So filter the
  // estimate the SOLVER sees. This is deliberately not a fix to the estimate itself -- the
  // principled complement is to subtract the payload's dynamic reaction using the commanded
  // acceleration, which needs the distal spatial inertia and is worth doing -- but a model-based
  // compensation is only ever as good as the model, and the loop needs no gain margin to ring.
  //
  // Nothing recorded is filtered: externalWrench_ and the datastore channels are read before
  // this runs. This shapes the control path only.
  if(extTorqueFilterHz_ > 0.0)
  {
    const Eigen::VectorXd & tau = realRobot().externalTorques();
    if(tau.size() != extTorqueFiltered_.size())
    {
      extTorqueFiltered_ = tau; // first tick, or a size change: seed rather than step
    }
    else
    {
      const double a = std::exp(-2.0 * M_PI * extTorqueFilterHz_ * timeStep);
      extTorqueFiltered_ = a * extTorqueFiltered_ + (1.0 - a) * tau;
    }
    robot().setExternalTorques(extTorqueFiltered_);
    realRobot().setExternalTorques(extTorqueFiltered_);
  }

  // --- Then saturate. ---
  extTorqueClampedDofs_ = 0;
  if(extTorqueRatio_ <= 0.0 || extTorqueLimit_.size() == 0)
  {
    return; // saturation disabled
  }
  auto clamp = [this](mc_rbdyn::Robot & r)
  {
    const Eigen::VectorXd & tau = r.externalTorques();
    if(tau.size() != extTorqueLimit_.size())
    {
      return;
    }
    Eigen::VectorXd bounded = tau;
    for(Eigen::Index i = 0; i < bounded.size(); ++i)
    {
      const double bound = extTorqueRatio_ * extTorqueLimit_(i);
      if(!std::isfinite(bound))
      {
        continue;
      }
      if(std::abs(bounded(i)) > bound)
      {
        bounded(i) = std::copysign(bound, bounded(i));
        ++extTorqueClampedDofs_;
      }
    }
    r.setExternalTorques(bounded);
  };
  // Both, and in this order: the solver copies realRobot()'s value onto robot() every tick, but
  // CompliantEndEffectorTask reads robot()'s, and anything running before the next solver pass
  // would otherwise see the unbounded one.
  clamp(realRobot());
  clamp(robot());
  // Halved: each clamped dof is counted once per robot.
  extTorqueClampedDofs_ /= 2;
  if(extTorqueClampedDofs_ > 0 && !warnedExtTorqueClamp_)
  {
    mc_rtc::log::warning("[CollabPegInHoleController] External torque estimate saturated on {} joint(s) at {:.0f}% of "
                         "their torque limit. The push exceeds what the arm can oppose - the QP would otherwise have "
                         "no feasible solution. Logged as collab_tau_ext_clamped_dofs (warned once).",
                         extTorqueClampedDofs_, 100.0 * extTorqueRatio_);
    warnedExtTorqueClamp_ = true;
  }
}

void CollabPegInHoleController::switchToInitialState()
{
  setEstimatorActive(false);

  if(eeInSolver_)
  {
    solver().removeTask(compEETask);
    eeInSolver_ = false;
  }
  if(eeUpVectorTask)
  {
    solver().removeTask(eeUpVectorTask);
    eeUpVectorTask.reset();
  }
  if(!compPostureTask->inSolver())
  {
    solver().addTask(compPostureTask);
  }
  compPostureTask->reset();
  compPostureTask->makeCompliant(false);
  startHoming();

  datastore().assign<std::string>("ControlMode", "Position");
  requestState("Initial");
  mc_rtc::log::success("[CollabPegInHoleController] Initial state - position controlled");
}

void CollabPegInHoleController::switchToCompliantState()
{
  if(homing_)
  {
    homing_ = false;
    clearHomingRefs();
  }
  setEstimatorActive(true);

  // Without the residual estimator the compliant tasks get no external-force
  // feedback, so the arm will not move when pushed even once the EE is "freed".
  // (setEstimatorActive already warns once if the observer is missing.)
  if(estimatorAvailable() && !datastore().call<bool>("EF_Estimator::isActive"))
  {
    mc_rtc::log::warning("[CollabPegInHoleController] ExternalForcesObserver is loaded but not active - compliance "
                         "feedback is off.");
  }
  if(!robot().hasForceSensor(ft_sensor))
  {
    mc_rtc::log::warning("[CollabPegInHoleController] F/T sensor '{}' missing - the plate cannot be freed on contact.",
                         ft_sensor);
  }
  else
  {
    // If the resting signal alone already trips the enter threshold the EE will be permanently
    // free (looks like "it doesn't hold"). Flag it -- but against the signal the trigger
    // actually uses, which is never the raw wrench: 'ft' uses the gravity-free one and
    // 'external' the reconstructed external wrench. Comparing the raw norm here used to warn on
    // the plate's own weight every single run, on a threshold that signal never reaches.
    const double restNorm = robot().forceSensor(ft_sensor).wrench().vector().norm();
    const double restNormNoGrav = robot().forceSensor(ft_sensor).wrenchWithoutGravity(robot()).vector().norm();
    const bool useFtTrigger = (triggerSource_ == "ft");
    const double restTrigger = useFtTrigger ? restNormNoGrav : externalWrench_.vector().norm();
    mc_rtc::log::info("[CollabPegInHoleController] Resting F/T wrench norm: raw={:.2f}, gravity-free={:.2f}; trigger "
                      "source '{}' rests at {:.2f} (enter={:.2f}, exit={:.2f})",
                      restNorm, restNormNoGrav, triggerSource_, restTrigger, dualForceEnter_, dualForceExit_);
    if(restNorm >= dualForceEnter_ && restNormNoGrav < dualForceEnter_)
    {
      mc_rtc::log::info("[CollabPegInHoleController] (The raw norm {:.2f} is above dual_force_enter {:.2f}, but that "
                        "is the uncompensated plate weight and no trigger uses it.)",
                        restNorm, dualForceEnter_);
    }
    if(restTrigger >= dualForceEnter_)
    {
      mc_rtc::log::warning("[CollabPegInHoleController] Resting trigger norm {:.2f} (source '{}') already exceeds "
                           "dual_force_enter {:.2f}: the EE will be permanently free. Raise the threshold, or check "
                           "the F/T gravity compensation.",
                           restTrigger, triggerSource_, dualForceEnter_);
    }
  }

  // Compliant posture so the arm can be reconfigured in the null space
  // (person A, observed through the residual).
  compPostureTask->reset();
  compPostureTask->stiffness(postureCompliantStiffness_);
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

  // Keep the plate level, and only that. The vector held is whichever tool-frame direction
  // points up right now, latched here rather than hard-coded, so the task never asks the arm
  // to rotate to some nominal pose it does not happen to be in -- it only asks it to stay
  // where it already is about two of the three angular axes.
  if(eeUpVectorEnabled_)
  {
    // frame.position().rotation() is E_0_f, world to frame, so it maps world vertical into
    // frame coordinates.
    eeUpVectorInTool_ = robot().frame(tool_frame).position().rotation() * Eigen::Vector3d::UnitZ();
    eeUpVectorTask = std::make_shared<mc_tasks::VectorOrientationTask>(robot().frame(tool_frame), eeUpVectorInTool_,
                                                                       eeUpVectorStiffness_, eeUpVectorWeight_);
    eeUpVectorTask->targetVector(Eigen::Vector3d::UnitZ());
    solver().addTask(eeUpVectorTask);
    mc_rtc::log::info("[CollabPegInHoleController] Tool up-vector held ({:.0f} N.m/rad, weight {:.0f}); "
                      "rotation about the vertical stays free",
                      eeUpVectorStiffness_, eeUpVectorWeight_);
  }

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
  compEETask->orientationTask->weight(eeUpVectorEnabled_ && eeOrientationWeight_ >= 0.0 ? eeOrientationWeight_
                                                                                        : eeTaskWeight_);
  compEETask->orientationTask->setGains(currentEEStiffness_, currentEEDamping_);
  applyEECompliance(false);
}

double CollabPegInHoleController::toolTilt() const
{
  // The angle the held axis has actually drifted from vertical: the one number that says
  // whether the up-vector task is doing its job.
  const Eigen::Vector3d up = robot().frame(tool_frame).position().rotation().transpose() * eeUpVectorInTool_;
  return std::acos(std::clamp(up.z() / std::max(1e-9, up.norm()), -1.0, 1.0));
}

void CollabPegInHoleController::applyEECompliance(bool free)
{
  // Compliance is a 6-vector ordered angular-then-linear, like every other sva motion vector,
  // so the translation can float while the rotation does not.
  //
  // The term it scales is a feed-forward of the disturbance the external-wrench estimator
  // reports: at 1 the task goes along with that disturbance, at 0 it does not. Withholding it
  // from the angular axes is what "the orientation is not compliant" means here -- an external
  // wrench moves the plate without turning it.
  //
  // Nothing else about the orientation task changes: its gains still ramp down with the
  // translation's, and its target is still re-latched every tick. That matters. Pairing this
  // with gains pinned at 400 on a target left where it was latched diverges inside 10 ms -- a
  // stiff task with no feed-forward chases the estimator's noise, which at Compliant entry
  // swings +/-100 rad/s^2. Damping rotation is achievable; holding an exact orientation
  // against that signal is not.
  Eigen::Vector6d gamma = Eigen::Vector6d::Zero();
  if(free)
  {
    gamma.tail<3>().setOnes();
    if(eeOrientationCompliant_)
    {
      gamma.head<3>().setOnes();
    }
  }
  compEETask->setComplianceVector(gamma);
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

sva::ForceVecd CollabPegInHoleController::wrenchFromTorques(const Eigen::VectorXd & tau) const
{
  if(!extWrenchJac_)
  {
    return sva::ForceVecd::Zero();
  }
  const auto nrDof = robot().mb().nrDof();
  if(tau.size() != nrDof)
  {
    if(!warnedExtTorqueSize_)
    {
      mc_rtc::log::warning("[CollabPegInHoleController] external torque vector size {} != nrDof {}; cannot reconstruct "
                           "the wrench.",
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
  // Express in world, as the estimator does for its own fused wrench.
  const auto R = robot().bodyPosW(tool_frame).rotation();
  extW.force() = R * extW.force();
  extW.couple() = R * extW.couple();
  return extW;
}

sva::ForceVecd CollabPegInHoleController::computeExternalWrench()
{
  // Full external joint torques written by the ExternalForcesObserver: the momentum-observer
  // residual (sees person A pushing the arm) plus the F/T sensor projection (person B pushing
  // the plate). Already gravity-free. This is the A+B signal, and it is what drives both the
  // free/hold trigger and the compliant task's feed-forward.
  return wrenchFromTorques(robot().externalTorques());
}

sva::ForceVecd CollabPegInHoleController::computeResidualWrench() const
{
  // Person A alone. The observer feeds the F/T torques into its momentum integral as a known
  // input, so what comes back from getResidualOnly is the unexplained remainder - person B is
  // already subtracted. Logged, not acted on: it is the signal that says which of the two
  // people an angular term belongs to.
  return wrenchFromTorques(datastoreVector("EF_Estimator::getResidualOnly", robot().mb().nrDof()));
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
  // estimator). It is refreshed once per tick in run() - use that value here so the logged and
  // the acted-upon wrench are always the same sample.
  const double ftNorm = hasFt ? robot().forceSensor(ft_sensor).wrenchWithoutGravity(robot()).vector().norm() : 0.0;
  // The policy's own contribution belongs in the trigger even though it is deliberately kept
  // out of externalTorques(). In the demonstrations person A was a real force and fed this
  // same signal; with the policy at the task level the trigger would otherwise see person B
  // alone, re-latch the end-effector earlier, and cost the accuracy that
  // dual_force_exit / combined stiffness buys. Added to the norm, not to the estimate: the
  // observer's channels stay a record of what the world did.
  currentWrenchNorm_ = useFtTrigger ? ftNorm : (externalWrench_.vector().norm() + policyWrenchNorm_);

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
                             "the ExternalForcesObserver in the observer pipeline and active? Dual compliance cannot "
                             "trigger without it.");
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
    applyEECompliance(true);
    mc_rtc::log::info("[CollabPegInHoleController] EE freeing (wrench {:.1f})", triggerWrenchNorm_);
  }
  else if(dualFree_ && triggerWrenchNorm_ < dualForceExit_)
  {
    // Falling edge (free -> hold): latch the target at the current pose, then stiffen.
    dualFree_ = false;
    compEETask->reset();
    applyEECompliance(false);
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

// --- Behaviour-cloned partner ------------------------------------------------------------

std::vector<std::string> CollabPegInHoleController::policyFeatureNames(bool withPhase)
{
  // Mirrors cpih.policy.features.FEATURES. Order is the contract, and PolicyPartner checks
  // this list against the one in the policy file: a policy fed a correctly sized vector in
  // the wrong order emits a plausible wrong force and nothing reports it.
  const std::vector<std::pair<std::string, int>> spec = {
      {"peg_error", 3}, {"hole_velocity", 3}, {"tool_orientation", 4}, {"partner_wrench", 6}, {"q", 7}, {"dq", 7}};
  std::vector<std::string> names;
  for(const auto & [name, width] : spec)
  {
    for(int i = 0; i < width; ++i) { names.push_back(name + "_" + std::to_string(i)); }
  }
  if(withPhase) { names.push_back("phase"); }
  return names;
}

Eigen::VectorXd CollabPegInHoleController::policyFeatures() const
{
  const auto & r = robot();
  const Eigen::Vector3d hole = r.frame(insertion_frame).position().translation();
  const Eigen::Matrix3d toolRotation = r.frame(tool_frame).position().rotation();
  const Eigen::Quaterniond quaternion(toolRotation);

  // Gravity-free and in the world, which is collab_ft_wrench_world -- NOT collab_ft_wrench.
  // The raw sensor reading is dominated by the tool's own 6.5 N, near-constant across every
  // configuration, and a policy trained on it would be reading mostly weight.
  sva::ForceVecd partnerWrench = sva::ForceVecd::Zero();
  if(r.hasForceSensor(ft_sensor)) { partnerWrench = r.forceSensor(ft_sensor).worldWrenchWithoutGravity(r); }

  // The encoder channels, because the dataset's obs/q and obs/dq are qIn and alphaIn.
  const auto & q = r.encoderValues();
  const auto & dq = r.encoderVelocities();

  // 3 + 3 + 4 + 6 + 7 + 7 = 30. Must equal cpih.policy.features.feature_width().
  Eigen::VectorXd x = Eigen::VectorXd::Zero(30);
  x.segment<3>(0) = pegTarget_ - hole;
  x.segment<3>(3) = holeVelocity_;
  x(6) = quaternion.w();
  x(7) = quaternion.x();
  x(8) = quaternion.y();
  x(9) = quaternion.z();
  x.segment<3>(10) = partnerWrench.couple();
  x.segment<3>(13) = partnerWrench.force();
  for(std::size_t i = 0; i < 7 && i < q.size(); ++i) { x(16 + static_cast<Eigen::Index>(i)) = q[i]; }
  for(std::size_t i = 0; i < 7 && i < dq.size(); ++i) { x(23 + static_cast<Eigen::Index>(i)) = dq[i]; }
  return x;
}

Eigen::VectorXd CollabPegInHoleController::policyObservation() const
{
  const auto & r = robot();

  const auto & q = r.encoderValues();
  const auto & dq = r.encoderVelocities();

  const sva::PTransformd & X_0_tool = r.frame(tool_frame).position();
  // Rebuilt from the quaternion rather than used directly, so this is the same matrix the
  // training data's quaternion produced when RMB reconstructed it. The two agree, but going
  // through the quaternion is what makes that a fact rather than a hope.
  const Eigen::Matrix3d rotation = Eigen::Quaterniond(X_0_tool.rotation()).toRotationMatrix();

  sva::ForceVecd wrench = sva::ForceVecd::Zero();
  if(r.hasForceSensor(ft_sensor)) { wrench = r.forceSensor(ft_sensor).worldWrenchWithoutGravity(r); }

  // 7 + 7 + 9 + 6 = 29, and the order is RMB's --state_keys order:
  // measured_joint_pos, measured_joint_vel, measured_eef_pose, measured_eef_wrench.
  Eigen::VectorXd x = Eigen::VectorXd::Zero(29);
  for(std::size_t i = 0; i < 7 && i < q.size(); ++i) { x(static_cast<Eigen::Index>(i)) = q[i]; }
  for(std::size_t i = 0; i < 7 && i < dq.size(); ++i) { x(7 + static_cast<Eigen::Index>(i)) = dq[i]; }
  x.segment<3>(14) = X_0_tool.translation();
  // ROWS of the rotation matrix, not columns. See the header.
  x.segment<3>(17) = rotation.row(0);
  x.segment<3>(20) = rotation.row(1);
  // Force first, then couple -- the opposite of how ForceVecd stores them.
  x.segment<3>(23) = wrench.force();
  x.segment<3>(26) = wrench.couple();
  return x;
}

void CollabPegInHoleController::aimedPoint(const mc_rbdyn::Robot & robot,
                                           const std::string & body,
                                           const Eigen::Vector3d & point,
                                           Eigen::Vector3d & position,
                                           Eigen::Vector3d & velocity)
{
  const sva::PTransformd & X_0_b = robot.bodyPosW(body);
  // sva stores the rotation as world->body, so the transpose maps a body-frame offset back
  // into the world.
  const Eigen::Vector3d offsetW = X_0_b.rotation().transpose() * point;
  position = X_0_b.translation() + offsetW;
  const sva::MotionVecd & V_0_b = robot.bodyVelW(body);
  velocity = V_0_b.linear() + V_0_b.angular().cross(offsetW);
}

void CollabPegInHoleController::updateIntentReference(const Eigen::Vector3d & intent)
{
  const double interval = timeStep * policyDecimation_;
  // The derivative of a prediction, so it carries the prediction's error: filtered, and the
  // first step taken as zero rather than as a step from an arbitrary initial value.
  const Eigen::Vector3d raw = haveLastIntent_ ? Eigen::Vector3d((intent - lastIntent_) / interval)
                                              : Eigen::Vector3d(Eigen::Vector3d::Zero());
  policyRefVelocity_ = policyVelocityFilter_ * raw + (1.0 - policyVelocityFilter_) * policyRefVelocity_;
  policyIntent_ = intent;
  lastIntent_ = intent;
  haveLastIntent_ = true;
}

Eigen::Vector3d CollabPegInHoleController::intentForce()
{
  Eigen::Vector3d handPos = Eigen::Vector3d::Zero();
  Eigen::Vector3d handVel = Eigen::Vector3d::Zero();
  aimedPoint(robot(), policyRefBody_, policyRefPoint_, handPos, handVel);

  Eigen::Vector3d force = policyStiffness_.cwiseProduct(policyIntent_ - handPos)
                          + policyDamping_.cwiseProduct(policyRefVelocity_ - handVel);
  const double norm = force.norm();
  // The same saturation the simulated partner applies: a human is not an ideal spring, and
  // without it an initial offset produces a transient no arm could apply.
  policySaturated_ = norm > policyMaxForce_;
  if(policySaturated_ && norm > 1e-9) { force *= policyMaxForce_ / norm; }
  return force;
}

void CollabPegInHoleController::updatePolicyPartner()
{
  // `recording_` is the same gate the partner this policy replaces runs behind
  // (SimHumanPartner::update: `acting = phase == Recording`). Without it the policy starts
  // pushing the instant the end-effector is freed, while the arm is still at its home pose --
  // an observation nothing in the training set resembles. The predicted intent is then far
  // from the hand, person A's 520 N/m spring saturates at its 80 N ceiling immediately, and
  // the QP dies on the first tick of the Compliant state with "QLD fail code = 144".
  if(!policyEnabled_ || (!policy_ && !replay_) || !eeInSolver_ || !recording_)
  {
    // Zero it rather than leaving it: refAccel_ persists across ticks and compEETask->reset()
    // does not clear it, so a stale feed-forward would keep pushing after the episode.
    if(compEETask) { compEETask->refAccel(Eigen::Vector6d::Zero()); }
    policyForce_.setZero();
    policyAccel_.setZero();
    policyWrenchNorm_ = 0.0;
    // The intent's derivative must not survive a gap in which the arm kept moving: resuming
    // with a stale reference velocity injects a `kd vRef` term from before the pause.
    policyRefVelocity_.setZero();
    haveLastIntent_ = false;
    policySaturated_ = false;
    // The anchor is per-episode: it is taken again, from wherever the next episode's hand
    // starts, exactly as SimHumanPartner rebases its own plan at each episode start.
    haveReplayAnchor_ = false;
    return;
  }

  auto & r = robot();
  if(!policyJac_)
  {
    policyJac_ = std::make_unique<rbd::Jacobian>(r.mb(), policyBody_, policyPoint_);
  }
  const Eigen::Vector3d hole = r.frame(insertion_frame).position().translation();

  const bool policyStep = policyTick_ % policyDecimation_ == 0;
  if(policyOutputKind_ == "force")
  {
    if(policyStep)
    {
      const double interval = timeStep * policyDecimation_;
      // Not a ternary: the two Eigen expression types have no common type to collapse to.
      if(haveLastHole_) { holeVelocity_ = (hole - lastHole_) / interval; }
      else { holeVelocity_.setZero(); }
      lastHole_ = hole;
      haveLastHole_ = true;
      policyForce_ = policy_->predict(policyFeatures()).head<3>();
    }
  }
  else
  {
    // The *intent* is predicted at the policy rate and held; the *force* is recomputed every
    // tick from it. That asymmetry is the point: an impedance held constant between policy
    // steps is not a spring, it is a constant push, and it would not react to the arm moving
    // under it in the 9 ms between predictions. The demonstrating partner evaluated its own
    // spring every tick against a reference it interpolated -- this is the same shape.
    if(replay_ && !haveReplayAnchor_)
    {
      // Taken at the first recording tick, before anything has moved, because that is the
      // moment the recording's own plan was rebased. Taking it later would fold whatever the
      // dyad had already done into the offset.
      Eigen::Vector3d handPos = Eigen::Vector3d::Zero();
      Eigen::Vector3d handVel = Eigen::Vector3d::Zero();
      aimedPoint(r, policyRefBody_, policyRefPoint_, handPos, handVel);
      replay_->anchor(handPos);
      haveReplayAnchor_ = true;
      mc_rtc::log::info("[CollabPegInHoleController] replay anchored at [{:.4f} {:.4f} {:.4f}], {:.1f} mm from the "
                        "recording's own start",
                        handPos.x(), handPos.y(), handPos.z(), 1000.0 * replay_->offset().norm());
    }
    // One ternary, and everything after it is shared. A replay is worth running only if the
    // code it runs through is bit-for-bit the code a policy runs through; a second branch
    // below this line would make the result unfalsifiable.
    if(policyStep)
    {
      // +1: run() calls this BEFORE incrementing episodeTick_, and startEpisode sets it to
      // -1 so that the first recorded *sample* carries tick 0. So the tick this call will be
      // logged under is one ahead of the member, and indexing on the member directly would
      // play the recording back one control tick early.
      updateIntentReference(replay_ ? replay_->at(episodeTick_ + 1)
                                    : Eigen::Vector3d(policy_->predict(policyObservation()).head<3>()));
    }
    policyForce_ = intentForce();
  }
  ++policyTick_;

  // f -> joint torques at person A's contact point -> joint acceleration -> task frame.
  // Through joint space on purpose: it is what gives a pure force at an offset point the
  // right angular term at the tool frame.
  const auto nrDof = r.mb().nrDof();
  // rbd::Jacobian::jacobian(mb, mbc) is in the WORLD frame, so the policy's world-frame force
  // maps straight through and the task acceleration comes out world-frame too -- which is what
  // PositionTask::refAccel expects and what CompliantEndEffectorTask feeds it.
  Eigen::MatrixXd contactFull = Eigen::MatrixXd::Zero(6, nrDof);
  policyJac_->fullJacobian(r.mb(), policyJac_->jacobian(r.mb(), r.mbc()), contactFull);
  Eigen::Vector6d wrench6 = Eigen::Vector6d::Zero();
  wrench6.tail<3>() = policyForce_; // pure force, no couple: a point contact adds none
  const Eigen::VectorXd tau = contactFull.transpose() * wrench6;

  rbd::ForwardDynamics fd(r.mb());
  fd.computeH(r.mb(), r.mbc());
  const Eigen::VectorXd accel = fd.H().ldlt().solve(tau);

  Eigen::MatrixXd toolFull = Eigen::MatrixXd::Zero(6, nrDof);
  extWrenchJac_->fullJacobian(r.mb(), extWrenchJac_->jacobian(r.mb(), r.mbc()), toolFull);
  const Eigen::Vector6d taskAccel = toolFull * accel;

  // Gamma by hand: CompliantEndEffectorTask adds refAccel_ OUTSIDE the compliance matrix, so
  // without this the policy would drive axes the operator chose to keep stiff.
  const Eigen::Vector6d gamma = compEETask->getComplianceVector();
  policyAccel_ = gamma.asDiagonal() * taskAccel;
  compEETask->refAccel(policyAccel_);

  // What the trigger sees. The equivalent wrench at the tool frame, matching the units
  // externalWrench_ is in so the two can be added: the force is the same vector, the couple
  // is the moment the offset produces about the tool frame.
  const auto & X_0_b = r.mbc().bodyPosW[static_cast<std::size_t>(r.mb().bodyIndexByName(policyBody_))];
  // policyPoint_ is in BODY coordinates; rotate before adding, or the lever arm is wrong by
  // the body's orientation and the trigger term drifts with the arm's configuration.
  const Eigen::Vector3d contactWorld = X_0_b.rotation().transpose() * policyPoint_ + X_0_b.translation();
  const Eigen::Vector3d lever = contactWorld - r.frame(tool_frame).position().translation();
  const sva::ForceVecd policyWrench(lever.cross(policyForce_), policyForce_);
  policyWrenchNorm_ = policyWrench.vector().norm();
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
  // A double press would otherwise abandon the running episode mid-stream, leaving a recording
  // that never falls back to 0 in the log: close it properly first.
  if(recording_)
  {
    mc_rtc::log::warning("[CollabPegInHoleController] Episode {} was still recording - closing it before starting a "
                         "new one (it keeps the outcome '{}').",
                         episodeId_, episodeOutcome_);
    stopEpisode();
  }
  ++episodeId_;
  recording_ = true;
  // Per-episode, so every episode's policy steps fall on the same ticks. Left running across
  // episodes, the decimation phase would depend on how long the previous episode happened to
  // be, and the first prediction of an episode could be up to `policyDecimation_` ticks late.
  policyTick_ = 0;
  // run() increments before the logger samples the tick, so -1 here makes the first recorded
  // sample of the episode carry tick 0.
  episodeTick_ = -1;
  episodeLabel_ = "recording";
  episodeOutcome_ = "none";
  datastore().assign<int>("EpisodeId", episodeId_);
  datastore().assign<bool>("Recording", recording_);
  datastore().assign<std::string>("EpisodeLabel", episodeLabel_);
  datastore().assign<std::string>("EpisodeOutcome", episodeOutcome_);
  // NOTE: episodes deliberately stay inside the single session log rather than getting a file
  // each. Both ways of rolling the mc_rtc log over mid-run are unsafe here:
  //  - Logger::open(file, ...) writes no KeyAddedEvent for the already-registered entries, so the
  //    resulting .bin has data rows with no schema and segfaults mc_bin_utils / mc_bin_to_flat;
  //  - Logger::start(name, dt, true) re-declares the entries but names the file from a
  //    second-resolution timestamp, so a stop-then-start inside the same second silently
  //    truncates the episode that just finished.
  // The in-log markers below segment the session log unambiguously; scripts/extract_episodes.py
  // slices on them.
  mc_rtc::log::info("[CollabPegInHoleController] Episode {} started", episodeId_);
}

void CollabPegInHoleController::markEpisode(const std::string & label)
{
  if(!recording_)
  {
    mc_rtc::log::warning("[CollabPegInHoleController] 'Mark {}' pressed while no episode is recording - ignored (start "
                         "an episode first).",
                         label);
    return;
  }
  episodeOutcome_ = label;
  datastore().assign<std::string>("EpisodeOutcome", episodeOutcome_);
  mc_rtc::log::info("[CollabPegInHoleController] Episode {} labelled '{}'", episodeId_, label);
}

void CollabPegInHoleController::stopEpisode()
{
  if(!recording_)
  {
    mc_rtc::log::warning("[CollabPegInHoleController] 'Stop episode' pressed while nothing is recording - ignored.");
    return;
  }
  if(episodeOutcome_ == "none")
  {
    mc_rtc::log::warning("[CollabPegInHoleController] Episode {} stopped without an outcome - it will be dropped by "
                         "extract_episodes.py --only-success.",
                         episodeId_);
  }
  recording_ = false;
  episodeTick_ = -1;
  episodeLabel_ = "idle";
  // episodeOutcome_ is deliberately kept: it must stay valid for the samples of this episode and
  // is only cleared by the next startEpisode().
  datastore().assign<bool>("Recording", recording_);
  datastore().assign<std::string>("EpisodeLabel", episodeLabel_);
  mc_rtc::log::info("[CollabPegInHoleController] Episode {} stopped (outcome '{}')", episodeId_, episodeOutcome_);
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
      mc_rtc::log::warning("[CollabPegInHoleController] ExternalForcesObserver datastore calls unavailable - the "
                           "observer is not in the pipeline; residual-based compliance is disabled.");
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
    // The observer spells this pair differently from the plugin it replaces, but it means the
    // same thing: fold the F/T sensor into the estimate. Here it enters the momentum integral
    // as a known input rather than as a high-passed addition, so it also decides whether
    // EF_Estimator::getResidualOnly is person A alone or both people.
    if(datastore().has("EF_Estimator::isUsingFTSensorMeasurements")
       && !datastore().call<bool>("EF_Estimator::isUsingFTSensorMeasurements"))
    {
      datastore().call("EF_Estimator::toggleFTSensorMeasurements");
    }
    if(datastore().has("EF_Estimator::setGain"))
    {
      datastore().call<void, double>("EF_Estimator::setGain", residualGain_);
    }
  }
}

Eigen::VectorXd CollabPegInHoleController::datastoreVector(const std::string & key, Eigen::Index size) const
{
  // ExternalForcesObserver publishes its torque signals as datastore *calls*, not as stored
  // values the way the ExternalForcesEstimator plugin did - hence call<> rather than get<>.
  if(datastore().has(key))
  {
    return datastore().call<Eigen::VectorXd>(key);
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
      mc_rtc::gui::Label("External wrench norm (A+B)", [this]() { return externalWrench_.vector().norm(); }),
      mc_rtc::gui::Label("Residual wrench norm (A only)", [this]() { return computeResidualWrench().vector().norm(); }),
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
          "Wrench filter time (s)", [this]() { return wrenchFilterTime_; },
          [this](double v) { wrenchFilterTime_ = v; }),
      mc_rtc::gui::NumberInput(
          "Trigger averaging window (s)", [this]() { return triggerAveragingWindow_; },
          [this](double v) { triggerAveragingWindow_ = v; }),
      mc_rtc::gui::NumberInput(
          "Gain transition time (s)", [this]() { return gainTransitionTime_; },
          [this](double v) { gainTransitionTime_ = v; }));

  gui()->addElement({"Collab", "Episode"},
                    mc_rtc::gui::Label("Episode id", [this]() { return std::to_string(episodeId_); }),
                    mc_rtc::gui::Label("Recording", [this]() { return recording_ ? "yes" : "no"; }),
                    mc_rtc::gui::Label("Outcome", [this]() { return episodeOutcome_; }),
                    mc_rtc::gui::Label("Duration (s)", [this]() { return recording_ ? episodeTick_ * timeStep : 0.0; }),
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
  logger().addLogEntry("collab_tau_ext_clamped_dofs", this, [this]() -> double { return extTorqueClampedDofs_; });
  // Person A alone. Under the ExternalForcesObserver this is narrower than it was under the
  // ExternalForcesEstimator plugin: the F/T torques go into the momentum integral as a known
  // input, so what is left in the residual is the part the sensor does NOT explain. Under the
  // plugin the same key carried the raw residual, which contained both people. Logs recorded
  // before the migration are not comparable on this entry.
  logger().addLogEntry("collab_residual", this, [this, nrDof]() -> Eigen::VectorXd
                       { return datastoreVector("EF_Estimator::getResidualOnly", nrDof); });
  // Person B alone, in joint space: the observer's own sum(J^T R^T w) over the force sensors.
  logger().addLogEntry("collab_tau_ft", this, [this, nrDof]() -> Eigen::VectorXd
                       { return datastoreVector("EF_Estimator::getFTSensorTorques", nrDof); });

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
  // Sticky outcome: valid for every sample of the episode, so the extractor can filter on it.
  logger().addLogEntry("collab_episode_outcome", this, [this]() -> std::string { return episodeOutcome_; });
  logger().addLogEntry("collab_episode_tick", this, [this]() -> double { return static_cast<double>(episodeTick_); });
  logger().addLogEntry("collab_posture_compliant", this, [this]() -> bool { return postureCompliant_; });

  // --- Provenance: what the samples were recorded with. The frames/sensor are fixed at load
  // time; the tuning below is editable from the GUI while recording, so it is sampled per tick.
  logger().addLogEntry("collab_meta_tool_frame", this, [this]() -> std::string { return tool_frame; });
  logger().addLogEntry("collab_meta_ft_sensor", this, [this]() -> std::string { return ft_sensor; });
  logger().addLogEntry("collab_meta_insertion_frame", this, [this]() -> std::string { return insertion_frame; });
  logger().addLogEntry("collab_meta_trigger_source", this, [this]() -> std::string { return triggerSource_; });
  logger().addLogEntry("collab_meta_residual_gain", this, [this]() -> double { return residualGain_; });
  logger().addLogEntry("collab_meta_dual_force_enter", this, [this]() -> double { return dualForceEnter_; });
  logger().addLogEntry("collab_meta_dual_force_exit", this, [this]() -> double { return dualForceExit_; });
  logger().addLogEntry("collab_meta_wrench_filter_time", this, [this]() -> double { return wrenchFilterTime_; });

  // --- Dual-compliance state ---
  // --- Dual-compliance trigger (full external wrench: person A + person B) ---
  // --- Policy partner. Logged whether or not it is enabled, so an episode's record says
  // which of the two produced the demonstration rather than leaving it to the config file.
  logger().addLogEntry("collab_policy_active", this, [this]() -> double { return policyEnabled_ ? 1.0 : 0.0; });
  logger().addLogEntry("collab_policy_force", this, [this]() -> const Eigen::Vector3d & { return policyForce_; });
  logger().addLogEntry("collab_policy_accel", this, [this]() -> const Eigen::Vector6d & { return policyAccel_; });
  logger().addLogEntry("collab_policy_trigger", this, [this]() { return policyWrenchNorm_; });
  // The intent path's own state. Logged even on the force path, where it stays zero: a
  // channel that appears and disappears between runs is one the extractor has to special-case.
  logger().addLogEntry("collab_policy_intent", this, [this]() -> const Eigen::Vector3d & { return policyIntent_; });
  logger().addLogEntry("collab_policy_intent_velocity", this,
                       [this]() -> const Eigen::Vector3d & { return policyRefVelocity_; });
  // Distance from the predicted intent to the point the impedance pulls: the policy's own
  // tracking error, and the thing to read when the dyad moves but does not insert.
  logger().addLogEntry("collab_policy_intent_error", this,
                       [this]() -> double
                       {
                         if(policyOutputKind_ != "intent" || policyRefBody_.empty()) { return 0.0; }
                         Eigen::Vector3d position = Eigen::Vector3d::Zero();
                         Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
                         aimedPoint(robot(), policyRefBody_, policyRefPoint_, position, velocity);
                         return (policyIntent_ - position).norm();
                       });
  logger().addLogEntry("collab_policy_saturated", this, [this]() -> double { return policySaturated_ ? 1.0 : 0.0; });
  if(!policyReplayPath_.empty())
  {
    // Only when replaying, so a policy log cannot be mistaken for one. The offset says how
    // far this episode started from the recording's own start: near zero means the replay is
    // running in the episode it came from, which is the comparison worth making. `held` marks
    // the ticks past the end of the recording, where the last sample is being held.
    logger().addLogEntry("collab_policy_replay_offset", this,
                         [this]() -> double { return replay_ ? 1000.0 * replay_->offset().norm() : 0.0; });
    logger().addLogEntry("collab_policy_replay_held", this,
                         [this]() -> double { return replay_ && replay_->exhausted(episodeTick_ + 1) ? 1.0 : 0.0; });
  }
  if(logPolicyObservation_)
  {
    // Off by default -- 29 doubles per tick at 1 kHz is real log volume for something no run
    // needs. On, it is the only way to check the observation the controller assembles against
    // the one export_rmb.py wrote from the *same episode*: record a demonstration with this
    // set, and the two vectors must agree frame for frame. That comparison is what proves the
    // pose is rows-not-columns and the wrench force-first, which nothing else would report.
    logger().addLogEntry("collab_policy_observation", this,
                         [this]() -> Eigen::VectorXd { return policyObservation(); });
  }
  logger().addLogEntry("collab_ext_wrench", this, [this]() -> sva::ForceVecd { return externalWrench_; });
  logger().addLogEntry("collab_ext_wrench_norm", this, [this]() -> double { return externalWrench_.vector().norm(); });
  // The A-only counterpart of collab_ext_wrench, in the same frame and the same units, so the
  // two can be read against each other directly.
  logger().addLogEntry("collab_residual_wrench", this, [this]() -> sva::ForceVecd { return computeResidualWrench(); });
  logger().addLogEntry("collab_residual_wrench_norm", this,
                       [this]() -> double { return computeResidualWrench().vector().norm(); });

  logger().addLogEntry("collab_ee_free", this, [this]() -> bool { return dualFree_; });
  // How far the plate has tilted off level, in degrees. The whole point of the up-vector task,
  // and the number to compare against the 46-53 deg swing that preceded it.
  logger().addLogEntry("collab_ee_tilt_deg", this,
                       [this]() -> double { return eeUpVectorTask ? toolTilt() * 180.0 / M_PI : 0.0; });
  logger().addLogEntry("collab_trigger_norm", this, [this]() -> double { return currentWrenchNorm_; });
  logger().addLogEntry("collab_trigger_norm_filtered", this, [this]() -> double { return filteredWrenchNorm_; });
  logger().addLogEntry("collab_trigger_norm_smoothed", this, [this]() -> double { return triggerWrenchNorm_; });
  logger().addLogEntry("collab_ee_stiffness", this, [this]() -> double { return currentEEStiffness_; });
  logger().addLogEntry("collab_ee_orientation_compliant", this, [this]() -> bool { return eeOrientationCompliant_; });
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
