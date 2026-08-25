#pragma once

#include <mc_rtc/Configuration.h>

#include <Eigen/Core>

#include <string>
#include <vector>

namespace collab
{

/** A behaviour-cloned partner: the force person A applied, as a function of what they saw.
 *
 * This is the evaluation half of `cpih.policy` in the learning repo. That side trains an MLP
 * in torch and writes it out as plain matrices; this side reloads them and runs the forward
 * pass in Eigen. Deliberately not ONNX or TorchScript: at this size the network is four
 * matrix products, and a new runtime in the nix superbuild costs rebuild hours for nothing.
 *
 * ### Where the output goes, and where it must not
 *
 * The force is **not** applied to the robot as an external wrench. Nothing pushes the arm at
 * test time -- there is no person A any more. It enters at the *task* level, converted to a
 * task-frame acceleration and handed to the compliant end-effector task's `refAccel`, which
 * is the same place the demonstrating partner's force arrived through the estimator's
 * `disturbance_` term. See CollabPegInHoleController::updatePolicyPartner.
 *
 * It must not go through `setCompensationTorques`. That channel exists so the QP can enforce
 * torque and joint limits *under* a real disturbance, and the compliant tasks read it in
 * preference to the estimate -- so writing a commanded interaction force there would both
 * corrupt the safety accounting and hijack the disturbance term, neither of them the intent.
 *
 * ### The check pair
 *
 * The file carries one real (features, force) pair recorded at export time, and `load`
 * refuses the policy unless the forward pass reproduces it. A transposed weight, a
 * column-major read, or a normalisation applied in the wrong direction all produce a network
 * that runs happily and emits a plausible wrong force. Without this that is found by watching
 * an arm behave oddly and not knowing why.
 */
struct PolicyPartner
{
  /** Load and self-check. Throws on any inconsistency -- there is no degraded mode: a policy
   * that cannot be trusted to be the one that was trained is worse than no policy. */
  void load(const std::string & path);

  bool loaded() const noexcept { return !layers_.empty(); }

  /** Forward pass. `x` must be `nIn()` long and in `featureNames()` order. */
  Eigen::Vector3d operator()(const Eigen::VectorXd & x) const;

  Eigen::Index nIn() const noexcept { return layers_.empty() ? 0 : layers_.front().weight.cols(); }
  Eigen::Index nOut() const noexcept { return layers_.empty() ? 0 : layers_.back().weight.rows(); }

  const std::vector<std::string> & featureNames() const noexcept { return featureNames_; }
  const std::string & partner() const noexcept { return partner_; }
  bool withPhase() const noexcept { return withPhase_; }

  /** Compare the file's column order against what the caller builds.
   *
   * Throws on a mismatch rather than warning. A policy handed a correctly *sized* vector in
   * the wrong order is the one failure that produces no symptom until the arm moves.
   */
  void checkFeatureOrder(const std::vector<std::string> & expected) const;

private:
  struct Layer
  {
    Eigen::MatrixXd weight; // (out, in)
    Eigen::VectorXd bias;
  };

  static Eigen::VectorXd vector(const mc_rtc::Configuration & config, const std::string & key);

  std::vector<Layer> layers_;
  std::string activation_ = "tanh";
  std::string partner_;
  std::vector<std::string> featureNames_;
  bool withPhase_ = false;
  Eigen::VectorXd xMean_, xScale_, yMean_, yScale_;
};

} // namespace collab
