#pragma once

#include <Eigen/Core>

#include <string>

namespace collab
{

/** Whatever runs a trained policy's forward pass, behind one call.
 *
 * There are two policies in this project and they have nothing in common but this signature.
 * `PolicyPartner` reads the hand-rolled YAML `cpih.policy.mlp` writes: 30 features in, a 3-D
 * world-frame force out. `OnnxBackend` runs a RoboManipBaselines checkpoint exported to ONNX:
 * 29 state values in, a 9-D pose out, of which the controller uses the translation. Both are
 * state-feedback maps with no history and no clock.
 *
 * The seam exists so the *observation assembly*, the impedance that turns an intent into a
 * force, and the injection into `CompliantEndEffectorTask::refAccel_` are written once above
 * it. That matters because the next backend is already known: a nanomsg request/reply pair
 * serving a policy from Python, which is the only way ACT or a diffusion policy can run --
 * action chunking and iterative denoising have no representation in either file format. When
 * that arrives it should be a new file, not a second integration.
 *
 * ### Dimensions come from the policy, never from the config
 *
 * `nIn` and `nOut` are read out of the loaded artefact. A controller that is *told* how wide
 * its policy is will one day be told wrong, and a forward pass on a mis-sized vector is
 * either a crash in the best case or a plausible wrong answer in the worst.
 */
struct PolicyBackend
{
  virtual ~PolicyBackend() = default;

  /** Forward pass. `x` must be `nIn()` long, in the order the policy was trained on. */
  virtual Eigen::VectorXd predict(const Eigen::VectorXd & x) const = 0;

  virtual Eigen::Index nIn() const noexcept = 0;
  virtual Eigen::Index nOut() const noexcept = 0;

  /** The simulated partner this policy stands in for, named by the file itself.
   *
   * Read at config-generation time to decide who stops pushing. Not a flag: an evaluation in
   * which the imitated partner is still doing the task scores well and means nothing.
   */
  virtual const std::string & partner() const noexcept = 0;
};

} // namespace collab
