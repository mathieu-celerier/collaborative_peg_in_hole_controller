#pragma once

#include "PolicyBackend.h"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace collab
{

/** A RoboManipBaselines policy, exported to ONNX and run in this process.
 *
 * Written by `scripts/export_onnx.py` in the learning repo. Two properties of that file are
 * what make this class short, and both are deliberate:
 *
 * **The graph takes raw SI in and raw SI out.** RMB normalises the state and denormalises the
 * action outside the network, from a `model_meta_info.pkl` written beside the checkpoint. The
 * exporter folds both affine transforms into the graph instead, so there is no second file to
 * keep in step and nothing here that could apply a normalisation backwards.
 *
 * **The file carries its own check pair.** `check_input` and `check_output` are a real state
 * and the action the exporter measured the graph producing for it. The constructor re-runs it
 * and throws on disagreement. Every interesting failure in this path -- a graph that lost its
 * weights to an external-data sidecar, a runtime version that changed an operator, an
 * observation assembled in the wrong order upstream -- otherwise produces a policy that runs
 * happily and emits a plausible wrong pose.
 *
 * ### Why in-process, and not over a socket
 *
 * The alternative was serving the policy from Python. In a headless batch that is a second
 * process per simulator launch across 54 configurations, with a timeout policy to design for
 * the case where it stops answering mid-episode. Running the graph here is deterministic,
 * tick-synchronous and has nothing to orchestrate. It stops working for ACT and diffusion
 * policies, which is what `PolicyBackend`'s second implementation will be for.
 */
struct OnnxBackend final : public PolicyBackend
{
  /** Load, read the metadata, and run the file's own check pair.
   *
   * Throws on anything unexpected. There is no degraded mode on purpose: a policy that cannot
   * be shown to be the one that was trained is worse than no policy, because the run it
   * produces looks like a result.
   */
  explicit OnnxBackend(const std::string & path);

  Eigen::VectorXd predict(const Eigen::VectorXd & x) const override;

  Eigen::Index nIn() const noexcept override { return nIn_; }
  Eigen::Index nOut() const noexcept override { return nOut_; }
  const std::string & partner() const noexcept override { return partner_; }

  /** ``intent`` (the action is a pose, and its translation is where the partner's plan asked
   * that hand to be) or ``wrench`` (the action is the force itself). The controller does
   * completely different things with the two, so it is read from the file rather than
   * configured. */
  const std::string & outputKind() const noexcept { return outputKind_; }

private:
  std::string metadataEntry(const std::string & key, bool required = true) const;
  void runCheckPair();

  // Declared before the session: Ort::Session holds a reference to the env, so the env has to
  // outlive it and therefore be destroyed after it.
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::string path_;
  std::string partner_;
  std::string outputKind_;
  Eigen::Index nIn_ = 0;
  Eigen::Index nOut_ = 0;

  // Ort takes `const char *` and does not copy, so the strings have to be owned here.
  std::string inputName_;
  std::string outputName_;
};

} // namespace collab
