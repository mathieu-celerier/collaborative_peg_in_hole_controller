#include "PolicyPartner.h"

#include <mc_rtc/logging.h>

#include <cmath>

namespace collab
{

//! Must match cpih.policy.mlp.FORMAT_VERSION. Refused rather than guessed at: an older
//! loader reading a newer layout produces a running network, not an error.
static constexpr int kFormatVersion = 1;

Eigen::VectorXd PolicyPartner::vector(const mc_rtc::Configuration & config, const std::string & key)
{
  std::vector<double> values = config(key);
  return Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
}

void PolicyPartner::load(const std::string & path)
{
  mc_rtc::Configuration config(path);

  const int version = config("format_version", -1);
  if(version != kFormatVersion)
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] {}: format version {}, this build reads {}. Re-export the "
                                 "policy from the learning repo (scripts/train_bc.py).",
                                 path, version, kFormatVersion);
  }

  partner_ = static_cast<std::string>(config("partner"));
  activation_ = static_cast<std::string>(config("activation", std::string("tanh")));
  if(activation_ != "tanh" && activation_ != "relu")
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] {}: unknown activation '{}'", path, activation_);
  }
  withPhase_ = config("with_phase", false);
  featureNames_ = static_cast<std::vector<std::string>>(config("feature_names"));

  auto normalisation = config("normalisation");
  xMean_ = vector(normalisation, "x_mean");
  xScale_ = vector(normalisation, "x_scale");
  yMean_ = vector(normalisation, "y_mean");
  yScale_ = vector(normalisation, "y_scale");

  layers_.clear();
  for(const auto & entry : config("layers"))
  {
    const auto rows = static_cast<Eigen::Index>(static_cast<int>(entry("rows")));
    const auto cols = static_cast<Eigen::Index>(static_cast<int>(entry("cols")));
    const Eigen::VectorXd flat = vector(entry, "weight");
    if(flat.size() != rows * cols)
    {
      mc_rtc::log::error_and_throw("[PolicyPartner] {}: layer declares {}x{} but carries {} weights", path, rows,
                                   cols, flat.size());
    }
    Layer layer;
    // Row-major on the wire, which is how numpy flattened it. Reading it column-major here
    // is the classic transpose bug, and the only thing that catches it is the check pair
    // below -- a square hidden layer would otherwise load and run without complaint.
    layer.weight = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        flat.data(), rows, cols);
    layer.bias = vector(entry, "bias");
    if(layer.bias.size() != rows)
    {
      mc_rtc::log::error_and_throw("[PolicyPartner] {}: layer has {} rows but {} biases", path, rows,
                                   layer.bias.size());
    }
    layers_.push_back(std::move(layer));
  }
  if(layers_.empty())
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] {}: no layers", path);
  }
  if(nOut() != 3)
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] {}: the policy emits {} values; a partner applies a pure force "
                                 "at a point, so it must emit exactly 3.",
                                 path, nOut());
  }
  if(xMean_.size() != nIn() || xScale_.size() != nIn())
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] {}: {} inputs but {} normalisation entries", path, nIn(),
                                 xMean_.size());
  }
  if(static_cast<Eigen::Index>(featureNames_.size()) != nIn())
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] {}: {} inputs but {} feature names", path, nIn(),
                                 featureNames_.size());
  }

  // --- The self-check. See the header for why this is not optional. ---
  auto check = config("check");
  const Eigen::VectorXd checkInput = vector(check, "input");
  const Eigen::VectorXd checkOutput = vector(check, "output");
  if(checkInput.size() != nIn())
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] {}: check pair has {} inputs, the network takes {}", path,
                                 checkInput.size(), nIn());
  }
  const Eigen::Vector3d got = (*this)(checkInput);
  const double error = (got - checkOutput).cwiseAbs().maxCoeff();
  if(error > 1e-5)
  {
    layers_.clear();
    mc_rtc::log::error_and_throw("[PolicyPartner] {}: failed its own check pair by {:.3g} N (expected [{:.4f}, "
                                 "{:.4f}, {:.4f}], got [{:.4f}, {:.4f}, {:.4f}]). The weights and this forward "
                                 "pass disagree -- a transposed matrix or a normalisation applied in the wrong "
                                 "direction. The policy is NOT the one that was trained.",
                                 path, error, checkOutput(0), checkOutput(1), checkOutput(2), got(0), got(1),
                                 got(2));
  }

  mc_rtc::log::info("[PolicyPartner] loaded '{}' from {}: {} inputs -> {} outputs, {} layers, check pair matched "
                    "to {:.1e} N",
                    partner_, path, nIn(), nOut(), layers_.size(), error);
}

void PolicyPartner::checkFeatureOrder(const std::vector<std::string> & expected) const
{
  if(expected.size() != featureNames_.size())
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] the policy takes {} features, this controller builds {}. The "
                                 "feature contract in cpih.policy.features has changed on one side only.",
                                 featureNames_.size(), expected.size());
  }
  for(std::size_t i = 0; i < expected.size(); ++i)
  {
    if(expected[i] != featureNames_[i])
    {
      mc_rtc::log::error_and_throw("[PolicyPartner] feature {} is '{}' in the policy but '{}' here. A policy fed "
                                   "a correctly sized vector in the wrong order emits a plausible wrong force "
                                   "and nothing reports it.",
                                   i, featureNames_[i], expected[i]);
    }
  }
}

Eigen::Vector3d PolicyPartner::operator()(const Eigen::VectorXd & x) const
{
  if(layers_.empty())
  {
    mc_rtc::log::error_and_throw("[PolicyPartner] evaluated before load()");
  }
  Eigen::VectorXd h = (x - xMean_).cwiseQuotient(xScale_);
  for(std::size_t i = 0; i < layers_.size(); ++i)
  {
    h = layers_[i].weight * h + layers_[i].bias;
    if(i + 1 < layers_.size())
    {
      // Written out rather than as a ternary: the two Eigen expression types differ, so the
      // conditional operator has no common type to collapse to.
      if(activation_ == "tanh") { h = h.array().tanh().matrix(); }
      else { h = h.cwiseMax(0.0); }
    }
  }
  return h.cwiseProduct(yScale_) + yMean_;
}

} // namespace collab
