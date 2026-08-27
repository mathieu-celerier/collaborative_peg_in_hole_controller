#include "OnnxBackend.h"

#include <mc_rtc/Configuration.h>
#include <mc_rtc/logging.h>

#include <array>
#include <filesystem>

namespace collab
{

namespace
{

//: Written by scripts/export_onnx.py. A file without it is some other ONNX model, and the
//: controller would have no idea what its columns mean.
constexpr auto FORMAT = "cpih-rmb-onnx";
constexpr auto FORMAT_VERSION = "1";

//: The check pair is float32 through the same graph on both sides, so the only difference is
//: the order the runtime schedules its operators in. 1e-5 is loose enough for that and far
//: tighter than any real mistake.
constexpr double CHECK_TOLERANCE = 1e-5;

/** A JSON array of numbers, as it was written into the metadata, as an Eigen vector.
 *
 * mc_rtc's Configuration reads JSON, so this needs no parser of its own.
 */
Eigen::VectorXd parseVector(const std::string & json, const std::string & what)
{
  std::vector<double> values;
  try
  {
    values = mc_rtc::Configuration::fromData(json).operator std::vector<double>();
  }
  catch(const std::exception & e)
  {
    mc_rtc::log::error_and_throw("[OnnxBackend] {} is not a JSON array of numbers: {}", what, e.what());
  }
  return Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
}

} // namespace

OnnxBackend::OnnxBackend(const std::string & path) : path_(path)
{
  if(!std::filesystem::exists(path))
  {
    mc_rtc::log::error_and_throw("[OnnxBackend] no such policy file: {}", path);
  }

  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CollabPegInHolePolicy");
  Ort::SessionOptions options;
  // One thread, and no CUDA. This runs inside the control loop of a 1 kHz controller: a
  // thread pool spun up per inference costs more than the four matrix products it parallelises,
  // and a GPU transfer costs more again. The network is small enough that this is not close.
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), options);

  const auto format = metadataEntry("format");
  if(format != FORMAT)
  {
    mc_rtc::log::error_and_throw(
        "[OnnxBackend] {} declares format '{}', expected '{}'. This controller reads the files "
        "scripts/export_onnx.py writes, not arbitrary ONNX models: it has to know what the "
        "columns of the input mean.",
        path, format, FORMAT);
  }
  const auto version = metadataEntry("format_version");
  if(version != FORMAT_VERSION)
  {
    mc_rtc::log::error_and_throw(
        "[OnnxBackend] {} is format version {}, and this build reads version {}. Refusing "
        "rather than guessing at a layout that has changed.",
        path, version, FORMAT_VERSION);
  }

  partner_ = metadataEntry("partner");
  outputKind_ = metadataEntry("output_kind");
  if(outputKind_ != "intent" && outputKind_ != "wrench")
  {
    mc_rtc::log::error_and_throw("[OnnxBackend] unknown output_kind '{}' in {}", outputKind_, path);
  }

  if(session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
  {
    mc_rtc::log::error_and_throw("[OnnxBackend] {} has {} input(s) and {} output(s), expected 1 of each",
                                 path, session_->GetInputCount(), session_->GetOutputCount());
  }
  Ort::AllocatorWithDefaultOptions allocator;
  inputName_ = session_->GetInputNameAllocated(0, allocator).get();
  outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

  // Shapes are (batch, width) with batch dynamic, so the width is the trailing entry. Read it
  // off the graph rather than from the metadata: the metadata says what the exporter believed,
  // the graph is what will actually run.
  const auto inShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const auto outShape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if(inShape.size() != 2 || outShape.size() != 2)
  {
    mc_rtc::log::error_and_throw("[OnnxBackend] {} has rank {} input and rank {} output, expected 2 and 2",
                                 path, inShape.size(), outShape.size());
  }
  nIn_ = static_cast<Eigen::Index>(inShape[1]);
  nOut_ = static_cast<Eigen::Index>(outShape[1]);

  runCheckPair();

  mc_rtc::log::info("[OnnxBackend] loaded {} ({} -> {}, {}) for partner '{}', check pair verified",
                    path, nIn_, nOut_, outputKind_, partner_);
}

std::string OnnxBackend::metadataEntry(const std::string & key, bool required) const
{
  Ort::AllocatorWithDefaultOptions allocator;
  auto value = session_->GetModelMetadata().LookupCustomMetadataMapAllocated(key.c_str(), allocator);
  if(value == nullptr)
  {
    if(!required) { return {}; }
    mc_rtc::log::error_and_throw(
        "[OnnxBackend] {} carries no '{}' metadata entry. Was it written by "
        "scripts/export_onnx.py, or exported straight out of torch?",
        path_, key);
  }
  return std::string(value.get());
}

void OnnxBackend::runCheckPair()
{
  const auto input = parseVector(metadataEntry("check_input"), "check_input");
  const auto expected = parseVector(metadataEntry("check_output"), "check_output");
  if(input.size() != nIn_ || expected.size() != nOut_)
  {
    mc_rtc::log::error_and_throw(
        "[OnnxBackend] the check pair in {} is {}x{}, but the graph is {}x{}. The metadata and "
        "the weights are not from the same export.",
        path_, input.size(), expected.size(), nIn_, nOut_);
  }
  const Eigen::VectorXd produced = predict(input);
  const double error = (produced - expected).cwiseAbs().maxCoeff();
  if(!(error <= CHECK_TOLERANCE))
  {
    mc_rtc::log::error_and_throw(
        "[OnnxBackend] {} does not reproduce its own check pair: worst component off by {:.3g}, "
        "tolerance {:g}. The policy that would run is not the policy that was trained; refusing "
        "to use it.",
        path_, error, CHECK_TOLERANCE);
  }
}

Eigen::VectorXd OnnxBackend::predict(const Eigen::VectorXd & x) const
{
  if(x.size() != nIn_)
  {
    mc_rtc::log::error_and_throw("[OnnxBackend] policy takes {} inputs, got {}", nIn_, x.size());
  }
  // The graph is float32; the controller is double throughout. Convert here rather than
  // anywhere else, so there is exactly one place where the precision changes.
  const Eigen::VectorXf in = x.cast<float>();
  std::array<int64_t, 2> inShape{1, nIn_};
  auto tensor = Ort::Value::CreateTensor<float>(memory_, const_cast<float *>(in.data()),
                                                static_cast<size_t>(in.size()), inShape.data(),
                                                inShape.size());
  const char * inputNames[] = {inputName_.c_str()};
  const char * outputNames[] = {outputName_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &tensor, 1, outputNames, 1);
  const float * data = outputs.front().GetTensorData<float>();
  return Eigen::Map<const Eigen::VectorXf>(data, nOut_).cast<double>();
}

} // namespace collab
