#include "IntentReplay.h"

#include <mc_rtc/Configuration.h>
#include <mc_rtc/logging.h>

#include <cmath>
#include <filesystem>

namespace collab
{

namespace
{

//: Written by the learning repo's scripts/export_replay.py. A file without it is some other
//: JSON, and playing back whatever numbers it happens to hold is worse than refusing.
constexpr auto FORMAT = "cpih-intent-replay";
constexpr int FORMAT_VERSION = 1;

} // namespace

IntentReplay::IntentReplay(const std::string & path, double timeStep)
{
  if(!std::filesystem::exists(path))
  {
    mc_rtc::log::error_and_throw("[IntentReplay] no such file: {}", path);
  }
  mc_rtc::Configuration conf(path);

  const std::string format = conf("format", std::string{});
  const int version = conf("format_version", 0);
  if(format != FORMAT)
  {
    mc_rtc::log::error_and_throw("[IntentReplay] {} is not a replay file: format is '{}', expected '{}'", path, format,
                                 FORMAT);
  }
  if(version != FORMAT_VERSION)
  {
    mc_rtc::log::error_and_throw("[IntentReplay] {} is format_version {}, this controller reads {}", path, version,
                                 FORMAT_VERSION);
  }

  conf("partner", partner_);
  conf("source", source_);
  conf("rate", rate_);
  origin_ = conf("origin", Eigen::Vector3d(Eigen::Vector3d::Zero()));
  anchor_ = origin_;
  if(partner_.empty())
  {
    mc_rtc::log::error_and_throw("[IntentReplay] {} names no partner, so nothing decides who stops pushing", path);
  }

  // A whole number of control ticks per sample, or the replay drifts out of phase with the
  // episode it is meant to reproduce. Refused rather than resampled: an interpolation here
  // would be a second thing under test in an experiment whose whole point is to have only one.
  if(!(rate_ > 0.0) || !(timeStep > 0.0))
  {
    mc_rtc::log::error_and_throw("[IntentReplay] {}: rate {} Hz at timestep {} s is not usable", path, rate_, timeStep);
  }
  const double ticks = 1.0 / (rate_ * timeStep);
  stride_ = static_cast<int>(std::lround(ticks));
  if(stride_ < 1 || std::abs(ticks - stride_) > 1e-6)
  {
    mc_rtc::log::error_and_throw("[IntentReplay] {}: {} Hz is not a whole number of {} s control ticks per sample "
                                 "({:.4f}). Re-export at a rate that divides the control rate.",
                                 path, rate_, timeStep, ticks);
  }

  std::vector<std::vector<double>> rows = conf("reference");
  if(rows.empty())
  {
    mc_rtc::log::error_and_throw("[IntentReplay] {} holds an empty reference", path);
  }
  reference_.reserve(rows.size());
  for(std::size_t i = 0; i < rows.size(); ++i)
  {
    if(rows[i].size() != 3)
    {
      mc_rtc::log::error_and_throw("[IntentReplay] {}: reference row {} has {} values, expected 3", path, i,
                                   rows[i].size());
    }
    reference_.emplace_back(rows[i][0], rows[i][1], rows[i][2]);
  }

  mc_rtc::log::info("[IntentReplay] {} samples of '{}' at {:.0f} Hz ({:.2f} s), from {}", reference_.size(), partner_,
                    rate_, reference_.size() / rate_, source_.empty() ? std::string("an unnamed log") : source_);
}

Eigen::Vector3d IntentReplay::at(int episodeTick) const
{
  const int index = episodeTick <= 0 ? 0 : episodeTick / stride_;
  const auto last = static_cast<int>(reference_.size()) - 1;
  return reference_[static_cast<std::size_t>(index < last ? index : last)] + (anchor_ - origin_);
}

bool IntentReplay::exhausted(int episodeTick) const noexcept
{
  return episodeTick > 0 && episodeTick / stride_ >= static_cast<int>(reference_.size());
}

} // namespace collab
