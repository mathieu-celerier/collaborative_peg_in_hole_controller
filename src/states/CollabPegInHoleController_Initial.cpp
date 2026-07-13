#include "CollabPegInHoleController_Initial.h"

#include "../CollabPegInHoleController.h"

void CollabPegInHoleController_Initial::configure(const mc_rtc::Configuration & config) {}

void CollabPegInHoleController_Initial::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
}

bool CollabPegInHoleController_Initial::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
  output("OK");
  return true;
}

void CollabPegInHoleController_Initial::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
}

EXPORT_SINGLE_STATE("CollabPegInHoleController_Initial", CollabPegInHoleController_Initial)
