#include "CollabPegInHoleController_Initial.h"

#include "../CollabPegInHoleController.h"

void CollabPegInHoleController_Initial::configure(const mc_rtc::Configuration & config) {}

void CollabPegInHoleController_Initial::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
  ctl.switchToInitialState();
}

bool CollabPegInHoleController_Initial::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
  if(ctl.requestedState() == "Compliant")
  {
    output("GoToCompliant");
    return true;
  }
  return false;
}

void CollabPegInHoleController_Initial::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
}

EXPORT_SINGLE_STATE("CollabPegInHoleController_Initial", CollabPegInHoleController_Initial)
