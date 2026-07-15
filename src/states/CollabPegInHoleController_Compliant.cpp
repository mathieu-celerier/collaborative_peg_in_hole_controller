#include "CollabPegInHoleController_Compliant.h"

#include "../CollabPegInHoleController.h"

void CollabPegInHoleController_Compliant::configure(const mc_rtc::Configuration & config) {}

void CollabPegInHoleController_Compliant::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
  ctl.switchToCompliantState();
}

bool CollabPegInHoleController_Compliant::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
  if(ctl.requestedState() == "Initial")
  {
    output("GoToInitial");
    return true;
  }
  // Force-triggered dual compliance: free the end-effector while a force is applied.
  ctl.updateDualCompliance();
  return false;
}

void CollabPegInHoleController_Compliant::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<CollabPegInHoleController &>(ctl_);
}

EXPORT_SINGLE_STATE("CollabPegInHoleController_Compliant", CollabPegInHoleController_Compliant)
