#include "CollabPegInHoleController.h"

#include <mc_control/mc_global_controller.h>

int main(int argc, char * argv[])
{
  if(argc < 2)
  {
    mc_rtc::log::critical("Should provide a configuration file for mc_rtc");
    return 1;
  }
  mc_rtc::log::critical("LOADING {}", argv[1]);
  mc_control::MCGlobalController gc(argv[1]);

  const auto & mb = gc.robot().mb();
  const auto & mbc = gc.robot().mbc();
  const auto & rjo = gc.ref_joint_order();
  std::vector<double> initq;
  for(const auto & jn : rjo)
  {
    for(const auto & qi : mbc.q[static_cast<unsigned int>(mb.jointIndexByName(jn))])
    {
      initq.push_back(qi);
    }
  }

  std::vector<double> qEnc(initq.size(), 0);
  std::vector<double> alphaEnc(initq.size(), 0);
  // Joint torques are a real measurement on hardware and under mc_mujoco (both call
  // setJointTorques); feed them here too, otherwise tauIn stays empty in the log and the
  // ExternalForcesObserver - if its torque_source_type is a measurement - has no input.
  std::vector<double> tauMeas(initq.size(), 0);
  auto simulateSensors = [&]()
  {
    auto & robot = gc.robot();
    for(unsigned i = 0; i < robot.refJointOrder().size(); i++)
    {
      auto jIdx = robot.jointIndexInMBC(i);
      if(jIdx != -1)
      {
        auto jointIndex = static_cast<unsigned>(jIdx);
        qEnc[i] = robot.mbc().q[jointIndex][0];
        alphaEnc[i] = robot.mbc().alpha[jointIndex][0];
        const auto & tau = robot.mbc().jointTorque[jointIndex];
        tauMeas[i] = tau.empty() ? 0.0 : tau[0];
      }
    }
    gc.setEncoderValues(qEnc);
    gc.setEncoderVelocities(alphaEnc);
    gc.setJointTorques(tauMeas);
    if(gc.robot().hasBodySensor("FloatingBase"))
    {
      gc.setSensorPositions({{"FloatingBase", robot.posW().translation()}});
      gc.setSensorOrientations({{"FloatingBase", Eigen::Quaterniond{robot.posW().rotation()}}});
    }
  };

  gc.setEncoderValues(qEnc);
  gc.init(initq, gc.robot().module().default_attitude());
  gc.running = true;

  auto tick = [&](size_t n)
  {
    for(size_t i = 0; i < n; ++i)
    {
      simulateSensors();
      if(!gc.run())
      {
        return false;
      }
    }
    return true;
  };

  if(!tick(10))
  {
    return 1;
  }

  // Exercise the episode-recording path the operator drives from the GUI: two demonstrations
  // with opposite outcomes, plus a double "Start episode" that must close the running one
  // rather than abandon it. The logs it produces feed scripts/extract_episodes.py.
  auto & ctl = static_cast<CollabPegInHoleController &>(gc.controller());
  ctl.startEpisode();
  if(!tick(200))
  {
    return 1;
  }
  ctl.markEpisode("success");
  if(!tick(50))
  {
    return 1;
  }
  ctl.stopEpisode();
  if(!tick(20))
  {
    return 1;
  }
  ctl.startEpisode();
  if(!tick(100))
  {
    return 1;
  }
  ctl.markEpisode("failure");
  if(!tick(5)) // the outcome is sampled per tick: a mark needs at least one tick to reach the log
  {
    return 1;
  }
  ctl.startEpisode(); // double start: closes episode 2, opens episode 3
  if(!tick(150))
  {
    return 1;
  }
  ctl.markEpisode("success");
  ctl.stopEpisode();
  if(!tick(10))
  {
    return 1;
  }
  ctl.stopEpisode(); // no-op: nothing is recording

  return 0;
}
