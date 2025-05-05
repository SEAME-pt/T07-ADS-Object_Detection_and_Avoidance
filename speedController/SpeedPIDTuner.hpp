// SpeedPIDTuner.hpp**
#pragma once

#include <tuple>

std::tuple<float, float, float> auto_tune_pid(float dt = 0.1f, float sim_time = 10.0f, float v_target = 2.0f);
