// SpeedPIDTuner.hpp**
#pragma once

#include <tuple>
static auto simulateVelocity(float v_current, float pwm_input, float dt) -> float;
static auto evaluatePid(float kp, float ki, float kd, float dt, float sim_time, float v_target) -> float;
auto autoTunePid(float dt = 0.1F, float sim_time = 10.0F, float v_target = 2.0F) -> std::tuple<float, float, float>;
