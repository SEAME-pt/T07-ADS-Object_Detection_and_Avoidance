// SpeedPIDTuner.hpp**
#pragma once

#include <tuple>
static float simulate_velocity(float v_current, float pwm_input, float dt);
static float evaluate_pid(float kp, float ki, float kd, float dt, float sim_time, float v_target);
std::tuple<float, float, float> auto_tune_pid(float dt = 0.1f, float sim_time = 10.0f, float v_target = 2.0f);
