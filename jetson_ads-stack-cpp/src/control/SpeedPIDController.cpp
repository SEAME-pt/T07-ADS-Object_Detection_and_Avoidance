// SpeedPIDController.cpp**
#include "SpeedPIDController.hpp"
#include <algorithm>

SpeedPIDController::SpeedPIDController(float kp, float ki, float kd, float pwm_min, float pwm_max)
    : kp_(kp), ki_(ki), kd_(kd), pwm_min_(pwm_min), pwm_max_(pwm_max),
      prev_error_(0.0f), integral_(0.0f) {}

void SpeedPIDController::reset() {
    prev_error_ = 0.0f;
    integral_ = 0.0f;
}

float SpeedPIDController::update(float v_current, float v_target, float dt) {
    float error = v_target - v_current;
    integral_ += error * dt;
    float derivative = (error - prev_error_) / dt;
    prev_error_ = error;

    float output = kp_ * error + ki_ * integral_ + kd_ * derivative;
    return std::clamp(output, pwm_min_, pwm_max_);
}