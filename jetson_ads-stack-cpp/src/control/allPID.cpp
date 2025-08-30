// SpeedPIDController.hpp**
#pragma once

class SpeedPIDController {
public:
    SpeedPIDController(float kp, float ki, float kd, float pwm_min, float pwm_max);

    auto update(float v_current, float v_target, float dt) -> float;
    void reset();

private:
    float kp_, ki_, kd_;
    float pwm_min_, pwm_max_;
    float prev_error_, integral_;
};


// SpeedPIDController.cpp**
#include "SpeedPIDController.hpp"
#include <algorithm>

SpeedPIDController::SpeedPIDController(float kp, float ki, float kd, float pwm_min, float pwm_max)
    : kp_(kp), ki_(ki), kd_(kd), pwm_min_(pwm_min), pwm_max_(pwm_max),
      prev_error_(0.0F), integral_(0.0F) {}

void SpeedPIDController::reset() {
    prev_error_ = 0.0F;
    integral_ = 0.0F;
}

// execute this function in a loop
// to update the speed
// v_current: current speed
// v_target: target speed
// dt: time step
// returns the PWM value to be sent to the motor controller
auto SpeedPIDController::update(float v_current, float v_target, float dt) -> float {
    float error = v_target - v_current;
    integral_ += error * dt;
    float derivative = (error - prev_error_) / dt;
    prev_error_ = error;

    float output = kp_ * error + ki_ * integral_ + kd_ * derivative;
    return std::clamp(output, pwm_min_, pwm_max_);
}

// SpeedPIDTuner.hpp**
#pragma once

#include <tuple>

auto autoTunePid(float dt = 0.1F, float sim_time = 10.0F, float v_target = 2.0F) -> std::tuple<float, float, float>;

// SpeedPIDTuner.cpp**
#include "SpeedPIDTuner.hpp"
#include <cmath>
#include <limits>

static auto simulateVelocity(float v_current, float pwm_input, float dt) -> float {
    float damping = 0.1F;
    float max_accel = 3.0F;
    float accel = pwm_input * max_accel / 100.0F - damping * v_current;
    return v_current + accel * dt;
}

static auto evaluatePid(float kp, float ki, float kd, float dt, float sim_time, float v_target) -> float {
    SpeedPIDController pid(kp, ki, kd, 0.0F, 100.0F);
    float v = 0.0F;
    float total_error = 0.0F;
    float max_overshoot = 0.0F;
    float final_error = 0.0F;

    for (float t = 0.0F; t <= sim_time; t += dt) {
        float pwm = pid.update(v, v_target, dt);
        v = simulateVelocity(v, pwm, dt);
        float error = std::abs(v_target - v);
        total_error += error * dt;

        if (v > v_target) {
            max_overshoot = std::max(max_overshoot, v - v_target);
        }

        if (t >= sim_time - 1.0F) {
            final_error += error * dt;
        }
    }

    return total_error + 10.0F * max_overshoot + 20.0F * final_error;
}

auto autoTunePid(float dt, float sim_time, float v_target) -> std::tuple<float, float, float> {
    float best_score = std::numeric_limits<float>::max();
    float best_kp = 0;
    float best_ki = 0;
    float best_kd = 0;

    for (float kp = 0.1F; kp <= 1.0F; kp += 0.1F) {
        for (float ki = 0.0F; ki <= 0.2F; ki += 0.02F) {
            for (float kd = 0.0F; kd <= 0.2F; kd += 0.02F) {
                float score = evaluatePid(kp, ki, kd, dt, sim_time, v_target);
                if (score < best_score) {
                    best_score = score;
                    best_kp = kp;
                    best_ki = ki;
                    best_kd = kd;
                }
            }
        }
    }

    return {best_kp, best_ki, best_kd};
}
