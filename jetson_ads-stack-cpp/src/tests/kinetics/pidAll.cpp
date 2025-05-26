// SpeedPIDController.hpp**
#pragma once

class SpeedPIDController {
public:
    SpeedPIDController(float kp, float ki, float kd, float pwm_min, float pwm_max);

    float update(float v_current, float v_target, float dt);
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
      prev_error_(0.0f), integral_(0.0f) {}

void SpeedPIDController::reset() {
    prev_error_ = 0.0f;
    integral_ = 0.0f;
}

// execute this function in a loop
// to update the speed
// v_current: current speed
// v_target: target speed
// dt: time step
// returns the PWM value to be sent to the motor controller
float SpeedPIDController::update(float v_current, float v_target, float dt) {
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

std::tuple<float, float, float> auto_tune_pid(float dt = 0.1f, float sim_time = 10.0f, float v_target = 2.0f);

// SpeedPIDTuner.cpp**
#include "SpeedPIDTuner.hpp"
#include "SpeedPIDController.hpp"
#include <cmath>
#include <limits>

static float simulate_velocity(float v_current, float pwm_input, float dt) {
    float damping = 0.1f;
    float max_accel = 3.0f;
    float accel = pwm_input * max_accel / 100.0f - damping * v_current;
    return v_current + accel * dt;
}

static float evaluate_pid(float kp, float ki, float kd, float dt, float sim_time, float v_target) {
    SpeedPIDController pid(kp, ki, kd, 0.0f, 100.0f);
    float v = 0.0f;
    float total_error = 0.0f;
    float max_overshoot = 0.0f;
    float final_error = 0.0f;

    for (float t = 0.0f; t <= sim_time; t += dt) {
        float pwm = pid.update(v, v_target, dt);
        v = simulate_velocity(v, pwm, dt);
        float error = std::abs(v_target - v);
        total_error += error * dt;

        if (v > v_target) {
            max_overshoot = std::max(max_overshoot, v - v_target);
        }

        if (t >= sim_time - 1.0f) {
            final_error += error * dt;
        }
    }

    return total_error + 10.0f * max_overshoot + 20.0f * final_error;
}

std::tuple<float, float, float> auto_tune_pid(float dt, float sim_time, float v_target) {
    float best_score = std::numeric_limits<float>::max();
    float best_kp = 0, best_ki = 0, best_kd = 0;

    for (float kp = 0.1f; kp <= 1.0f; kp += 0.1f) {
        for (float ki = 0.0f; ki <= 0.2f; ki += 0.02f) {
            for (float kd = 0.0f; kd <= 0.2f; kd += 0.02f) {
                float score = evaluate_pid(kp, ki, kd, dt, sim_time, v_target);
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
