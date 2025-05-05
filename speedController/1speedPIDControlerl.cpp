#include <iostream>
#include <cmath>
#include <limits>
#include <tuple>

#include "speedPIDController.hpp"
// Default constructor
// Initializes PID parameters to zero and PWM limits to default values
// This constructor is useful for creating an instance without any specific parameters
// and allows for later initialization or tuning of the PID controller.
// It is also useful for creating a default instance of the controller in cases where
// specific tuning is not required or when the parameters will be set later.
SpeedPIDController::SpeedPIDController(){}
// Constructor to initialize PID parameters and PWM limits
SpeedPIDController::SpeedPIDController(float kp, float ki, float kd, float pwm_min, float pwm_max)
	: kp_(kp), ki_(ki), kd_(kd), pwm_min_(pwm_min), pwm_max_(pwm_max),
		prev_error_(0.0f), integral_(0.0f) {}
// Copy constructor
SpeedPIDController::SpeedPIDController(const SpeedPIDController& other)
	: kp_(other.kp_), ki_(other.ki_), kd_(other.kd_),
		pwm_min_(other.pwm_min_), pwm_max_(other.pwm_max_),
		prev_error_(other.prev_error_), integral_(other.integral_) {}
// Move constructor
SpeedPIDController::SpeedPIDController(SpeedPIDController&& other) noexcept
	: kp_(other.kp_), ki_(other.ki_), kd_(other.kd_),
		pwm_min_(other.pwm_min_), pwm_max_(other.pwm_max_),
		prev_error_(other.prev_error_), integral_(other.integral_) {
	// Reset the moved-from object
	other.kp_ = 0;
	other.ki_ = 0;
	other.kd_ = 0;
	other.pwm_min_ = 0;
	other.pwm_max_ = 0;
	other.prev_error_ = 0;
	other.integral_ = 0;
}
// Destructor
SpeedPIDController::~SpeedPIDController() {
	// Cleanup if needed
}
// Assignment operator
SpeedPIDController& SpeedPIDController::operator=(const SpeedPIDController& other) {
	if (this != &other) {
		this->kp_ = other.kp_;
		this->ki_ = other.ki_;
		this->kd_ = other.kd_;
		this->pwm_min_ = other.pwm_min_;
		this->pwm_max_ = other.pwm_max_;
		this->prev_error_ = other.prev_error_;
		this->integral_ = other.integral_;
	}
	return *this;
}
// Reset PID controller state
void SpeedPIDController::reset() {
	prev_error_ = 0.0f;
	integral_ = 0.0f;
}
// Update the PID controller with current speed and target speed
float SpeedPIDController::update(float v_current, float v_target, float dt) {
	float error = v_target - v_current;
	integral_ += error * dt;
	float derivative = (error - prev_error_) / dt;
	prev_error_ = error;

	float output = kp_ * error + ki_ * integral_ + kd_ * derivative;
	return std::clamp(output, pwm_min_, pwm_max_);
}
// Simple model: PWM controls acceleration with damping
// Simulate the velocity based on current speed, PWM input, and time step
// This function models the vehicle's acceleration and damping effect
// It takes the current speed, PWM input (as a percentage), and time step
// and returns the new speed after applying the PWM input and damping
float SpeedPIDController::simulate_velocity(float v_current, float pwm_input, float dt) {
	float damping = 0.1f;
	float max_accel = 3.0f;
	float accel = pwm_input * max_accel / 100.0f - damping * v_current;
	return v_current + accel * dt;
}

std::tuple<float, float, float>  SpeedPIDController::auto_tune_pid(float dt = 0.1f, float sim_time = 10.0f, float v_target = 2.0f) {
	float best_score = std::numeric_limits<float>::max();

	for (float kp = 0.1f; kp <= 1.0f; kp += 0.1f) {
		for (float ki = 0.0f; ki <= 0.2f; ki += 0.02f) {
			for (float kd = 0.0f; kd <= 0.2f; kd += 0.02f) {
				float score = evaluate_pid(kp, ki, kd, dt, sim_time, v_target);
				if (score < best_score) {
					best_score = score;
					this->kp_ = kp;
					this->ki_ = ki;
					this->kd_ = kd;
				}
			}
		}
	}
	std::cout << "Best PID parameters: Kp = " << this->kp_ << ", Ki = " << this->ki_ << ", Kd = " << this->kd_ << std::endl;
	return {this->kp_, this->ki_, this->kd_};
	// Penalize overshoot and final error
}

// Score a set of PID gains by simulating response
float SpeedPIDController::evaluate_pid(float kp, float ki, float kd, float dt, float sim_time, float v_target) {
    SpeedPIDController test(kp, ki, kd, 0.0f, 100.0f);
    float v = 0.0f;
    float total_error = 0.0f;
    float max_overshoot = 0.0f;
    float final_error = 0.0f;

    for (float t = 0.0f; t <= sim_time; t += dt) {
        float pwm = this->update(v, v_target, dt);
        v = simulate_velocity(v, pwm, dt);// replace with read_velocity function
        float error = std::abs(v_target - v);
        total_error += error * dt;

        if (v > v_target) {
            max_overshoot = std::max(max_overshoot, v - v_target);
        }

        if (t >= sim_time - 1.0f) { // check final second
            final_error += error * dt;
        }
    }

    return total_error + 10.0f * max_overshoot + 20.0f * final_error;
}


