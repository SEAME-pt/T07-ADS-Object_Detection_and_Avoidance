// SpeedPIDTuner.cpp**
#include "SpeedPIDTuner.hpp"
#include "SpeedPIDController.hpp"
#include "DebugTools.hpp"
#include <cmath>
#include <limits>


float speed_from_zermq = 0.0f; // Placeholder for the speed from ZMQ service

static float real_velocity(float v_current, float pwm_input, float dt) {
	// Fetch the real speed from the vehicle's sensors
	// This is a placeholder function and should be replaced with actual sensor reading
	// For example, you can use a function like get_speed_from_sensor() to get the real speed
	// This function should return the current speed of the vehicle
	// For example, if you have a speed sensor, you can read the speed from it
	// and return it as a float value
	// In this example, we will just return the current speed from the zero mq service
    SpeedSubscriber subscriber;
    subscriber.start([&](float speed) {
        speed_from_zermq = speed;
    });
	return (speed_from_zermq);
}

static float simulate_velocity(float v_current, float pwm_input, float dt) {
	// Simulate the velocity based on PWM input and current velocity
	// This is a simple model and can be adjusted for more accuracy
	// Damping factor and max acceleration can be tuned
	// For example, damping = 0.1f, max_accel = 3.0f
	// These values can be adjusted based on the vehicle's characteristics
	// and the desired simulation accuracy
	// The damping factor simulates the resistance to acceleration
	// The max_accel simulates the maximum acceleration based on PWM input
	// The formula used is a simple Euler integration step
	// where the new velocity is calculated based on the current velocity,
	// the acceleration (based on PWM input), and the time step (dt)
	// The acceleration is calculated as:
	// accel = pwm_input * max_accel / 100.0f - damping * v_current
	// This means that the acceleration is proportional to the PWM input
	// and inversely proportional to the current velocity, simulating a damping effect
	float damping = 0.1f;
    float max_accel = 3.0f;
    float accel = pwm_input * max_accel / 100.0f - damping * v_current;
    return v_current + accel * dt;
}

static float get_velocity(float v_current, float pwm_input, float dt, bool real = true) {
    return (real ? real_velocity(v_current, pwm_input, dt) : simulate_velocity(v_current, pwm_input, dt));
}

static float evaluate_pid(float kp, float ki, float kd, float dt, float sim_time, float v_target, bool real = true) {
    SpeedPIDController pid(kp, ki, kd, 0.0f, 100.0f);
    float v = 0.0f;
    float total_error = 0.0f;
    float max_overshoot = 0.0f;
    float final_error = 0.0f;

    for (float t = 0.0f; t <= sim_time; t += dt) {
        float pwm = pid.update(v, v_target, dt);
		v = get_velocity(v, pwm, dt, real);
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
                float score = evaluate_pid(kp, ki, kd, dt, sim_time, v_target, true);
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