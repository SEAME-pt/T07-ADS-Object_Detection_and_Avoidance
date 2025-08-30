// SpeedPIDTuner.cpp**
#include "SpeedPIDTuner.hpp"
#include "SpeedPIDController.hpp"
#include <cmath>
#include <limits>


float speed_from_zermq = 0.0F; // Placeholder for the speed from ZMQ service

static auto realVelocity(float pwm_input, float dt) -> float {
	// Fetch the real speed from the vehicle's sensors
	// This is a placeholder function and should be replaced with actual sensor reading
	// For example, you can use a function like get_speed_from_sensor() to get the real speed
	// This function should return the current speed of the vehicle
	// For example, if you have a speed sensor, you can read the speed from it
	// and return it as a float value
	// In this example, we will just return the current speed from the zero mq service
	return (speed_from_zermq);
}

static auto simulateVelocity(float v_current, float pwm_input, float dt) -> float {
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
	float damping = 0.1F;
    float max_accel = 3.0F;
    float accel = pwm_input * max_accel / 100.0F - damping * v_current;
    return v_current + accel * dt;
}

static auto getVelocity(float v_current, float pwm_input, float dt, bool real = true) -> float {
    return (real ? realVelocity(pwm_input, dt) : simulateVelocity(v_current, pwm_input, dt));
}

static auto evaluatePid(float kp, float ki, float kd, float dt, float sim_time, float v_target, bool real = true) -> float {
    SpeedPIDController pid(kp, ki, kd, 0.0f, 100.0f);
    float v = 0.0F;
    float total_error = 0.0F;
    float max_overshoot = 0.0F;
    float final_error = 0.0F;

    for (float t = 0.0F; t <= sim_time; t += dt) {
        float pwm = pid.update(v, v_target, dt);
		v = getVelocity(v, pwm, dt, real);
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

auto autoTunePid(float dt, float sim_time, float v_target, bool real = true) -> std::tuple<float, float, float> {
    float best_score = std::numeric_limits<float>::max();
    float best_kp = 0;
    float best_ki = 0;
    float best_kd = 0;

    for (float kp = 0.1F; kp <= 1.0F; kp += 0.1F) {
        for (float ki = 0.0F; ki <= 0.2F; ki += 0.02F) {
            for (float kd = 0.0F; kd <= 0.2F; kd += 0.02F) {
                float score = evaluatePid(kp, ki, kd, dt, sim_time, v_target, real);
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
