# include <iostream>
#include "SpeedPIDController.hpp"
#include "SpeedPIDTuner.hpp"

int main() {
	SpeedPIDController pid(0.5f, 0.1f, 0.05f, 0.0f, 100.0f);
	float dt = 0.1f; // Time step
	float sim_time = 10.0f; // Simulation time
	float v_target = 2.0f; // Target speed
	float v_current = 0.0f; // Initial speed
	float pwm_input = 0.0f; // Initial PWM input
	float total_error = 0.0f;
	float max_overshoot = 0.0f;
	float final_error = 0.0f;

    auto [kp, ki, kd] = auto_tune_pid();

    std::cout << "Best PID gains found:\n";
    std::cout << "Kp = " << kp << "\n";
    std::cout << "Ki = " << ki << "\n";
    std::cout << "Kd = " << kd << "\n";

    return 0;
}