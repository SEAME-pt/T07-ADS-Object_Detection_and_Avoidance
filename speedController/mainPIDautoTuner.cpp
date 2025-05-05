# include <iostream>
#include "SpeedPIDController.hpp"
#include "SpeedPIDTuner.hpp"

const float PWM_MIN = 0.0f; // Minimum PWM value
const float PWM_MAX = 100.0f; // Maximum PWM value

int main() {
	float dt = 0.1f; // Time step
	float sim_time = 10.0f; // Simulation time
	float v_target = 2.0f; // Target speed
	float v_current = 0.0f; // Initial speed
	float pwm_input = PWM_MIN; // Initial PWM input

	auto [kp, ki, kd] = auto_tune_pid(dt, sim_time, v_target);;

    std::cout << "Best PID gains found:\n";
    std::cout << "Kp = " << kp << "\n";
    std::cout << "Ki = " << ki << "\n";
    std::cout << "Kd = " << kd << "\n";
	std::cout << "Simulating with tuned PID controller...\n";
	SpeedPIDController traction(kp, ki, kd, PWM_MIN, PWM_MAX);
	traction.reset();
	v_current = 0.0f;//go fetch current speed from ZMQ
	
    return 0;
}
