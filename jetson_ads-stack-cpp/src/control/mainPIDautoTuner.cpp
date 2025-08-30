# include <iostream>
#include "SpeedPIDController.hpp"
#include "SpeedPIDTuner.hpp"

const float pwm_min = 0.0F; // Minimum PWM value
const float pwm_max = 100.0F; // Maximum PWM value

auto main() -> int {
	float dt = 0.03F;//Time step
	float sim_time = 10.0F; // Simulation time
	float v_target = 2.0F; // Target speed
	float v_current = 0.0F; // Initial speed
	float pwm_input = pwm_min; // Initial PWM input

	auto [kp, ki, kd] = auto_tune_pid(dt, sim_time, v_target);;

    std::cout << "Best PID gains found:\n";
    std::cout << "Kp = " << kp << "\n";
    std::cout << "Ki = " << ki << "\n";
    std::cout << "Kd = " << kd << "\n";
	std::cout << "Simulating with tuned PID controller...\n";
	SpeedPIDController traction(kp, ki, kd, PWM_MIN, PWM_MAX);
	traction.reset();
	v_current = 0.0F;//go fetch current speed from ZMQ
	v_target =0.0F;//test
	traction.update(v_current, v_target, dt);
	std::cout << "Current speed: " << v_current << "\n";
    return 0;
}
