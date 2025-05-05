#include <iostream>
#include <cmath>
#include <limits>
#include <tuple>

class SpeedPIDController {
private:
	float kp_, ki_, kd_;
	float pwm_min_, pwm_max_;
	float prev_error_, integral_;
public:
	// Default constructor
	// Initializes PID parameters to zero and PWM limits to default values
	// This constructor is useful for creating an instance without any specific parameters
	// and allows for later initialization or tuning of the PID controller.
	// It is also useful for creating a default instance of the controller in cases where
	// specific tuning is not required or when the parameters will be set later.
    SpeedPIDController();
	// Constructor to initialize PID parameters and PWM limits
	SpeedPIDController(float kp, float ki, float kd, float pwm_min, float pwm_max);
	// Copy constructor
	SpeedPIDController(const SpeedPIDController& other){};
	// Move constructor
	SpeedPIDController(SpeedPIDController&& other) noexcept;
	// Destructor
	~SpeedPIDController();
	// Assignment operator
	SpeedPIDController& operator=(const SpeedPIDController& other);

	// Reset PID controller state
    void reset();
	// Update the PID controller with current speed and target speed
    float update(float v_current, float v_target, float dt);
	float simulate_velocity(float v_current, float pwm_input, float dt);
	std::tuple<float, float, float>  auto_tune_pid(float dt = 0.1f, float sim_time = 10.0f, float v_target = 2.0f);
	float evaluate_pid(float kp, float ki, float kd, float dt, float sim_time, float v_target);
	float simulate_velocity(float v_current, float pwm_input, float dt);
};
