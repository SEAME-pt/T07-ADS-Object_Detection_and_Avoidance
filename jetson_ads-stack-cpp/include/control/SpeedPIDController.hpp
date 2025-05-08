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