#ifndef JETCAR_HPP
#define JETCAR_HPP

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <gpiod.h>

class JetCar {
public:
    JetCar();
    ~JetCar();

    void setMotorSpeed(int speed);
    void setServoAngle(int angle);
    void sequence();

private:
    void setPWM(int channel, int value); // Ajustado para a nova versão

    static constexpr int MAX_ANGLE = 45;
    
    static constexpr int SERVO_LEFT_PWM = 205;
    static constexpr int SERVO_CENTER_PWM = 307;
    static constexpr int SERVO_RIGHT_PWM = 410;

    static constexpr int STEERING_CHANNEL = 0; // Canal do servo

    gpiod_chip *chip;
    gpiod_line *line_pwm, *line_in1, *line_in2, *line_in3, *line_in4;
};

#endif
