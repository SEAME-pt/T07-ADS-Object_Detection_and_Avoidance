#include "JetCar.hpp"
#include <unistd.h>
#include <cmath>         // For std::abs, std::floor, etc.
#include <stdexcept>     // For std::runtime_error
#include <algorithm>     // For std::max, std::min>
#include <gpiod.h>       // Biblioteca GPIOD para GPIO

#define CHIP_NAME "/dev/gpiochip0"  // Defina o nome correto do chip GPIO

// Ponteiros para as linhas GPIO
// gpiod_chip *chip;
gpiod_line *line_pwm, *line_in1, *line_in2, *line_in3, *line_in4;

void JetCar::setPWM(int channel, int value) {
    if (!line_pwm) return;
    gpiod_line_set_value(line_pwm, value);
}

void JetCar::setServoAngle(int angle) {
    angle = std::max(-MAX_ANGLE, std::min(45, angle));
    int pwm;
    if (angle < 0) {
        pwm = SERVO_CENTER_PWM + (angle / (float)MAX_ANGLE) * (SERVO_CENTER_PWM - SERVO_LEFT_PWM);
    } else if (angle > 0) {
        pwm = SERVO_CENTER_PWM + (angle / (float)MAX_ANGLE) * (SERVO_RIGHT_PWM - SERVO_CENTER_PWM);
    } else {
        pwm = SERVO_CENTER_PWM;
    }
    setPWM(STEERING_CHANNEL, pwm);
}

void JetCar::setMotorSpeed(int speed) {
    speed = std::max(-100, std::min(100, speed));

    if (speed > 0) { // Forward
        gpiod_line_set_value(line_in1, 1);
        gpiod_line_set_value(line_in2, 0);
        gpiod_line_set_value(line_in3, 1);
        gpiod_line_set_value(line_in4, 0);
    } else if (speed < 0) { // Backward
        gpiod_line_set_value(line_in1, 0);
        gpiod_line_set_value(line_in2, 1);
        gpiod_line_set_value(line_in3, 0);
        gpiod_line_set_value(line_in4, 1);
    } else { // Stop
        gpiod_line_set_value(line_in1, 0);
        gpiod_line_set_value(line_in2, 0);
        gpiod_line_set_value(line_in3, 0);
        gpiod_line_set_value(line_in4, 0);
    }
}

JetCar::JetCar() {
    chip = gpiod_chip_open(CHIP_NAME);
    if (!chip) {
        throw std::runtime_error("Failed to open GPIO chip");
    }

    line_pwm = gpiod_chip_get_line(chip, 18); // Exemplo: GPIO18 para PWM
    line_in1 = gpiod_chip_get_line(chip, 23);
    line_in2 = gpiod_chip_get_line(chip, 24);
    line_in3 = gpiod_chip_get_line(chip, 25);
    line_in4 = gpiod_chip_get_line(chip, 12);

    gpiod_line_request_output(line_pwm, "JetCar", 0);
    gpiod_line_request_output(line_in1, "JetCar", 0);
    gpiod_line_request_output(line_in2, "JetCar", 0);
    gpiod_line_request_output(line_in3, "JetCar", 0);
    gpiod_line_request_output(line_in4, "JetCar", 0);
}

JetCar::~JetCar() {
    gpiod_chip_close(chip);
}

void JetCar::sequence() {
    setMotorSpeed(100);
    sleep(2);

    setMotorSpeed(-100);
    sleep(2);

    setMotorSpeed(0);

    setServoAngle(-45);
    sleep(1);

    setServoAngle(45);
    sleep(1);

    setServoAngle(0);
    setMotorSpeed(0);
}
