#include "JetCar.hpp"
#include <unistd.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <gpiod.h>

#define PWM_CHIP "/sys/class/pwm/pwmchip0"  // Ajuste conforme necessário
#define PWM_PERIOD "20000000"  // 20ms (frequência de 50Hz)

#define CHIP_NAME "/dev/gpiochip0"

gpiod_line *line_in1, *line_in2, *line_in3, *line_in4;

void writeToFile(const std::string &path, const std::string &value) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Erro ao acessar " + path);
    }
    file << value;
}

void JetCar::setPWM(int channel, int value) {
    std::string pwm_path = std::string(PWM_CHIP) + "/pwm0/";
    writeToFile(pwm_path + "duty_cycle", std::to_string(value));
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

    if (speed > 0) {
        gpiod_line_set_value(line_in1, 1);
        gpiod_line_set_value(line_in2, 0);
        gpiod_line_set_value(line_in3, 1);
        gpiod_line_set_value(line_in4, 0);
    } else if (speed < 0) {
        gpiod_line_set_value(line_in1, 0);
        gpiod_line_set_value(line_in2, 1);
        gpiod_line_set_value(line_in3, 0);
        gpiod_line_set_value(line_in4, 1);
    } else {
        gpiod_line_set_value(line_in1, 0);
        gpiod_line_set_value(line_in2, 0);
        gpiod_line_set_value(line_in3, 0);
        gpiod_line_set_value(line_in4, 0);
    }
}

JetCar::JetCar() {
    gpiod_chip *chip = gpiod_chip_open(CHIP_NAME);
    if (!chip) {
        throw std::runtime_error("Erro ao abrir GPIO chip");
    }

    line_in1 = gpiod_chip_get_line(chip, 23);
    line_in2 = gpiod_chip_get_line(chip, 24);
    line_in3 = gpiod_chip_get_line(chip, 25);
    line_in4 = gpiod_chip_get_line(chip, 12);

    gpiod_line_request_output(line_in1, "JetCar", 0);
    gpiod_line_request_output(line_in2, "JetCar", 0);
    gpiod_line_request_output(line_in3, "JetCar", 0);
    gpiod_line_request_output(line_in4, "JetCar", 0);

    // Configuração do PWM
    writeToFile(std::string(PWM_CHIP) + "/export", "0");  // Ativar PWM0
    writeToFile(std::string(PWM_CHIP) + "/pwm0/period", PWM_PERIOD);
    writeToFile(std::string(PWM_CHIP) + "/pwm0/enable", "1");
}

JetCar::~JetCar() {
    writeToFile(std::string(PWM_CHIP) + "/pwm0/enable", "0");  // Desativar PWM
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
