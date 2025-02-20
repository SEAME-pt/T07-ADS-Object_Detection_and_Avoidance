#ifndef JETCAR_HPP
#define JETCAR_HPP

#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>    // Para open e O_RDWR
#include <unistd.h>   // Para close
#include <sys/ioctl.h>   // Para ioctl
#include <linux/i2c-dev.h>  // Para I2C_SLAVE
#include <cmath>      // Para std::floor

class JetCar {
public:
    JetCar(int motorAddr, int servoAddr);
    ~JetCar();

    void set_servo_angle(int angle);  // Função para ajustar o ângulo do servo
    void set_motor_speed(int speed);  // Função para controlar a velocidade do motor

private:
    void open_motor_i2c_bus();
    void open_servo_i2c_bus();
    bool init_motors();
    bool init_servo();
    void writeByteData(int fd, uint8_t reg, uint8_t value);
    uint8_t readByteData(int fd, uint8_t reg);
    bool setMotorPwm(const int channel, int value);

    int _fdMotor;
    int _fdServo;
    int _motorAddr;
    int _servoAddr;

    int _maxAngle;
    int _servoLeftPwm;
    int _servoRightPwm;
    int _servoCenterPwm;
    int _steeringChannel;
    int _currentAngle;
};

#endif  // JETCAR_HPP
