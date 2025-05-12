#include "JetCar.hpp"

// Implementação da classe JetCar

JetCar::JetCar(int motorAddr, int servoAddr)
    : _motorAddr(motorAddr), _servoAddr(servoAddr), _fdMotor(-1), _fdServo(-1),
      _maxAngle(30), _servoLeftPwm(170), _servoRightPwm(430), _servoCenterPwm(300),
      _steeringChannel(0), _currentAngle(0) {

    // Inicializar servo e motores
    open_servo_i2c_bus();
    if (!init_servo()) {
        throw std::runtime_error("Erro ao inicializar o servo.");
    }

    open_motor_i2c_bus();
    if (!init_motors()) {
        throw std::runtime_error("Erro ao inicializar os motores.");
    }

    std::cout << "JetCar inicializado com sucesso!" << std::endl;
}

JetCar::~JetCar() {
    set_motor_speed(0);
    close(_fdMotor);
    close(_fdServo);
    std::cout << "Destruindo o JetCar..." << std::endl;
}

void JetCar::set_servo_angle(int angle) {
    angle = std::max(-_maxAngle, std::min(_maxAngle, angle));

    int pwm;
    if (angle < 0) {
        pwm = static_cast<int>(_servoCenterPwm + (static_cast<float>(angle) / _maxAngle) * (_servoCenterPwm - _servoLeftPwm));
    } else if (angle > 0) {
        pwm = static_cast<int>(_servoCenterPwm + (static_cast<float>(angle) / _maxAngle) * (_servoRightPwm - _servoCenterPwm));
    } else {
        pwm = _servoCenterPwm;
    }

    writeByteData(_fdServo, 0x06 + 4 * _steeringChannel, 0);
    writeByteData(_fdServo, 0x07 + 4 * _steeringChannel, 0);
    writeByteData(_fdServo, 0x08 + 4 * _steeringChannel, pwm & 0xFF);
    writeByteData(_fdServo, 0x09 + 4 * _steeringChannel, pwm >> 8);
    _currentAngle = angle;
}

void JetCar::set_motor_speed(int speed) {
    int pwmValue;
    speed = std::max(-100, std::min(100, speed));
    pwmValue = static_cast<int>(std::abs(speed) / 100.0 * 4095);
    std::cout << "Motor speed: " << speed << ", PWM value: " << pwmValue << std::endl;

    if (speed > 0) {
        // Movendo para frente
        setMotorPwm(0, pwmValue);  // IN1
        setMotorPwm(1, 0);         // IN2
        setMotorPwm(2, pwmValue);  // ENA

        setMotorPwm(5, pwmValue);  // IN3
        setMotorPwm(6, 0);         // IN4
        setMotorPwm(7, pwmValue);  // ENB
    } else if (speed < 0) {
        // Movendo para trás
        setMotorPwm(0, pwmValue);  // IN1
        setMotorPwm(1, pwmValue);  // IN2
        setMotorPwm(2, 0);         // ENA

        setMotorPwm(5, 0);         // IN3
        setMotorPwm(6, pwmValue);  // IN4
        setMotorPwm(7, pwmValue);  // ENB
    } else {
        // Parando
        for (int channel = 0; channel < 9; ++channel) {
            setMotorPwm(channel, 0);
        }
    }
}

// Métodos privados para controle de I2C

void JetCar::open_motor_i2c_bus() {
    std::string i2cDevice = "/dev/i2c-1";
    _fdMotor = open(i2cDevice.c_str(), O_RDWR);
    if (_fdMotor < 0) throw std::runtime_error("Erro ao abrir o barramento I2C para o motor");

    if (ioctl(_fdMotor, I2C_SLAVE, _motorAddr) < 0) {
        close(_fdMotor);
        throw std::runtime_error("Erro ao configurar o endereço I2C do motor");
    }
}

void JetCar::open_servo_i2c_bus() {
    std::string i2cDevice = "/dev/i2c-1";
    _fdServo = open(i2cDevice.c_str(), O_RDWR);
    if (_fdServo < 0) throw std::runtime_error("Erro ao abrir o barramento I2C para o servo");

    if (ioctl(_fdServo, I2C_SLAVE, _servoAddr) < 0) {
        close(_fdServo);
        throw std::runtime_error("Erro ao configurar o endereço I2C do servo");
    }
}

bool JetCar::init_motors() {
    try {
        writeByteData(_fdMotor, 0x00, 0x20);
        usleep(1000);

        int preScale;
        uint8_t oldMode, newMode;

        oldMode = readByteData(_fdMotor, 0x00);
        preScale = static_cast<int>(std::floor(25000000.0 / 4096.0 / 60 - 1));
        newMode = (oldMode & 0x7F) | 0x10;

        writeByteData(_fdMotor, 0x00, newMode);
        writeByteData(_fdMotor, 0xFE, preScale);
        writeByteData(_fdMotor, 0x00, oldMode);
        
        usleep(5000);

        writeByteData(_fdMotor, 0x00, oldMode | 0xa1);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erro na inicialização dos motores: " << e.what() << std::endl;
        return false;
    }
}

bool JetCar::init_servo() {
    try {
        writeByteData(_fdServo, 0x00, 0x06);  // Reset do servo
        usleep(100000);

        writeByteData(_fdServo, 0x00, 0x10);  // Configura o controle
        usleep(100000);

        writeByteData(_fdServo, 0xFE, 0x79);  // Frequência do PWM
        usleep(100000);

        writeByteData(_fdServo, 0x01, 0x04);  // Configura o MODE2
        usleep(100000);

        writeByteData(_fdServo, 0x00, 0x20);  // Habilita o auto-incremento
        usleep(100000);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erro na inicialização do servo: " << e.what() << std::endl;
        return false;
    }
}

void JetCar::writeByteData(int fd, uint8_t reg, uint8_t value) {
    uint8_t buffer[2] = {reg, value};
    if (write(fd, buffer, 2) != 2) {
        throw std::runtime_error("Erro ao escrever no dispositivo I2C.");
    }
}

uint8_t JetCar::readByteData(int fd, uint8_t reg) {
    if (write(fd, &reg, 1) != 1)
        throw std::runtime_error("Erro ao enviar o registrador ao dispositivo I2C.");
    
    uint8_t value;
    if (read(fd, &value, 1) != 1)
        throw std::runtime_error("Erro ao ler o registrador ao dispositivo I2C.");
    
    return value;
}

bool JetCar::setMotorPwm(const int channel, int value) {
    value = std::max(0, std::min(4095, value));
    writeByteData(_fdMotor, 0x06 + 4 * channel, 0);
    writeByteData(_fdMotor, 0x07 + 4 * channel, 0);
    writeByteData(_fdMotor, 0x08 + 4 * channel, value & 0xFF);
    writeByteData(_fdMotor, 0x09 + 4 * channel, value >> 8);
    return true;
}
