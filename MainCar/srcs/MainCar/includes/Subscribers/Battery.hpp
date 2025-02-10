#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <cstdint>
#include <stdexcept>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <fcntl.h>

#define I2C_ADDRESS 0x41
#define CONFIG_REGISTER 0x00
#define SHUNT_VOLTAGE_REGISTER 0x01
#define BUS_VOLTAGE_REGISTER 0x02
#define CALIBRATION_REGISTER 0x05

#define VOLTAGE_MAX 4.2  // Tensão máxima da bateria (em Volts)
#define VOLTAGE_MIN 3.0  // Tensão mínima da bateria (em Volts)
#define LSB_BUS_VOLTAGE 0.004  // 1 LSB = 4 mV
#define LSB_SHUNT_VOLTAGE 0.00001  // 1 LSB = 10 µV

class INA219 {
private:
    int file;
    const char *i2c_device;

    uint16_t readRegister(uint8_t registerAddress) {
        uint8_t buffer[2];

        // Escreve o endereço do registrador
        if (write(file, &registerAddress, 1) != 1) {
            throw std::runtime_error("Erro ao escrever no barramento I2C");
        }

        // Lê os dois bytes do registrador
        if (read(file, buffer, 2) != 2) {
            throw std::runtime_error("Erro ao ler do barramento I2C");
        }

        // Combina os bytes em um valor de 16 bits
        return (buffer[0] << 8) | buffer[1];
    }

    float calculateBatteryPercentage(float voltage) {
        if (voltage > VOLTAGE_MAX) {
            return 100.0f;
        } else if (voltage < VOLTAGE_MIN) {
            return 0.0f;
        } else {
            return ((voltage - VOLTAGE_MIN) / (VOLTAGE_MAX - VOLTAGE_MIN)) * 100.0f;
        }
    }

public:
    INA219(const char *device) : i2c_device(device) {
        file = open(i2c_device, O_RDWR);
        if (file < 0) {
            throw std::runtime_error("Erro ao abrir o dispositivo I2C");
        }

        // Configura o endereço do dispositivo I2C
        if (ioctl(file, I2C_SLAVE, I2C_ADDRESS) < 0) {
            throw std::runtime_error("Erro ao configurar o endereço do dispositivo I2C");
        }
    }

    ~INA219() {
        close(file);
    }

    float getBatteryPercentage() {
        // Lê o registrador de tensão do barramento
        uint16_t rawBusVoltage = readRegister(BUS_VOLTAGE_REGISTER) >> 3;  // Desloca 3 bits para a direita
        float busVoltage = rawBusVoltage * LSB_BUS_VOLTAGE;  // Converte para volts

        // Calcula o percentual da bateria
        return calculateBatteryPercentage(busVoltage);
    }
};
