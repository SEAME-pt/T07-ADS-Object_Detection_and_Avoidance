#include "SpeedSensor.hpp"
#include <iostream>  // Remover ou condicionar as impressões de logs

SpeedSensor::SpeedSensor(CANBus& can, uint32_t id) : ISensor(can, id), _lastSpeed(0) {
    initialize();
}

void SpeedSensor::initialize() {
    // Inicialização do sensor
    // Você pode optar por remover este log para produção
    std::cout << "Initializing speed sensor..." << std::endl;
}

int SpeedSensor::readData() {
    uint32_t id;
    std::vector<uint8_t> data;

    if (canBus.receiveMessage(id, data)) {
        if (id == canId) {
            if (data.size() >= 4) {  // Agora são 4 bytes
                union {
                    float f;
                    uint8_t b[4];
                } speedData;

                for (int i = 0; i < 4; ++i) {
                    speedData.b[i] = data[i];
                }

                _lastSpeed = speedData.f;
                return 0;
            } else {
                return -1;  // Dados insuficientes
            }
        } else {
            return -1;  // ID diferente
        }
    } else {
        return -1;  // Nenhuma mensagem
    }
}

const float SpeedSensor::getValue() const {
    return _lastSpeed;  // Retornar o último valor de velocidade
}

const std::string SpeedSensor::getType() const {
    return "SpeedSensor";  // Retornar o tipo do sensor
}
