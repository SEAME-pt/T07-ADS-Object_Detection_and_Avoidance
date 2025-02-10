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
    uint32_t id;  // Variável para armazenar o ID da mensagem CAN
    std::vector<uint8_t> data;  // Vetor para armazenar os dados da mensagem CAN

    // Verificar se uma mensagem foi recebida no barramento CAN
    if (canBus.receiveMessage(id, data)) {
        // Ignorar mensagens com IDs inesperados
        if (id == canId) {
            // Certificar-se de que há dados suficientes
            if (data.size() >= 2) {
                int sensorValue = data[0] | (data[1] << 8);

                std::cout << "inside sensor class: " << _lastSpeed << std::endl;
                _lastSpeed = sensorValue;  // Atualizar o valor de velocidade
                return 0;  // Indicar sucesso
            } else {
                // Mensagem recebida com dados insuficientes
                return -1;  // Indicar falha
            }
        } else {
            // Ignorar mensagens com IDs inesperados (sem imprimir, apenas retornar falha)
            return -1;  // Indicar falha
        }
    } else {
        // Nenhuma mensagem disponível no barramento CAN (sem imprimir)
        return -1;  // Indicar falha
    }
}

const float SpeedSensor::getValue() const {
    return _lastSpeed;  // Retornar o último valor de velocidade
}

const std::string SpeedSensor::getType() const {
    return "SpeedSensor";  // Retornar o tipo do sensor
}
