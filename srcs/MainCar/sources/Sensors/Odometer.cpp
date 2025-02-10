#include "Odometer.hpp"
#include <iostream>  // Remover ou condicionar as impressões de logs

Odometer::Odometer(CANBus& can, uint32_t id) : ISensor(can, id), _lastDistance(0) {
    initialize();
}

void Odometer::initialize() {
    // Inicialização do sensor
    // Você pode optar por remover este log para produção
    std::cout << "Initializing odometer sensor..." << std::endl;
}

int Odometer::readData() {
    uint32_t id;  // Variável para armazenar o ID da mensagem CAN
    std::vector<uint8_t> data;  // Vetor para armazenar os dados da mensagem CAN

    // Verificar se uma mensagem foi recebida no barramento CAN
    if (canBus.receiveMessage(id, data)) {
        // Ignorar mensagens com IDs inesperados
        if (id == canId) {
            // Certificar-se de que há dados suficientes
            if (data.size() >= 2) {
                int sensorValue = data[0] | (data[1] << 8);

                std::cout << "inside sensor class: " << _lastDistance << std::endl;
                _lastDistance = sensorValue;  // Atualizar o valor de velocidade
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

const float Odometer::getValue() const {
    return _lastDistance;  // Retornar o último valor de velocidade
}

const std::string Odometer::getType() const {
    return "Odometer";  // Retornar o tipo do sensor
}
