#ifndef CONTROLLER_SUBSCRIBER_HPP
#define CONTROLLER_SUBSCRIBER_HPP

#include <zmq.hpp>
#include <string>
#include <unordered_map>
#include <iostream>
#include <mutex>

class ControllerSubscriber {
public:
    explicit ControllerSubscriber(const std::string& address)
        : context(1), subscriber(context, zmq::socket_type::sub) {
        subscriber.connect(address);
        subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0); // Usando setsockopt de maneira correta
    }

    void startListening() {
        std::thread listener([this]() {
            while (true) {
                zmq::message_t message;
                
                // Verificando o retorno de recv
                zmq::recv_result_t result = subscriber.recv(message, zmq::recv_flags::none);
                [[maybe_unused]] bool success = result.has_value(); // Verifica se o valor foi recebido corretamente

                if (!success) {
                    std::cerr << "Erro ao receber mensagem." << std::endl;
                    continue;
                }

                std::string receivedMessage(static_cast<char*>(message.data()), message.size());

                // Atualiza o estado dos valores do controller
                processControllerMessage(receivedMessage);
            }
        });

        listener.detach();
    }

    // Obtém o estado de um controle específico (ex: "horn")
    std::string getControlState(const std::string& key) {
        std::lock_guard<std::mutex> lock(dataMutex);
        return controllerData[key];
    }

private:
    zmq::context_t context;
    zmq::socket_t subscriber;
    std::unordered_map<std::string, std::string> controllerData;
    std::mutex dataMutex; // Para evitar condições de corrida

    void processControllerMessage(const std::string& message) {
        auto delimiterPos = message.find(' ');
        if (delimiterPos != std::string::npos) {
            std::string key = message.substr(0, delimiterPos);
            std::string value = message.substr(delimiterPos + 1);

            {
                // thread-safe
                std::lock_guard<std::mutex> lock(dataMutex);
                controllerData[key] = value;
            }

            std::cout << "Atualizado: " << key << " = " << value << std::endl;
        } else {
            std::cerr << "Formato de mensagem inválido: " << message << std::endl;
        }
    }
};

#endif // CONTROLLER_SUBSCRIBER_HPP
