#include "zmqreader.h"
#include <QDebug>
#include <cstring>
#include <iostream>
#include <zmq_addon.hpp> // Para zmq::poll

ZMQReader::ZMQReader(const QString &address, QObject *parent)
    : QThread(parent), m_address(address), m_running(false)
{
    m_context = std::make_unique<zmq::context_t>(1);
    m_socket = std::make_unique<zmq::socket_t>(*m_context, zmq::socket_type::sub);
        
    // Configurar linger para 0 (fechamento imediato)
    m_socket->set(zmq::sockopt::linger, 0);
}

ZMQReader::~ZMQReader()
{
    stop();
}


// void ZMQReader::run()
// {
//     try {
//         m_socket->connect(m_address.toStdString());
//         m_socket->set(zmq::sockopt::subscribe, "");
//         // m_socket->set(zmq::sockopt::linger, 0);  // Fecha socket imediatamente

//         QMap<QString, std::function<void(QString)>> handlers = {
//             {"speed", [this](QString v) { emit speedReceived(v); }},
//             {"battery", [this](QString v) { emit batteryReceived(v); }},  // Confirmar valor da bateria
//             {"lightshigh", [this](QString v) { emit headLightsReceived(v); }},
//             {"brake", [this](QString v) { emit brakeLightReceived(v); }},
//             {"lightsemergency", [this](QString v) { emit emergencyLightsReceived(v); }},
//             {"lightsleft", [this](QString v) { emit turnLightLeftReceived(v); }},
//             {"lightsright", [this](QString v) { emit turnLightRightReceived(v); }},
//             {"totaldistance", [this](QString v) { emit totalDistanceReceived(v); }},
//             {"lka", [this](QString v) { emit lkasReceived(v); }},
//             {"autopilot", [this](QString v) { emit autoPilotReceived(v); }},
//             {"lineleft", [this](QString v) { emit lineLeftReceived(v); }},
//             {"lineright", [this](QString v) { emit lineRightReceived(v); }},

//             {"horn", [this](QString v) { emit hornReceived(v); }},
//             {"lightslow", [this](QString v) { emit lightsLowReceived(v); }},
//             {"lightspark", [this](QString v) { emit lightSparkReceived(v); }},
//             {"ismoving", [this](QString v) { emit isMovingReceived(v); }},
//             {"batterypercentage", [this](QString v) { emit batteryPercentageReceived(v); }}  // Para corrigir bateria por percentagem
//         };

//         m_running = true;

//         while (m_running) {

//             if (!m_context) {
//                 qWarning() << "Contexto foi destruído!";
//                 break;
//             }

//             // Configura polling com timeout de 100ms
//             zmq::pollitem_t items[] = {{*m_socket, 0, ZMQ_POLLIN, 0}};
//             zmq::poll(items, 1, std::chrono::milliseconds(100));

//             if (items[0].revents & ZMQ_POLLIN) {
//                 zmq::message_t message;
//                 zmq::recv_result_t result = m_socket->recv(message, zmq::recv_flags::none);

//                 if (!result) {
//                     qWarning() << "Erro ao receber mensagem!";
//                     continue;
//                 }

//                 if (!m_socket || !m_context) {
//                     qWarning() << "Socket/Contexto inválido!";
//                     break;
//                 }

//                 // Processamento da mensagem
//                 std::string msg(static_cast<const char*>(message.data()), message.size());
//                 // std::cout << "Message: " << msg << std::endl;

//                 // auto parts = QString::fromStdString(msg).split(" ");
//                 // if (parts.size() == 2 && handlers.contains(parts[0])) {
//                 //     handlers[parts[0]](parts[1]);
//                 // }
                
//                 QString receivedValue = QString::fromStdString(msg);
//                 if (!receivedValue.isEmpty() && receivedValue.back() == QChar::Null) {
//                     receivedValue.chop(1); // remove 1  char do fim da string se for o \0
//                 }
//                 auto parts = receivedValue.split(" ", Qt::SkipEmptyParts);
//                 if (parts.size() == 2 && handlers.contains(parts[0])) {
//                     handlers[parts[0]](parts[1]);
//                 }

//             }
//         }

//         // Fecha recursos dentro da thread
//         if (m_socket) m_socket->close();
//         if (m_context) m_context->close();

//     } catch (const zmq::error_t &e) {
//         qWarning() << "Erro no loop:" << e.what();
//     }
// }


void ZMQReader::run() {
    try {
        qDebug() << "Conectando ao endereço:" << m_address;
        m_socket->connect(m_address.toStdString());
        m_socket->set(zmq::sockopt::subscribe, "");
        qDebug() << "Conectado e inscrito para todas as mensagens";

        QMap<QString, std::function<void(QString)>> handlers = {
            {"speed", [this](QString v) { emit speedReceived(v); }},
            {"battery", [this](QString v) { emit batteryReceived(v); }},  // Confirmar valor da bateria
            {"lightshigh", [this](QString v) { emit headLightsReceived(v); }},
            {"brake", [this](QString v) { emit brakeLightReceived(v); }},
            {"lightsemergency", [this](QString v) { emit emergencyLightsReceived(v); }},
            {"lightsleft", [this](QString v) { emit turnLightLeftReceived(v); }},
            {"lightsright", [this](QString v) { emit turnLightRightReceived(v); }},
            {"totaldistance", [this](QString v) { emit totalDistanceReceived(v); }},
            {"lka", [this](QString v) { emit lkasReceived(v); }},
            {"autopilot", [this](QString v) { emit autoPilotReceived(v); }},
            {"lineleft", [this](QString v) { emit lineLeftReceived(v); }},
            {"lineright", [this](QString v) { emit lineRightReceived(v); }},
            {"horn", [this](QString v) { emit hornReceived(v); }},
            {"lightslow", [this](QString v) { emit lightsLowReceived(v); }},
            {"lightspark", [this](QString v) { emit lightSparkReceived(v); }},
            {"ismoving", [this](QString v) { emit isMovingReceived(v); }},
            {"batterypercentage", [this](QString v) { emit batteryPercentageReceived(v); }}  // Para corrigir bateria por percentagem
        };

        m_running = true;

        while (m_running) {
            if (!m_context) {
                qWarning() << "Contexto foi destruído!";
                break;
            }

            zmq::pollitem_t items[] = {{*m_socket, 0, ZMQ_POLLIN, 0}};
            zmq::poll(items, 1, std::chrono::milliseconds(100));

            if (items[0].revents & ZMQ_POLLIN) {
                zmq::message_t message;
                zmq::recv_result_t result = m_socket->recv(message, zmq::recv_flags::none);

                if (!result) {
                    qWarning() << "Erro ao receber mensagem!";
                    continue;
                }

                std::string msg(static_cast<const char*>(message.data()), message.size());
                qDebug() << "Mensagem bruta recebida:" << QString::fromStdString(msg);

                QString receivedValue = QString::fromStdString(msg);
                if (!receivedValue.isEmpty() && receivedValue.back() == QChar::Null) {
                    receivedValue.chop(1);
                }
                qDebug() << "Mensagem processada:" << receivedValue;

                auto parts = receivedValue.split(" ", Qt::SkipEmptyParts);
                qDebug() << "Partes da mensagem:" << parts;
                if (parts.size() == 2 && handlers.contains(parts[0])) {
                    qDebug() << "Chamando handler para tópico:" << parts[0] << "com valor:" << parts[1];
                    handlers[parts[0]](parts[1]);
                } else {
                    qWarning() << "Formato inválido ou tópico desconhecido:" << receivedValue;
                }
            }
        }

        if (m_socket) m_socket->close();
        if (m_context) m_context->close();

    } catch (const zmq::error_t &e) {
        qWarning() << "Erro no loop:" << e.what();
    }
}

void ZMQReader::stop() {
    m_running = false;  // Sinaliza para a thread parar
    if (isRunning()) {  // Verifica se a thread está ativa
        wait();         // Aguarda até que a thread termine
    }
}
