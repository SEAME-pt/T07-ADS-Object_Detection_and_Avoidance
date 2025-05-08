#define CATCH_CONFIG_MAIN
// #include <catch2/catch.hpp>
#include <catch2/catch_all.hpp>   
#include "../zmqreader.h"
#include "../System.h"
#include <zmq.hpp>
#include <QCoreApplication>
#include <QSignalSpy>
#include <thread>
#include <chrono>   

// Mock da classe ZMQ para testes
class MockZMQContext : public zmq::context_t {
public:
    MockZMQContext() : zmq::context_t(1) {}
};

class MockZMQSocket : public zmq::socket_t {
public:
    MockZMQSocket(zmq::context_t& context, int type) 
        : zmq::socket_t(context, type) {}
    
    void send(const std::string& data) {
        lastSent = data;
    }

    std::string lastSent;
};

TEST_CASE("Teste da Classe ZMQReader", "[zmqreader]") {

    // valores dummy
    int argc = 1;
    char* argv[] = {(char*)"test"};
    QCoreApplication app(argc, argv); // Necessário para eventos Qt

    System system;
    ZMQReader zmqReader("tcp://localhost:5555");
    QSignalSpy spySpeed(&zmqReader, &ZMQReader::speedReceived);
    QSignalSpy spyBattery(&zmqReader, &ZMQReader::batteryReceived);
    QSignalSpy spyHeadLights(&zmqReader, &ZMQReader::headLightsReceived);
    QSignalSpy spyBrake(&zmqReader, &ZMQReader::brakeLightReceived);
    QSignalSpy spyTurnLightLeft(&zmqReader, &ZMQReader::turnLightLeftReceived);
    QSignalSpy spyTurnLightRight(&zmqReader, &ZMQReader::turnLightRightReceived);
    QSignalSpy spyEmergencyLights(&zmqReader, &ZMQReader::emergencyLightsReceived);
    QSignalSpy spyTotalDistance(&zmqReader, &ZMQReader::totalDistanceReceived);
    

    // // Simular recebimento de mensagem
    // SECTION("Testar recebimento de velocidade") {
    //     // Mockar a lógica de recebimento (exemplo simplificado)
    //     zmqReader.processMessage("SPEED:100");
    //     REQUIRE(spySpeed.count() == 1);
    //     REQUIRE(system.speed() == "100");
    // }

    zmq::context_t context(1);
    zmq::socket_t publisher(context, zmq::socket_type::pub);
    publisher.bind("tcp://*:5555");

    zmqReader.start(); // Inicia a thread do ZMQReader

    // Aguarde a conexão ser estabelecida
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    SECTION("Testar recebimento de todos os sinais") {
        // Envie uma mensagem de teste
        std::string message = "speed 100";
        publisher.send(zmq::buffer(message), zmq::send_flags::none);

        std::string message1 = "battery 100";
        publisher.send(zmq::buffer(message1), zmq::send_flags::none);

        std::string message2 = "lightshigh true";
        publisher.send(zmq::buffer(message2), zmq::send_flags::none);

        std::string message3 = "brake true";
        publisher.send(zmq::buffer(message3), zmq::send_flags::none);

        std::string message4 = "lightsemergency true";
        publisher.send(zmq::buffer(message4), zmq::send_flags::none);

        std::string message5 = "lightsleft true";
        publisher.send(zmq::buffer(message5), zmq::send_flags::none);

        std::string message6 = "lightsright true";
        publisher.send(zmq::buffer(message6), zmq::send_flags::none);
        
        std::string message7 = "totaldistance 100";
        publisher.send(zmq::buffer(message7), zmq::send_flags::none);

        // Aguarde o sinal ser emitido
        REQUIRE(spySpeed.wait(1000)); // Espera até 1 segundo
        REQUIRE(spySpeed.count() == 1);
        REQUIRE(spySpeed.takeFirst().at(0).toString() == "100");

        // Validação do sinal battery
        // REQUIRE(spyBattery.wait(2000));
        REQUIRE(spyBattery.count() == 1);
        REQUIRE(spyBattery.takeFirst().at(0).toString() == "100");

        // Validação do sinal lightshigh
        // REQUIRE(spyHeadLights.wait(1000));
        REQUIRE(spyHeadLights.count() == 1);
        REQUIRE(spyHeadLights.takeFirst().at(0).toString() == "true");

        // Validação do sinal brake
        // REQUIRE(spyBrake.wait(1000));
        REQUIRE(spyBrake.count() == 1);
        REQUIRE(spyBrake.takeFirst().at(0).toString() == "true");

        // Validação do sinal lightsemergency
        // REQUIRE(spyEmergencyLights.wait(1000));
        REQUIRE(spyEmergencyLights.count() == 1);
        REQUIRE(spyEmergencyLights.takeFirst().at(0).toString() == "true");

        // Validação do sinal lightsleft
        // REQUIRE(spyTurnLightLeft.wait(1000));
        REQUIRE(spyTurnLightLeft.count() == 1);
        REQUIRE(spyTurnLightLeft.takeFirst().at(0).toString() == "true");

        // Validação do sinal lightsright
        // REQUIRE(spyTurnLightRight.wait(1000));
        REQUIRE(spyTurnLightRight.count() == 1);
        REQUIRE(spyTurnLightRight.takeFirst().at(0).toString() == "true");

        // Validação do sinal totaldistance
        // REQUIRE(spyTotalDistance.wait(1000));
        REQUIRE(spyTotalDistance.count() == 1);
        REQUIRE(spyTotalDistance.takeFirst().at(0).toString() == "100");
    }

    SECTION("Envio e recepção de 10 valores de velocidade") {

        // Envia 1000 mensagens no formato "speed <valor>"
        for (int i = 0; i < 10; ++i) {
            std::string message = "speed " + std::to_string(i);
            publisher.send(zmq::buffer(message), zmq::send_flags::none);
            // Opcional: você pode inserir um pequeno delay se necessário:
            // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Cria um timer para esperar que todas as mensagens sejam processadas (timeout de 5 segundos)
        QElapsedTimer timer;
        timer.start();
        while (spySpeed.count() < 1000 && timer.elapsed() < 5000) {
            // Processa os eventos pendentes do Qt
            QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Verifica se foram recebidas exatamente 1000 mensagens
        REQUIRE(spySpeed.count() == 10);

        // Verifica se cada mensagem possui o valor correto
        for (int i = 0; i < 10; ++i) {
            // takeFirst() remove e retorna a lista de argumentos da emissão do sinal
            QList<QVariant> argumentos = spySpeed.takeFirst();
            QString receivedSpeed = argumentos.at(0).toString();
            // Verifica se o valor recebido é igual ao esperado (como QString)
            REQUIRE(receivedSpeed == QString::number(i));
        }
    }

    SECTION("Envio e recepção de 1000 valores de velocidade e bateria") {
        // Envio de 1000 mensagens para velocidade e 1000 para bateria
        for (int i = 0; i < 1000; ++i) {
            // Mensagem de velocidade no formato "speed <valor>"
            std::string speedMessage = "speed " + std::to_string(i);
            publisher.send(zmq::buffer(speedMessage), zmq::send_flags::none);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            // Mensagem de bateria no formato "battery <valor>"
            std::string batteryMessage = "battery " + std::to_string(i);
            publisher.send(zmq::buffer(batteryMessage), zmq::send_flags::none);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Aguarda que todas as mensagens sejam processadas
        QElapsedTimer timer;
        timer.start();
        // O loop continua enquanto não tivermos recebido 1000 mensagens de cada ou até 5 segundos de timeout
        while ((spySpeed.count() < 1000 || spyBattery.count() < 1000) && timer.elapsed() < 50000) {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Verifica se foram recebidas exatamente 1000 mensagens de velocidade e 1000 de bateria
        REQUIRE(spySpeed.count() == 1000);
        REQUIRE(spyBattery.count() == 1000);

        // Verifica se cada mensagem recebida contém o valor correto
        for (int i = 0; i < 1000; ++i) {
            // Cada chamada a takeFirst() remove e retorna a lista de argumentos do sinal emitido
            QList<QVariant> speedArgs = spySpeed.takeFirst();
            QList<QVariant> batteryArgs = spyBattery.takeFirst();

            // Assume-se que o primeiro argumento do sinal é a string contendo o valor
            QString receivedSpeed = speedArgs.at(0).toString();
            QString receivedBattery = batteryArgs.at(0).toString();

            REQUIRE(receivedSpeed == QString::number(i));
            REQUIRE(receivedBattery == QString::number(i));
        }
    }

    zmqReader.stop();
    // Adicione mais testes para battery, headLights, etc.
}


// correr parcialmente com tests_zmqreader -s