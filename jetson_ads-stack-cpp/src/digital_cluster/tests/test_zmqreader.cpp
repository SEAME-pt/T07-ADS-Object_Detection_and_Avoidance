#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "../zmqreader.h"
#include "../System.h"
#include <zmq.hpp>
#include <QCoreApplication>
#include <QSignalSpy>
#include <QElapsedTimer>
#include <thread>
#include <chrono>

TEST_CASE("Testes completos da Classe ZMQReader", "[zmqreader]") {
    int argc = 1;
    char* argv[] = {(char*)"test"};
    QCoreApplication app(argc, argv);
    System system;

    // Configuração comum para todos os testes
    zmq::context_t context(1);
    zmq::socket_t publisher(context, zmq::socket_type::pub);
    publisher.bind("tcp://*:5555");

    ZMQReader zmqReader("tcp://localhost:5555");
    QSignalSpy spySpeed(&zmqReader, &ZMQReader::speedReceived);
    QSignalSpy spyBattery(&zmqReader, &ZMQReader::batteryReceived);
    QSignalSpy spyHeadLights(&zmqReader, &ZMQReader::headLightsReceived);
    QSignalSpy spyBrake(&zmqReader, &ZMQReader::brakeLightReceived);
    QSignalSpy spyTurnLightLeft(&zmqReader, &ZMQReader::turnLightLeftReceived);
    QSignalSpy spyTurnLightRight(&zmqReader, &ZMQReader::turnLightRightReceived);
    QSignalSpy spyEmergencyLights(&zmqReader, &ZMQReader::emergencyLightsReceived);
    QSignalSpy spyTotalDistance(&zmqReader, &ZMQReader::totalDistanceReceived);

    zmqReader.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    SECTION("Teste de mensagens válidas") {
        const std::vector<std::pair<std::string, QSignalSpy*>> testCases = {
            {"speed 100", &spySpeed},
            {"battery 75", &spyBattery},
            {"lightshigh true", &spyHeadLights},
            {"brake true", &spyBrake},
            {"lightsemergency true", &spyEmergencyLights},
            {"lightsleft true", &spyTurnLightLeft},
            {"lightsright true", &spyTurnLightRight},
            {"totaldistance 3000", &spyTotalDistance}
        };

        for (const auto& [message, spy] : testCases) {
            publisher.send(zmq::buffer(message.data(), message.size()), zmq::send_flags::none);
            REQUIRE(spy->wait(4000));  // Timeout aumentado para 3 segundos
            REQUIRE(spy->count() == 1);
            spy->clear();
        }
    }

    SECTION("Teste de mensagens inválidas") {
        spySpeed.clear();
        spyBattery.clear();
        spyHeadLights.clear();

        const std::vector<std::string> invalidMessages = {
            "invalid_command 123",
            "speed",
            "",
            "lightsright",
            "batterys invalid_value"
        };

        for (const auto& msg : invalidMessages) {
            publisher.send(zmq::buffer(msg.data(), msg.size()), zmq::send_flags::none);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Pequeno delay
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        REQUIRE(spySpeed.count() == 0);
        REQUIRE(spyBattery.count() == 0);
        REQUIRE(spyHeadLights.count() == 0);
    }

    SECTION("Teste de stress com múltiplas threads") {
        const int numThreads = 4;
        const int messagesPerThread = 250;
        
        // Cria um mutex para proteger o envio pelo socket já criado (publisher)
        std::mutex publisherMutex;
        
        auto sender = [&](int threadId) {
            for (int i = 0; i < messagesPerThread; ++i) {
                std::string msg = "speed " + std::to_string(threadId * 1000 + i);
                {
                    std::lock_guard<std::mutex> lock(publisherMutex);
                    publisher.send(zmq::buffer(msg.data(), msg.size()), zmq::send_flags::none);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(1000));  // Diminuir congestionamento
            }
        };
    
        std::vector<std::thread> threads;
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(sender, i);
        }
    
        QElapsedTimer timer;
        timer.start();
        // Aguarda até que todas as mensagens sejam recebidas ou até o timeout de 10 segundos
        while (spySpeed.count() < numThreads * messagesPerThread && timer.elapsed() < 10000) {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    
        for (auto& t : threads)
            t.join();
        
        REQUIRE(spySpeed.count() == numThreads * messagesPerThread);
    }
    

    SECTION("Teste de stop de emergência") {
        std::thread sender([&]() {
            zmq::socket_t sock(context, zmq::socket_type::pub);
            sock.connect("tcp://localhost:5555");
            
            for (int i = 0; i < 1000; ++i) {
                std::string msg = "speed " + std::to_string(i);
                sock.send(zmq::buffer(msg.data(), msg.size()), zmq::send_flags::none);
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        zmqReader.stop();
        sender.join();

        REQUIRE(spySpeed.count() < 1000);
    }

    SECTION("Teste de limite de valores") {
        const std::string msg = "speed 214748364";
        publisher.send(zmq::buffer(msg.data(), msg.size()), zmq::send_flags::none);
        REQUIRE(spySpeed.wait(3000));  // Timeout aumentado
        if (spySpeed.count() > 0) {
            QVariant receivedValue = spySpeed.takeFirst().at(0);
            // qDebug() << "Received value:" << receivedValue;
            // qDebug() << "As string:" << receivedValue.toString();
            REQUIRE(receivedValue.toString() == "214748364");
        }
    }

    // Limpeza final
    zmqReader.stop();
    publisher.close();
    context.close();

    // Espera para garantir o fecho da thread
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}


// correr parcialmente com tests_zmqreader -s ou --reporter compact --success