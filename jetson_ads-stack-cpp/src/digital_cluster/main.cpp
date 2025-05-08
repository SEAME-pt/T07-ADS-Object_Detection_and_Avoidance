#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <zmq.hpp>
#include <QDebug>
#include "System.h"
#include "zmqreader.h"


int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    // System m_systemHandler;

    zmq::context_t zmqContext(1); // Contexto ZMQ
    System m_systemHandler(zmqContext); // Passa o contexto

    QQmlApplicationEngine engine;
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreationFailed,
        &app,
        []() { QCoreApplication::exit(-1); },
        Qt::QueuedConnection);

    // colocar aqui o carregar da classe para depois ao carregar o maincomponent carregar tudo
    QQmlContext *context(engine.rootContext());
    context->setContextProperty("systemHandler", &m_systemHandler);

    // Conectar o sinal lkasChanged para imprimir mudanças -teste
    // QObject::connect(&m_systemHandler, &System::lkasChanged, [&]() {
    //     qDebug() << "Main: lkas mudou para:" << m_systemHandler.lkas();
    // });

    // Inicializar o ZMQReader
    ZMQReader zmqReader("tcp://localhost:5555"); // Endereço da socket

    // Conecta o sinal de velocidade para o System
    QObject::connect(&zmqReader, &ZMQReader::speedReceived, [&](QString speed) {
        m_systemHandler.setSpeed(speed);
    });

    QObject::connect(&zmqReader, &ZMQReader::batteryReceived, [&](QString battery) {
        m_systemHandler.setBatteryPer(battery);
    });

    QObject::connect(&zmqReader, &ZMQReader::headLightsReceived, [&](QString headLights) {
        m_systemHandler.setHeadLights(headLights);
    });

    QObject::connect(&zmqReader, &ZMQReader::brakeLightReceived, [&](QString brakeLight) {
        m_systemHandler.setBrakeLight(brakeLight);
    });

    QObject::connect(&zmqReader, &ZMQReader::turnLightLeftReceived, [&](QString turnLightLeft) {
        m_systemHandler.setTurnLightLeft(turnLightLeft);
    });

    QObject::connect(&zmqReader, &ZMQReader::turnLightRightReceived, [&](QString turnLightRight) {
        m_systemHandler.setTurnLightRight(turnLightRight);
    });

    QObject::connect(&zmqReader, &ZMQReader::emergencyLightsReceived, [&](QString emergencyLights) {
        m_systemHandler.setEmergencyLights(emergencyLights);
    });

    QObject::connect(&zmqReader, &ZMQReader::totalDistanceReceived, [&](QString totalDistance) {
        m_systemHandler.setTotalDistance(totalDistance);
    });

    // QObject::connect(&zmqReader, &ZMQReader::lkasReceived, [&](QString lkas) {
    //     m_systemHandler.setLkas(lkas);
    // });

    QObject::connect(&zmqReader, &ZMQReader::lkasReceived, [&](QString lkas) {
        m_systemHandler.setLkasFromZMQ(lkas); // Usa método especial
    });

    // QObject::connect(&zmqReader, &ZMQReader::autoPilotReceived, [&](QString autoPilot) {
    //     m_systemHandler.setAutoPilot(autoPilot);
    // });

    QObject::connect(&zmqReader, &ZMQReader::autoPilotReceived, [&](QString autoPilot) {
        m_systemHandler.setAutoPilotFromZMQ(autoPilot); // Usa método especial
    });

    QObject::connect(&zmqReader, &ZMQReader::lineLeftReceived, [&](QString lineLeft) {
        m_systemHandler.setLineLeft(lineLeft);
    });

    QObject::connect(&zmqReader, &ZMQReader::lineRightReceived, [&](QString lineRight) {
        m_systemHandler.setLineRight(lineRight);
    });

    zmqReader.start(); // Inicia a thread - corre o run

    engine.loadFromModule("Digital_Cluster", "Main");

    if (engine.rootObjects().isEmpty()) {
        qWarning() << "Erro ao carregar o QML.";
        return -1;
    }

    //zmqReader.stop();

    return app.exec();
}
