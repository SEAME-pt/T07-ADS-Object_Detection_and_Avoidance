#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp> 
#include <QCoreApplication>  //ou QApplication, conforme sua necessidade
#include "../System.h"
#include <QtQml/QQmlEngine>
#include <QtQml/QQmlComponent>
#include <QtQml/QQmlContext>

TEST_CASE("Teste de Integração QML/C++", "[qml][integration]") {
    // Cria uma instância estática da aplicação se ainda não existir
    static int argc = 0;
    static char **argv = nullptr;
    static QCoreApplication app(argc, argv);

    QQmlEngine engine;
    System system;
    engine.rootContext()->setContextProperty("systemHandler", &system);

    QQmlComponent component(&engine);
    component.setData(R"(
        import QtQuick 2.15
        Item {
            property string speed: systemHandler.speed
            property string battery: systemHandler.batteryPer
            property string headLights: systemHandler.headLights
            property string brakeLight: systemHandler.brakeLight
            property string turnLightLeft: systemHandler.turnLightLeft
            property string turnLightRight: systemHandler.turnLightRight
            property string emergencyLights: systemHandler.emergencyLights
            property string totalDistance: systemHandler.totalDistance
        }
    )", QUrl());

    QObject *object = component.create();
    REQUIRE(object != nullptr);

    system.setSpeed("80");
    system.setBatteryPer("100");
    system.setHeadLights("true");
    system.setBrakeLight("true");
    system.setTurnLightLeft("true");
    system.setTurnLightRight("true");
    system.setEmergencyLights("true");
    system.setTotalDistance("1000");
    // Atualize os eventos pendentes se necessário
    QCoreApplication::processEvents();
    REQUIRE(object->property("speed").toString() == "80");
    REQUIRE(object->property("battery").toString() == "100");
    REQUIRE(object->property("headLights").toString() == "true");
    REQUIRE(object->property("brakeLight").toString() == "true");
    REQUIRE(object->property("turnLightLeft").toString() == "true");
    REQUIRE(object->property("turnLightRight").toString() == "true");
    REQUIRE(object->property("emergencyLights").toString() == "true");
    REQUIRE(object->property("totalDistance").toString() == "1000");

    delete object;
}
