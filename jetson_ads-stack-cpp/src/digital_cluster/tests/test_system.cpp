#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "../System.h"
#include <QSignalSpy>

TEST_CASE("Testes completos da Classe System", "[system]") {
    System system;

    // Spies para todos os sinais
    QSignalSpy spySpeed(&system, &System::speedChanged);
    QSignalSpy spyBattery(&system, &System::batteryPerChanged);
    QSignalSpy spyHeadLights(&system, &System::headLightsChanged);
    QSignalSpy spyBrakeLights(&system, &System::brakeLightChanged);
    QSignalSpy spyTurnLeft(&system, &System::turnLightLeftChanged);
    QSignalSpy spyTurnRight(&system, &System::turnLightRightChanged);
    QSignalSpy spyEmergencyLights(&system, &System::emergencyLightsChanged);
    QSignalSpy spyTotalDistance(&system, &System::totalDistanceChanged);

    // Testes para velocidade
    SECTION("Alterar velocidade emite sinal") {
        system.setSpeed("120");
        REQUIRE(system.speed() == "120");
        REQUIRE(spySpeed.count() == 1);
    }

    SECTION("Velocidade igual não emite sinal") {
        system.setSpeed("120");
        system.setSpeed("120");
        REQUIRE(spySpeed.count() == 1);
    }

    // Testes para bateria
    SECTION("Alterar bateria emite sinal") {
        system.setBatteryPer("75");
        REQUIRE(system.batteryPer() == "75");
        REQUIRE(spyBattery.count() == 1);
    }

    SECTION("Bateria igual não emite sinal") {
        system.setBatteryPer("75");
        system.setBatteryPer("75");
        REQUIRE(spyBattery.count() == 1);
    }

    // Testes para faróis
    SECTION("Alterar faróis emite sinal") {
        system.setHeadLights("true");
        REQUIRE(system.headLights() == "true");
        REQUIRE(spyHeadLights.count() == 1);
    }

    SECTION("Faróis iguais não emitem sinal") {
        system.setHeadLights("true");
        system.setHeadLights("true");
        REQUIRE(spyHeadLights.count() == 1);
    }

    // Testes para luz de freio
    SECTION("Alterar luz de freio emite sinal") {
        system.setBrakeLight("true");
        REQUIRE(system.brakeLight() == "true");
        REQUIRE(spyBrakeLights.count() == 1);
    }

    SECTION("Luz de freio igual não emite sinal") {
        system.setBrakeLight("true");
        system.setBrakeLight("true");
        REQUIRE(spyBrakeLights.count() == 1);
    }

    // Testes para seta esquerda
    SECTION("Alterar seta esquerda emite sinal") {
        system.setTurnLightLeft("true");
        REQUIRE(system.turnLightLeft() == "true");
        REQUIRE(spyTurnLeft.count() == 1);
    }

    SECTION("Seta esquerda igual não emite sinal") {
        system.setTurnLightLeft("true");
        system.setTurnLightLeft("true");
        REQUIRE(spyTurnLeft.count() == 1);
    }

    // Testes para seta direita
    SECTION("Alterar seta direita emite sinal") {
        system.setTurnLightRight("true");
        REQUIRE(system.turnLightRight() == "true");
        REQUIRE(spyTurnRight.count() == 1);
    }

    SECTION("Seta direita igual não emite sinal") {
        system.setTurnLightRight("true");
        system.setTurnLightRight("true");
        REQUIRE(spyTurnRight.count() == 1);
    }

    // Testes para luzes de emergência
    SECTION("Alterar luzes de emergência emite sinal") {
        system.setEmergencyLights("true");
        REQUIRE(system.emergencyLights() == "true");
        REQUIRE(spyEmergencyLights.count() == 1);
    }

    SECTION("Luzes de emergência iguais não emitem sinal") {
        system.setEmergencyLights("true");
        system.setEmergencyLights("true");
        REQUIRE(spyEmergencyLights.count() == 1);
    }

    // Testes para distância total
    SECTION("Alterar distância total emite sinal") {
        system.setTotalDistance("3000");
        REQUIRE(system.totalDistance() == "3000");
        REQUIRE(spyTotalDistance.count() == 1);
    }

    SECTION("Distância total igual não emite sinal") {
        system.setTotalDistance("3000");
        system.setTotalDistance("3000");
        REQUIRE(spyTotalDistance.count() == 1);
    }
}


// correr parcialmente com tests_system -s