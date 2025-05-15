#include <pigpio.h>
#include <thread>
#include "JetSnailsCar.hpp"
#include "vehicle.h"
#include "VehicleInfoSubscriber.hpp"
#include "VehicleSubscriber.hpp"
#include "ControllerSubscriber.hpp"
#include "HornSubscriber.hpp"
#include "Battery.hpp"

void publishControllerState(ControllerSubscriber& controllerSubscriber, zmq::socket_t& publisher) {
    const std::vector<std::string> controls = {
        "horn", "lightslow", "lightshigh", "break",
        "lightsleft", "lightsright", "lightsemergency", "lightspark", "battery"
    };

    for (const auto& control : controls) {
        std::string state = controllerSubscriber.getControlState(control);

        if (!state.empty()) {
            zmq::message_t message(control.size() + state.size() + 1);
            snprintf(static_cast<char*>(message.data()), message.size() + 1, "%s %s", control.c_str(), state.c_str());
            publisher.send(message, zmq::send_flags::none);
        }
    }
}

void readAndPublishSensorData(SpeedSensor& speedSensor, JetSnailsCar& delorean) {
    const int updateInterval = 30; // Intervalo de atualização reduzido para 30ms

    while (true) {
        // Leia a velocidade do sensor
        speedSensor.readData();

        // float batteryPercentage = ina219.getBatteryPercentage();
        float currentSpeed = speedSensor.getValue();


        // Atualize o estado no veículo
        delorean.vehicle->setSpeed(currentSpeed);

        // Publicar estados do controlador
        // publishControllerState(controllerSubscriber, publisher_sensors);

        // Atraso mais curto antes da próxima iteração
        std::this_thread::sleep_for(std::chrono::milliseconds(updateInterval));
    }
}

int main(int argc, char *argv[]) {
    zmq::context_t context(1);
    zmq::socket_t publisher_sensors(context, zmq::socket_type::pub);
    publisher_sensors.bind("tcp://*:5555");

    ControllerSubscriber controllerSubscriber("tcp://localhost:5556");
    controllerSubscriber.startListening();

    std::string can_device = "can0";
    if (argc > 1) {
        can_device = argv[1];
    }

    CANBus canBus(can_device, 500000);
    JetSnailsCar delorean;

    SpeedSensor speedSensor(canBus, 0x100); // Speed
    // Odometer odometer(canBus, 0x200); // Odometer
    // INA219 ina219("/dev/i2c-1");

    vehicleSensors vehicle_sensors(publisher_sensors);
    vehicleInformation vehicle_info(publisher_sensors);
    delorean.vehicle->_getPublisher().subscribeToAllChanges(vehicle_sensors);

    // delorean.vehicleIdentification->_getPublisher().setPublisher(&vehicle_info);

    // Criar e iniciar a thread para leitura e publicação de dados
    std::thread sensorThread(readAndPublishSensorData, std::ref(speedSensor), std::ref(delorean));

    // Aguardar a thread
    sensorThread.join();

    return 0;
}
