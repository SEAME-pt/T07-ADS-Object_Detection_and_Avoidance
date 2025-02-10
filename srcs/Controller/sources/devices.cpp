#include <iostream>
#include <string>

#include "devices.hpp"

#define CHIP_NAME "gpiochip0"  // Jetson usa gpiochip0

// Variáveis de estado
bool isLightsLowOn = false;
bool isLightsHighOn = false;
bool isLightsLeftOn = false;
bool isLightsRightOn = false;
bool isLightsEmergencyOn = false;

// Estruturas para gerenciar GPIO
gpiod_chip *chip;
gpiod_line *line_break, *line_low, *line_high, *line_left, *line_right;

// Função para configurar GPIOs
void setupGPIO() {
    chip = gpiod_chip_open_by_name(CHIP_NAME);
    if (!chip) {
        std::cerr << "Erro: Não foi possível abrir " << CHIP_NAME << std::endl;
        exit(1);
    }

    line_break = gpiod_chip_get_line(chip, GPIO_PIN_BREAK_LIGHTS);
    line_low = gpiod_chip_get_line(chip, GPIO_PIN_LOW_LIGHTS);
    line_high = gpiod_chip_get_line(chip, GPIO_PIN_HIGH_LIGHTS);
    line_left = gpiod_chip_get_line(chip, GPIO_PIN_LEFT_LIGHTS);
    line_right = gpiod_chip_get_line(chip, GPIO_PIN_RIGHT_LIGHTS);

    if (!line_break || !line_low || !line_high || !line_left || !line_right) {
        std::cerr << "Erro ao obter linhas GPIO!" << std::endl;
        exit(1);
    }

    gpiod_line_request_output(line_break, "car_control", 1);
    gpiod_line_request_output(line_low, "car_control", 1);
    gpiod_line_request_output(line_high, "car_control", 1);
    gpiod_line_request_output(line_left, "car_control", 1);
    gpiod_line_request_output(line_right, "car_control", 1);
}

void setGPIO(gpiod_line *line, int value) {
    if (line) gpiod_line_set_value(line, value);
}

// Funções de controle do carro
void breakOnPressed(zmq::socket_t &pub) {
    std::cout << "break on\n";
    setGPIO(line_break, 0);
    pub.send(zmq::buffer("break true"), zmq::send_flags::none);
}

void breakOnReleased(zmq::socket_t &pub) {
    std::cout << "break off\n";
    setGPIO(line_break, 1);
    pub.send(zmq::buffer("break false"), zmq::send_flags::none);
}

void lightsLowToggle(zmq::socket_t &pub) {
    isLightsLowOn = !isLightsLowOn;
    setGPIO(line_low, isLightsLowOn ? 0 : 1);
    pub.send(zmq::buffer("lightslow " + std::string(isLightsLowOn ? "true" : "false")), zmq::send_flags::none);
}

void lightsHighToggle(zmq::socket_t &pub) {
    isLightsHighOn = !isLightsHighOn;
    setGPIO(line_high, isLightsHighOn ? 0 : 1);
    pub.send(zmq::buffer("lightshigh " + std::string(isLightsHighOn ? "true" : "false")), zmq::send_flags::none);
}

void emergencyOnLights(zmq::socket_t &pub) {
    isLightsEmergencyOn = !isLightsEmergencyOn;
    setGPIO(line_left, isLightsEmergencyOn ? 0 : 1);
    setGPIO(line_right, isLightsEmergencyOn ? 0 : 1);
    pub.send(zmq::buffer("emergency " + std::string(isLightsEmergencyOn ? "true" : "false")), zmq::send_flags::none);
}

void indicationLightsRight(zmq::socket_t &pub) {
    isLightsRightOn = !isLightsRightOn;
    setGPIO(line_right, isLightsRightOn ? 0 : 1);
    if (isLightsLeftOn) setGPIO(line_left, 1);
    pub.send(zmq::buffer("lightsright " + std::string(isLightsRightOn ? "true" : "false")), zmq::send_flags::none);
}

void indicationLightsLeft(zmq::socket_t &pub) {
    isLightsLeftOn = !isLightsLeftOn;
    setGPIO(line_left, isLightsLeftOn ? 0 : 1);
    if (isLightsRightOn) setGPIO(line_right, 1);
    pub.send(zmq::buffer("lightsleft " + std::string(isLightsLeftOn ? "true" : "false")), zmq::send_flags::none);
}
