#pragma once

#include <zmq.hpp>
#include <sstream>
#include <gpiod.h>



// Definir as constantes para os pinos GPIO
#define GPIO_PIN_BREAK_LIGHTS 17  // Número de pino para luzes de freio
#define GPIO_PIN_LOW_LIGHTS 18    // Número de pino para luzes baixas
#define GPIO_PIN_HIGH_LIGHTS 19   // Número de pino para luzes altas
#define GPIO_PIN_LEFT_LIGHTS 20   // Número de pino para luzes à esquerda
#define GPIO_PIN_RIGHT_LIGHTS 21  // Número de pino para luzes à direita


void    breakOnReleased(zmq::socket_t& pub);
void    breakOnPressed(zmq::socket_t& pub);

void    lightsLowToggle(zmq::socket_t& pub);
void    lightsHighToggle(zmq::socket_t& pub);


void    indicationLightsLeft(zmq::socket_t& pub);
void    indicationLightsRight(zmq::socket_t& pub);

void    emergencyOnLights(zmq::socket_t& pub);
