#pragma once

#include <zmq.hpp>
#include <sstream>
#include <gpiod.h>



// GPIO PINS
#define GPIO_PIN_RIGHT_LIGHTS 16
#define GPIO_PIN_LEFT_LIGHTS 26
#define GPIO_PIN_LOW_LIGHTS 5
#define GPIO_PIN_HIGH_LIGHTS 6
#define GPIO_PIN_BREAK_LIGHTS 13
#define GPIO_PIN_HORN 16
#define GPIO_PIN_PARK 12

// KEY MAP
#define BTN_A 0 
#define BTN_B 1
#define BTN_X 2
#define BTN_Y 3
#define BTN_L 9
#define BTN_R 10
#define BTN_AXIS_LEFT 7
#define BTN_AXIS_RIGHT 8  
#define BTN_HOME 5
#define BTN_SELECT 4
#define BTN_START 6

void    hornOnPressed(zmq::socket_t& pub);
void    hornOnReleased(zmq::socket_t& pub);

void    breakOnReleased(zmq::socket_t& pub);
void    breakOnPressed(zmq::socket_t& pub);

void    lightsLowToggle(zmq::socket_t& pub);
void    lightsHighToggle(zmq::socket_t& pub);


void    indicationLightsLeft(zmq::socket_t& pub);
void    indicationLightsRight(zmq::socket_t& pub);

void    emergencyOnLights(zmq::socket_t& pub);
