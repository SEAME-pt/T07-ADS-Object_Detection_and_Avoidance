#include "JetSnailsCar.hpp"
#include "vehicle/implementation/vehiclevehicleidentification.h"
#include "vehicle/implementation/vehiclebodyhorn.h"
#include "vehicle/implementation/vehiclebodylightsbackup.h"
#include "vehicle/implementation/vehiclebodylightsbeamhigh.h"
#include "vehicle/implementation/vehiclebodylightsbeamlow.h"
#include "vehicle/implementation/vehiclebodylightsbrake.h"
#include "vehicle/implementation/vehiclebodylightsdirectionindicatorleft.h"
#include "vehicle/implementation/vehiclebodylightsdirectionindicatorright.h"
#include "vehicle/implementation/vehiclebodylightsparking.h"
#include "vehicle/implementation/vehiclecurrentlocation.h"
#include "vehicle/implementation/vehiclepowertraintractionbattery.h"
#include "vehicle/implementation/vehiclelowvoltagebattery.h"


// Constructors
JetSnailsCar::JetSnailsCar() : ACar(), _maxSpeed(0) {
    std::cout << "Default constructor called for JetSnailsCar" << std::endl;

    // Speed/Trip and Info
    vehicle = std::make_unique<Vehicle>();
    vehicleIdentification = std::make_unique<VehicleVehicleIdentification>();

    // Lights and Horn
    vehicleBodyHorn = std::make_unique<VehicleBodyHorn>();
    vehicleBodyLightsBeamHigh = std::make_unique<VehicleBodyLightsBeamHigh>();
    vehicleBodyLightsBeamLow = std::make_unique<VehicleBodyLightsBeamLow>();
    vehicleBodyLightsDirectionIndicatorLeft = std::make_unique<VehicleBodyLightsDirectionIndicatorLeft>();
    vehicleBodyLightsDirectionIndicatorRight = std::make_unique<VehicleBodyLightsDirectionIndicatorRight>();
    vehicleBodyLightsParking = std::make_unique<VehicleBodyLightsParking>();
    vehicleBodyLightsBrake = std::make_unique<VehicleBodyLightsBrake>();
    vehicleBodyLightsBackup = std::make_unique<VehicleBodyLightsBackup>();
    vehicleCurrentLocation = std::make_unique<VehicleCurrentLocation>();
    vehicleLowVoltageBattery = std::make_unique<VehicleLowVoltageBattery>();
    vehiclePowertrainTractionBattery = std::make_unique<VehiclePowertrainTractionBattery>();
    

    // Vehicle
    vehicle->setSpeed(0.0);
    vehicle->setTraveledDistance(0.0);
    vehicle->setTraveledDistanceSinceStart(0.0); // trip
    vehicle->setIsMoving("false");
    vehicle->setAverageSpeed(0.0);

    // Lights
    vehicleBodyHorn->setIsActive(false);
    vehicleBodyLightsBackup->setIsOn(false);
    vehicleBodyLightsBrake->setIsActive(Vehicle_Body_Lights_Brake_IsActive_ValueEnum::INACTIVE);
    vehicleBodyLightsParking->setIsOn(false);
    vehicleBodyLightsBeamHigh->setIsOn(false);
    vehicleBodyLightsBeamLow->setIsOn(false);
    vehicleBodyLightsDirectionIndicatorLeft->setIsSignaling(false);
    vehicleBodyLightsDirectionIndicatorRight->setIsSignaling(false);

    // Location
    vehicleCurrentLocation->setLatitude(41.1486580617545);
    vehicleCurrentLocation->setLongitude(-8.613125578821894);

    // Baterry



    // Info
    vehicleIdentification->setVin("CD010XXX");
    vehicleIdentification->setWmi("Europe");
    vehicleIdentification->setBrand("Delorean Motor Company");
    vehicleIdentification->setModel("BTF Delorean");
    vehicleIdentification->setYear(1981);
    vehicleIdentification->setBodyType("hatchback");
    vehicleIdentification->setLicensePlate("00-XX-00");
    vehicleIdentification->setVehicleSeatingCapacity(2);
    vehicleIdentification->setVehicleSpecialUsage("Travel in time");
    vehicleIdentification->setVehicleExteriorColor("Gray");
    vehicleIdentification->setOptionalExtras({"Flux capacitor",""});
}

JetSnailsCar::JetSnailsCar(const JetSnailsCar& original) : ACar(original), _maxSpeed(original._maxSpeed) {
    std::cout << "Copy constructor called for JetSnailsCar" << std::endl;
}

// Destructor
JetSnailsCar::~JetSnailsCar() {
    std::cout << "Destructor called for JetSnailsCar" << std::endl;
}

// Member function
void JetSnailsCar::drive() const {
    ;
}

void JetSnailsCar::printParts(void) const {
    std::cout << "Print parts goes here!" << std::endl;
}

void JetSnailsCar::printSensors(void) const {
    std::cout << "Print Sensors goes here!" << std::endl;
}


void JetSnailsCar::setSpeedSensor(SpeedSensor *speedSensor) {
     _speedSensor = speedSensor;
}

void JetSnailsCar::setParkSensor(ParkSensor *parkSensor) {
     _parkSensor = parkSensor;
}

void JetSnailsCar::setOdometer(Odometer *odometer) {
     _odometer = odometer;
}

float JetSnailsCar::getSpeed() const {
    return this->vehicle->getSpeed();
}

float JetSnailsCar::getDistance() const {
    return _parkSensor->getValue();
}

float JetSnailsCar::getOdometer() const {
    return _odometer->getValue();
}

// Operator overload
JetSnailsCar& JetSnailsCar::operator=(const JetSnailsCar& original) {
    if (this != &original) {
        ACar::operator=(original);
        _maxSpeed = original._maxSpeed;
    }
    std::cout << "Copy assignment operator called for JetSnailsCar" << std::endl;
    return *this;
}
