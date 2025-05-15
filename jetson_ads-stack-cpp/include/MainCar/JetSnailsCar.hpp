#pragma once

#include "ACar.hpp"
#include "ISensor.hpp"
#include "SpeedSensor.hpp"
#include "ParkSensor.hpp"
#include "Odometer.hpp"
#include "vehicle.h"
//#include "CANBus.hpp"

#include <vector>
#include <iostream>

using namespace Cpp::Vehicle;

class JetSnailsCar : public ACar {
    private:
        int _maxSpeed;
        SpeedSensor *_speedSensor;
        ParkSensor *_parkSensor;
        Odometer *_odometer;
        

    public:
        JetSnailsCar();
        JetSnailsCar(const JetSnailsCar& other);
        JetSnailsCar& operator=(const JetSnailsCar& other);
        ~JetSnailsCar();

        // Vehicle
        std::unique_ptr<IVehicle> vehicle;
        std::unique_ptr<IVehicleVehicleIdentification> vehicleIdentification;
        std::unique_ptr<IVehiclePowertrainTransmission> vehiclePowertrainTransmission;

        // Battery
        std::unique_ptr<IVehicleLowVoltageBattery> vehicleLowVoltageBattery;
        std::unique_ptr<IVehiclePowertrainTractionBattery> vehiclePowertrainTractionBattery;

        // Lights and Horn
        std::unique_ptr<IVehicleBodyHorn> vehicleBodyHorn;
        std::unique_ptr<IVehicleBodyLightsBeamHigh> vehicleBodyLightsBeamHigh;
        std::unique_ptr<IVehicleBodyLightsBeamLow> vehicleBodyLightsBeamLow;
        std::unique_ptr<IVehicleBodyLightsDirectionIndicatorLeft> vehicleBodyLightsDirectionIndicatorLeft;
        std::unique_ptr<IVehicleBodyLightsDirectionIndicatorRight> vehicleBodyLightsDirectionIndicatorRight;
        std::unique_ptr<IVehicleBodyLightsParking> vehicleBodyLightsParking;
        std::unique_ptr<IVehicleBodyLightsBrake> vehicleBodyLightsBrake;
        std::unique_ptr<IVehicleBodyLightsBackup> vehicleBodyLightsBackup;

        // Localization
        std::unique_ptr<IVehicleCurrentLocation> vehicleCurrentLocation;




        void drive() const;
        void printParts() const;
        void printSensors() const;

	    void setSpeedSensor(SpeedSensor *speedSensor);
        void setParkSensor(ParkSensor *parkSensor);
        void setOdometer(Odometer *odometer);

        float getSpeed() const;
        float getDistance() const;
        float getOdometer() const;
};

std::ostream &  operator<<( std::ostream & o, JetSnailsCar & i );
