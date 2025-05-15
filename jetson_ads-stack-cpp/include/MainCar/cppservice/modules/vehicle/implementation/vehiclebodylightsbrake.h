
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclebodylightsbrake.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Body_Lights_Brake implementation.
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsBrake : public IVehicleBodyLightsBrake
{
public:
    explicit VehicleBodyLightsBrake();
    ~VehicleBodyLightsBrake();
public:
    /**
    * IsActive Indicates if break-light is active. INACTIVE means lights are off. ACTIVE means lights are on. ADAPTIVE means that break-light is indicating emergency-breaking.
    */
    void setIsActive(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive) override;
    Vehicle_Body_Lights_Brake_IsActive_ValueEnum getIsActive() const override;
    
    /**
    * IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    void setIsDefect(bool IsDefect) override;
    bool getIsDefect() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Brake changes and signal emission.
    * @return The publisher for Vehicle_Body_Lights_Brake.
    */
    IVehicleBodyLightsBrakePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Body_Lights_Brake. */
    std::unique_ptr<IVehicleBodyLightsBrakePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Body_Lights_Brake. */
    VehicleBodyLightsBrakeData m_data;
};
} // namespace Vehicle
} // namespace Cpp
