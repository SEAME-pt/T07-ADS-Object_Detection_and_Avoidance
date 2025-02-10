
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclebodylights.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Body_Lights implementation.
*/
class CPP_VEHICLE_EXPORT VehicleBodyLights : public IVehicleBodyLights
{
public:
    explicit VehicleBodyLights();
    ~VehicleBodyLights();
public:
    /**
    * LightSwitch Status of the vehicle main light switch.
    */
    void setLightSwitch(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch) override;
    Vehicle_Body_Lights_LightSwitch_ValueEnum getLightSwitch() const override;
    
    /**
    * IsHighBeamSwitchOn Status of the high beam switch. True = high beam enabled. False = high beam not enabled.
    */
    void setIsHighBeamSwitchOn(bool IsHighBeamSwitchOn) override;
    bool getIsHighBeamSwitchOn() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights changes and signal emission.
    * @return The publisher for Vehicle_Body_Lights.
    */
    IVehicleBodyLightsPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Body_Lights. */
    std::unique_ptr<IVehicleBodyLightsPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Body_Lights. */
    VehicleBodyLightsData m_data;
};
} // namespace Vehicle
} // namespace Cpp
