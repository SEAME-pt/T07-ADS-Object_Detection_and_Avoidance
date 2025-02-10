
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclebodylightsbeamlow.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Body_Lights_Beam_Low implementation.
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsBeamLow : public IVehicleBodyLightsBeamLow
{
public:
    explicit VehicleBodyLightsBeamLow();
    ~VehicleBodyLightsBeamLow();
public:
    /**
    * IsOn Indicates if light is on or off. True = On. False = Off.
    */
    void setIsOn(bool IsOn) override;
    bool getIsOn() const override;
    
    /**
    * IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    void setIsDefect(bool IsDefect) override;
    bool getIsDefect() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Beam_Low changes and signal emission.
    * @return The publisher for Vehicle_Body_Lights_Beam_Low.
    */
    IVehicleBodyLightsBeamLowPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Body_Lights_Beam_Low. */
    std::unique_ptr<IVehicleBodyLightsBeamLowPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Body_Lights_Beam_Low. */
    VehicleBodyLightsBeamLowData m_data;
};
} // namespace Vehicle
} // namespace Cpp
