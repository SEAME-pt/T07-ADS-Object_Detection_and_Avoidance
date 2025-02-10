
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights interface.
* @see Vehicle_Body_Lights
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLights> testVehicleBodyLights = std::make_unique<VehicleBodyLightsThreadSafeDecorator>(std::make_shared<VehicleBodyLights>());

// Thread safe access
auto lightSwitch = testVehicleBodyLights->getLightSwitch();
testVehicleBodyLights->setLightSwitch(Vehicle_Body_Lights_LightSwitch_ValueEnum::OFF);
auto isHighBeamSwitchOn = testVehicleBodyLights->getIsHighBeamSwitchOn();
testVehicleBodyLights->setIsHighBeamSwitchOn(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsThreadSafeDecorator : public IVehicleBodyLights
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights object to make thread safe.
    */
    explicit VehicleBodyLightsThreadSafeDecorator(std::shared_ptr<IVehicleBodyLights> impl);

    /** Guards and forwards call to Vehicle_Body_Lights implementation. */
    void setLightSwitch(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch) override;
    /** Guards and forwards call to Vehicle_Body_Lights implementation. */
    Vehicle_Body_Lights_LightSwitch_ValueEnum getLightSwitch() const override;

    /** Guards and forwards call to Vehicle_Body_Lights implementation. */
    void setIsHighBeamSwitchOn(bool IsHighBeamSwitchOn) override;
    /** Guards and forwards call to Vehicle_Body_Lights implementation. */
    bool getIsHighBeamSwitchOn() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights.
    */
    IVehicleBodyLightsPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights object which is guarded */
    std::shared_ptr<IVehicleBodyLights> m_impl;
    // Mutex for lightSwitch property
    mutable std::shared_timed_mutex m_lightSwitchMutex;
    // Mutex for isHighBeamSwitchOn property
    mutable std::shared_timed_mutex m_isHighBeamSwitchOnMutex;
};
} // namespace Vehicle
} // namespace Cpp
