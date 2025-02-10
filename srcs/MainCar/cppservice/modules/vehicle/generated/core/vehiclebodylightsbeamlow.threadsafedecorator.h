
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsBeamLowThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights_Beam_Low interface.
* @see Vehicle_Body_Lights_Beam_Low
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLightsBeamLow> testVehicleBodyLightsBeamLow = std::make_unique<VehicleBodyLightsBeamLowThreadSafeDecorator>(std::make_shared<VehicleBodyLightsBeamLow>());

// Thread safe access
auto isOn = testVehicleBodyLightsBeamLow->getIsOn();
testVehicleBodyLightsBeamLow->setIsOn(false);
auto isDefect = testVehicleBodyLightsBeamLow->getIsDefect();
testVehicleBodyLightsBeamLow->setIsDefect(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsBeamLowThreadSafeDecorator : public IVehicleBodyLightsBeamLow
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights_Beam_Low object to make thread safe.
    */
    explicit VehicleBodyLightsBeamLowThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsBeamLow> impl);

    /** Guards and forwards call to Vehicle_Body_Lights_Beam_Low implementation. */
    void setIsOn(bool IsOn) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Beam_Low implementation. */
    bool getIsOn() const override;

    /** Guards and forwards call to Vehicle_Body_Lights_Beam_Low implementation. */
    void setIsDefect(bool IsDefect) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Beam_Low implementation. */
    bool getIsDefect() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Beam_Low changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights_Beam_Low.
    */
    IVehicleBodyLightsBeamLowPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights_Beam_Low object which is guarded */
    std::shared_ptr<IVehicleBodyLightsBeamLow> m_impl;
    // Mutex for isOn property
    mutable std::shared_timed_mutex m_isOnMutex;
    // Mutex for isDefect property
    mutable std::shared_timed_mutex m_isDefectMutex;
};
} // namespace Vehicle
} // namespace Cpp
