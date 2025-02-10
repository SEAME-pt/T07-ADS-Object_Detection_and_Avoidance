
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsRunningThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights_Running interface.
* @see Vehicle_Body_Lights_Running
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLightsRunning> testVehicleBodyLightsRunning = std::make_unique<VehicleBodyLightsRunningThreadSafeDecorator>(std::make_shared<VehicleBodyLightsRunning>());

// Thread safe access
auto isOn = testVehicleBodyLightsRunning->getIsOn();
testVehicleBodyLightsRunning->setIsOn(false);
auto isDefect = testVehicleBodyLightsRunning->getIsDefect();
testVehicleBodyLightsRunning->setIsDefect(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsRunningThreadSafeDecorator : public IVehicleBodyLightsRunning
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights_Running object to make thread safe.
    */
    explicit VehicleBodyLightsRunningThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsRunning> impl);

    /** Guards and forwards call to Vehicle_Body_Lights_Running implementation. */
    void setIsOn(bool IsOn) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Running implementation. */
    bool getIsOn() const override;

    /** Guards and forwards call to Vehicle_Body_Lights_Running implementation. */
    void setIsDefect(bool IsDefect) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Running implementation. */
    bool getIsDefect() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Running changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights_Running.
    */
    IVehicleBodyLightsRunningPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights_Running object which is guarded */
    std::shared_ptr<IVehicleBodyLightsRunning> m_impl;
    // Mutex for isOn property
    mutable std::shared_timed_mutex m_isOnMutex;
    // Mutex for isDefect property
    mutable std::shared_timed_mutex m_isDefectMutex;
};
} // namespace Vehicle
} // namespace Cpp
