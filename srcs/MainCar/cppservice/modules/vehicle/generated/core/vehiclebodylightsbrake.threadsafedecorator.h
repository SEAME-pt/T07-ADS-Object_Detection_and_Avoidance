
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsBrakeThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights_Brake interface.
* @see Vehicle_Body_Lights_Brake
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLightsBrake> testVehicleBodyLightsBrake = std::make_unique<VehicleBodyLightsBrakeThreadSafeDecorator>(std::make_shared<VehicleBodyLightsBrake>());

// Thread safe access
auto isActive = testVehicleBodyLightsBrake->getIsActive();
testVehicleBodyLightsBrake->setIsActive(Vehicle_Body_Lights_Brake_IsActive_ValueEnum::INACTIVE);
auto isDefect = testVehicleBodyLightsBrake->getIsDefect();
testVehicleBodyLightsBrake->setIsDefect(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsBrakeThreadSafeDecorator : public IVehicleBodyLightsBrake
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights_Brake object to make thread safe.
    */
    explicit VehicleBodyLightsBrakeThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsBrake> impl);

    /** Guards and forwards call to Vehicle_Body_Lights_Brake implementation. */
    void setIsActive(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Brake implementation. */
    Vehicle_Body_Lights_Brake_IsActive_ValueEnum getIsActive() const override;

    /** Guards and forwards call to Vehicle_Body_Lights_Brake implementation. */
    void setIsDefect(bool IsDefect) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Brake implementation. */
    bool getIsDefect() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Brake changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights_Brake.
    */
    IVehicleBodyLightsBrakePublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights_Brake object which is guarded */
    std::shared_ptr<IVehicleBodyLightsBrake> m_impl;
    // Mutex for isActive property
    mutable std::shared_timed_mutex m_isActiveMutex;
    // Mutex for isDefect property
    mutable std::shared_timed_mutex m_isDefectMutex;
};
} // namespace Vehicle
} // namespace Cpp
