
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleAngularVelocityThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_AngularVelocity interface.
* @see Vehicle_AngularVelocity
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleAngularVelocity> testVehicleAngularVelocity = std::make_unique<VehicleAngularVelocityThreadSafeDecorator>(std::make_shared<VehicleAngularVelocity>());

// Thread safe access
auto roll = testVehicleAngularVelocity->getRoll();
testVehicleAngularVelocity->setRoll(0.0f);
auto pitch = testVehicleAngularVelocity->getPitch();
testVehicleAngularVelocity->setPitch(0.0f);
auto yaw = testVehicleAngularVelocity->getYaw();
testVehicleAngularVelocity->setYaw(0.0f);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleAngularVelocityThreadSafeDecorator : public IVehicleAngularVelocity
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_AngularVelocity object to make thread safe.
    */
    explicit VehicleAngularVelocityThreadSafeDecorator(std::shared_ptr<IVehicleAngularVelocity> impl);

    /** Guards and forwards call to Vehicle_AngularVelocity implementation. */
    void setRoll(float Roll) override;
    /** Guards and forwards call to Vehicle_AngularVelocity implementation. */
    float getRoll() const override;

    /** Guards and forwards call to Vehicle_AngularVelocity implementation. */
    void setPitch(float Pitch) override;
    /** Guards and forwards call to Vehicle_AngularVelocity implementation. */
    float getPitch() const override;

    /** Guards and forwards call to Vehicle_AngularVelocity implementation. */
    void setYaw(float Yaw) override;
    /** Guards and forwards call to Vehicle_AngularVelocity implementation. */
    float getYaw() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_AngularVelocity changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_AngularVelocity.
    */
    IVehicleAngularVelocityPublisher& _getPublisher() const override;
private:
    /** The Vehicle_AngularVelocity object which is guarded */
    std::shared_ptr<IVehicleAngularVelocity> m_impl;
    // Mutex for roll property
    mutable std::shared_timed_mutex m_rollMutex;
    // Mutex for pitch property
    mutable std::shared_timed_mutex m_pitchMutex;
    // Mutex for yaw property
    mutable std::shared_timed_mutex m_yawMutex;
};
} // namespace Vehicle
} // namespace Cpp
