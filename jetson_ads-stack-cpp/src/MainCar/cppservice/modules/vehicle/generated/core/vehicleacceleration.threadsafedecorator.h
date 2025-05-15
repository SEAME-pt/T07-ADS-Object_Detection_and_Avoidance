
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleAccelerationThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Acceleration interface.
* @see Vehicle_Acceleration
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleAcceleration> testVehicleAcceleration = std::make_unique<VehicleAccelerationThreadSafeDecorator>(std::make_shared<VehicleAcceleration>());

// Thread safe access
auto longitudinal = testVehicleAcceleration->getLongitudinal();
testVehicleAcceleration->setLongitudinal(0.0f);
auto lateral = testVehicleAcceleration->getLateral();
testVehicleAcceleration->setLateral(0.0f);
auto vertical = testVehicleAcceleration->getVertical();
testVehicleAcceleration->setVertical(0.0f);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleAccelerationThreadSafeDecorator : public IVehicleAcceleration
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Acceleration object to make thread safe.
    */
    explicit VehicleAccelerationThreadSafeDecorator(std::shared_ptr<IVehicleAcceleration> impl);

    /** Guards and forwards call to Vehicle_Acceleration implementation. */
    void setLongitudinal(float Longitudinal) override;
    /** Guards and forwards call to Vehicle_Acceleration implementation. */
    float getLongitudinal() const override;

    /** Guards and forwards call to Vehicle_Acceleration implementation. */
    void setLateral(float Lateral) override;
    /** Guards and forwards call to Vehicle_Acceleration implementation. */
    float getLateral() const override;

    /** Guards and forwards call to Vehicle_Acceleration implementation. */
    void setVertical(float Vertical) override;
    /** Guards and forwards call to Vehicle_Acceleration implementation. */
    float getVertical() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Acceleration changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Acceleration.
    */
    IVehicleAccelerationPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Acceleration object which is guarded */
    std::shared_ptr<IVehicleAcceleration> m_impl;
    // Mutex for longitudinal property
    mutable std::shared_timed_mutex m_longitudinalMutex;
    // Mutex for lateral property
    mutable std::shared_timed_mutex m_lateralMutex;
    // Mutex for vertical property
    mutable std::shared_timed_mutex m_verticalMutex;
};
} // namespace Vehicle
} // namespace Cpp
