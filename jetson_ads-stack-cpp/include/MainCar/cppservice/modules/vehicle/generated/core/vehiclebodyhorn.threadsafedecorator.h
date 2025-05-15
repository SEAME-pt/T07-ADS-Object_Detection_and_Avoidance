
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyHornThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Horn interface.
* @see Vehicle_Body_Horn
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyHorn> testVehicleBodyHorn = std::make_unique<VehicleBodyHornThreadSafeDecorator>(std::make_shared<VehicleBodyHorn>());

// Thread safe access
auto isActive = testVehicleBodyHorn->getIsActive();
testVehicleBodyHorn->setIsActive(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyHornThreadSafeDecorator : public IVehicleBodyHorn
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Horn object to make thread safe.
    */
    explicit VehicleBodyHornThreadSafeDecorator(std::shared_ptr<IVehicleBodyHorn> impl);

    /** Guards and forwards call to Vehicle_Body_Horn implementation. */
    void setIsActive(bool IsActive) override;
    /** Guards and forwards call to Vehicle_Body_Horn implementation. */
    bool getIsActive() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Horn changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Horn.
    */
    IVehicleBodyHornPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Horn object which is guarded */
    std::shared_ptr<IVehicleBodyHorn> m_impl;
    // Mutex for isActive property
    mutable std::shared_timed_mutex m_isActiveMutex;
};
} // namespace Vehicle
} // namespace Cpp
