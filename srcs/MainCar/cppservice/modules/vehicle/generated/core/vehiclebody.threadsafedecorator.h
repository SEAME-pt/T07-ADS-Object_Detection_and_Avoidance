
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body interface.
* @see Vehicle_Body
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBody> testVehicleBody = std::make_unique<VehicleBodyThreadSafeDecorator>(std::make_shared<VehicleBody>());

// Thread safe access
auto powerOptimizeLevel = testVehicleBody->getPowerOptimizeLevel();
testVehicleBody->setPowerOptimizeLevel(0);
auto isAutoPowerOptimize = testVehicleBody->getIsAutoPowerOptimize();
testVehicleBody->setIsAutoPowerOptimize(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyThreadSafeDecorator : public IVehicleBody
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body object to make thread safe.
    */
    explicit VehicleBodyThreadSafeDecorator(std::shared_ptr<IVehicleBody> impl);

    /** Guards and forwards call to Vehicle_Body implementation. */
    void setPowerOptimizeLevel(int32_t PowerOptimizeLevel) override;
    /** Guards and forwards call to Vehicle_Body implementation. */
    int32_t getPowerOptimizeLevel() const override;

    /** Guards and forwards call to Vehicle_Body implementation. */
    void setIsAutoPowerOptimize(bool IsAutoPowerOptimize) override;
    /** Guards and forwards call to Vehicle_Body implementation. */
    bool getIsAutoPowerOptimize() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body.
    */
    IVehicleBodyPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body object which is guarded */
    std::shared_ptr<IVehicleBody> m_impl;
    // Mutex for powerOptimizeLevel property
    mutable std::shared_timed_mutex m_powerOptimizeLevelMutex;
    // Mutex for isAutoPowerOptimize property
    mutable std::shared_timed_mutex m_isAutoPowerOptimizeMutex;
};
} // namespace Vehicle
} // namespace Cpp
