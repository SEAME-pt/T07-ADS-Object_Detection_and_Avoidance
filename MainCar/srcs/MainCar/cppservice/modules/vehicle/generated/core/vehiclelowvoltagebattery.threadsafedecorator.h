
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleLowVoltageBatteryThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_LowVoltageBattery interface.
* @see Vehicle_LowVoltageBattery
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleLowVoltageBattery> testVehicleLowVoltageBattery = std::make_unique<VehicleLowVoltageBatteryThreadSafeDecorator>(std::make_shared<VehicleLowVoltageBattery>());

// Thread safe access
auto nominalVoltage = testVehicleLowVoltageBattery->getNominalVoltage();
testVehicleLowVoltageBattery->setNominalVoltage(0);
auto nominalCapacity = testVehicleLowVoltageBattery->getNominalCapacity();
testVehicleLowVoltageBattery->setNominalCapacity(0);
auto currentVoltage = testVehicleLowVoltageBattery->getCurrentVoltage();
testVehicleLowVoltageBattery->setCurrentVoltage(0.0f);
auto currentCurrent = testVehicleLowVoltageBattery->getCurrentCurrent();
testVehicleLowVoltageBattery->setCurrentCurrent(0.0f);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleLowVoltageBatteryThreadSafeDecorator : public IVehicleLowVoltageBattery
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_LowVoltageBattery object to make thread safe.
    */
    explicit VehicleLowVoltageBatteryThreadSafeDecorator(std::shared_ptr<IVehicleLowVoltageBattery> impl);

    /** Guards and forwards call to Vehicle_LowVoltageBattery implementation. */
    void setNominalVoltage(int32_t NominalVoltage) override;
    /** Guards and forwards call to Vehicle_LowVoltageBattery implementation. */
    int32_t getNominalVoltage() const override;

    /** Guards and forwards call to Vehicle_LowVoltageBattery implementation. */
    void setNominalCapacity(int32_t NominalCapacity) override;
    /** Guards and forwards call to Vehicle_LowVoltageBattery implementation. */
    int32_t getNominalCapacity() const override;

    /** Guards and forwards call to Vehicle_LowVoltageBattery implementation. */
    void setCurrentVoltage(float CurrentVoltage) override;
    /** Guards and forwards call to Vehicle_LowVoltageBattery implementation. */
    float getCurrentVoltage() const override;

    /** Guards and forwards call to Vehicle_LowVoltageBattery implementation. */
    void setCurrentCurrent(float CurrentCurrent) override;
    /** Guards and forwards call to Vehicle_LowVoltageBattery implementation. */
    float getCurrentCurrent() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_LowVoltageBattery changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_LowVoltageBattery.
    */
    IVehicleLowVoltageBatteryPublisher& _getPublisher() const override;
private:
    /** The Vehicle_LowVoltageBattery object which is guarded */
    std::shared_ptr<IVehicleLowVoltageBattery> m_impl;
    // Mutex for nominalVoltage property
    mutable std::shared_timed_mutex m_nominalVoltageMutex;
    // Mutex for nominalCapacity property
    mutable std::shared_timed_mutex m_nominalCapacityMutex;
    // Mutex for currentVoltage property
    mutable std::shared_timed_mutex m_currentVoltageMutex;
    // Mutex for currentCurrent property
    mutable std::shared_timed_mutex m_currentCurrentMutex;
};
} // namespace Vehicle
} // namespace Cpp
