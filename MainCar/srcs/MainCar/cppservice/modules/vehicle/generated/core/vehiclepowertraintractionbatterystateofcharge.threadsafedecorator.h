
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_StateOfCharge interface.
* @see Vehicle_Powertrain_TractionBattery_StateOfCharge
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryStateOfCharge> testVehiclePowertrainTractionBatteryStateOfCharge = std::make_unique<VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryStateOfCharge>());

// Thread safe access
auto current = testVehiclePowertrainTractionBatteryStateOfCharge->getCurrent();
testVehiclePowertrainTractionBatteryStateOfCharge->setCurrent(0.0f);
auto displayed = testVehiclePowertrainTractionBatteryStateOfCharge->getDisplayed();
testVehiclePowertrainTractionBatteryStateOfCharge->setDisplayed(0.0f);
auto currentEnergy = testVehiclePowertrainTractionBatteryStateOfCharge->getCurrentEnergy();
testVehiclePowertrainTractionBatteryStateOfCharge->setCurrentEnergy(0.0f);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator : public IVehiclePowertrainTractionBatteryStateOfCharge
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_StateOfCharge object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryStateOfCharge> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_StateOfCharge implementation. */
    void setCurrent(float Current) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_StateOfCharge implementation. */
    float getCurrent() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_StateOfCharge implementation. */
    void setDisplayed(float Displayed) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_StateOfCharge implementation. */
    float getDisplayed() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_StateOfCharge implementation. */
    void setCurrentEnergy(float CurrentEnergy) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_StateOfCharge implementation. */
    float getCurrentEnergy() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_StateOfCharge changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_StateOfCharge.
    */
    IVehiclePowertrainTractionBatteryStateOfChargePublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_StateOfCharge object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryStateOfCharge> m_impl;
    // Mutex for current property
    mutable std::shared_timed_mutex m_currentMutex;
    // Mutex for displayed property
    mutable std::shared_timed_mutex m_displayedMutex;
    // Mutex for currentEnergy property
    mutable std::shared_timed_mutex m_currentEnergyMutex;
};
} // namespace Vehicle
} // namespace Cpp
