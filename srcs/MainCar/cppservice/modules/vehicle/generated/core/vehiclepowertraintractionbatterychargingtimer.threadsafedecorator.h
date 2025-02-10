
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_Charging_Timer interface.
* @see Vehicle_Powertrain_TractionBattery_Charging_Timer
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryChargingTimer> testVehiclePowertrainTractionBatteryChargingTimer = std::make_unique<VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryChargingTimer>());

// Thread safe access
auto mode = testVehiclePowertrainTractionBatteryChargingTimer->getMode();
testVehiclePowertrainTractionBatteryChargingTimer->setMode(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum::INACTIVE);
auto time = testVehiclePowertrainTractionBatteryChargingTimer->getTime();
testVehiclePowertrainTractionBatteryChargingTimer->setTime(std::string());
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator : public IVehiclePowertrainTractionBatteryChargingTimer
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_Charging_Timer object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryChargingTimerThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingTimer> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Timer implementation. */
    void setMode(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Timer implementation. */
    Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum getMode() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Timer implementation. */
    void setTime(const std::string& Time) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Timer implementation. */
    const std::string& getTime() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_Timer changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_Timer.
    */
    IVehiclePowertrainTractionBatteryChargingTimerPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_Charging_Timer object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryChargingTimer> m_impl;
    // Mutex for mode property
    mutable std::shared_timed_mutex m_modeMutex;
    // Mutex for time property
    mutable std::shared_timed_mutex m_timeMutex;
};
} // namespace Vehicle
} // namespace Cpp
