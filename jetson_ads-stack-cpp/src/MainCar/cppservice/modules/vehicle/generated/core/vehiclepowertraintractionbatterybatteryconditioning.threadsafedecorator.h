
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_BatteryConditioning interface.
* @see Vehicle_Powertrain_TractionBattery_BatteryConditioning
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryBatteryConditioning> testVehiclePowertrainTractionBatteryBatteryConditioning = std::make_unique<VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryBatteryConditioning>());

// Thread safe access
auto isActive = testVehiclePowertrainTractionBatteryBatteryConditioning->getIsActive();
testVehiclePowertrainTractionBatteryBatteryConditioning->setIsActive(false);
auto isOngoing = testVehiclePowertrainTractionBatteryBatteryConditioning->getIsOngoing();
testVehiclePowertrainTractionBatteryBatteryConditioning->setIsOngoing(false);
auto requestedMode = testVehiclePowertrainTractionBatteryBatteryConditioning->getRequestedMode();
testVehiclePowertrainTractionBatteryBatteryConditioning->setRequestedMode(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum::INACTIVE);
auto startTime = testVehiclePowertrainTractionBatteryBatteryConditioning->getStartTime();
testVehiclePowertrainTractionBatteryBatteryConditioning->setStartTime(std::string());
auto targetTime = testVehiclePowertrainTractionBatteryBatteryConditioning->getTargetTime();
testVehiclePowertrainTractionBatteryBatteryConditioning->setTargetTime(std::string());
auto targetTemperature = testVehiclePowertrainTractionBatteryBatteryConditioning->getTargetTemperature();
testVehiclePowertrainTractionBatteryBatteryConditioning->setTargetTemperature(0.0f);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator : public IVehiclePowertrainTractionBatteryBatteryConditioning
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_BatteryConditioning object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryBatteryConditioningThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryBatteryConditioning> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    void setIsActive(bool IsActive) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    bool getIsActive() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    void setIsOngoing(bool IsOngoing) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    bool getIsOngoing() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    void setRequestedMode(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum getRequestedMode() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    void setStartTime(const std::string& StartTime) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    const std::string& getStartTime() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    void setTargetTime(const std::string& TargetTime) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    const std::string& getTargetTime() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    void setTargetTemperature(float TargetTemperature) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation. */
    float getTargetTemperature() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_BatteryConditioning changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_BatteryConditioning.
    */
    IVehiclePowertrainTractionBatteryBatteryConditioningPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_BatteryConditioning object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryBatteryConditioning> m_impl;
    // Mutex for isActive property
    mutable std::shared_timed_mutex m_isActiveMutex;
    // Mutex for isOngoing property
    mutable std::shared_timed_mutex m_isOngoingMutex;
    // Mutex for requestedMode property
    mutable std::shared_timed_mutex m_requestedModeMutex;
    // Mutex for startTime property
    mutable std::shared_timed_mutex m_startTimeMutex;
    // Mutex for targetTime property
    mutable std::shared_timed_mutex m_targetTimeMutex;
    // Mutex for targetTemperature property
    mutable std::shared_timed_mutex m_targetTemperatureMutex;
};
} // namespace Vehicle
} // namespace Cpp
