
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery interface.
* @see Vehicle_Powertrain_TractionBattery
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBattery> testVehiclePowertrainTractionBattery = std::make_unique<VehiclePowertrainTractionBatteryThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBattery>());

// Thread safe access
auto id = testVehiclePowertrainTractionBattery->getId();
testVehiclePowertrainTractionBattery->setId(std::string());
auto productionDate = testVehiclePowertrainTractionBattery->getProductionDate();
testVehiclePowertrainTractionBattery->setProductionDate(std::string());
auto errorCodes = testVehiclePowertrainTractionBattery->getErrorCodes();
testVehiclePowertrainTractionBattery->setErrorCodes(std::list<std::string>());
auto isPowerConnected = testVehiclePowertrainTractionBattery->getIsPowerConnected();
testVehiclePowertrainTractionBattery->setIsPowerConnected(false);
auto isGroundConnected = testVehiclePowertrainTractionBattery->getIsGroundConnected();
testVehiclePowertrainTractionBattery->setIsGroundConnected(false);
auto grossCapacity = testVehiclePowertrainTractionBattery->getGrossCapacity();
testVehiclePowertrainTractionBattery->setGrossCapacity(0);
auto netCapacity = testVehiclePowertrainTractionBattery->getNetCapacity();
testVehiclePowertrainTractionBattery->setNetCapacity(0);
auto stateOfHealth = testVehiclePowertrainTractionBattery->getStateOfHealth();
testVehiclePowertrainTractionBattery->setStateOfHealth(0.0f);
auto nominalVoltage = testVehiclePowertrainTractionBattery->getNominalVoltage();
testVehiclePowertrainTractionBattery->setNominalVoltage(0);
auto maxVoltage = testVehiclePowertrainTractionBattery->getMaxVoltage();
testVehiclePowertrainTractionBattery->setMaxVoltage(0);
auto currentVoltage = testVehiclePowertrainTractionBattery->getCurrentVoltage();
testVehiclePowertrainTractionBattery->setCurrentVoltage(0.0f);
auto currentCurrent = testVehiclePowertrainTractionBattery->getCurrentCurrent();
testVehiclePowertrainTractionBattery->setCurrentCurrent(0.0f);
auto currentPower = testVehiclePowertrainTractionBattery->getCurrentPower();
testVehiclePowertrainTractionBattery->setCurrentPower(0.0f);
auto accumulatedChargedEnergy = testVehiclePowertrainTractionBattery->getAccumulatedChargedEnergy();
testVehiclePowertrainTractionBattery->setAccumulatedChargedEnergy(0.0f);
auto accumulatedConsumedEnergy = testVehiclePowertrainTractionBattery->getAccumulatedConsumedEnergy();
testVehiclePowertrainTractionBattery->setAccumulatedConsumedEnergy(0.0f);
auto accumulatedChargedThroughput = testVehiclePowertrainTractionBattery->getAccumulatedChargedThroughput();
testVehiclePowertrainTractionBattery->setAccumulatedChargedThroughput(0.0f);
auto accumulatedConsumedThroughput = testVehiclePowertrainTractionBattery->getAccumulatedConsumedThroughput();
testVehiclePowertrainTractionBattery->setAccumulatedConsumedThroughput(0.0f);
auto powerLoss = testVehiclePowertrainTractionBattery->getPowerLoss();
testVehiclePowertrainTractionBattery->setPowerLoss(0.0f);
auto range = testVehiclePowertrainTractionBattery->getRange();
testVehiclePowertrainTractionBattery->setRange(0);
auto timeRemaining = testVehiclePowertrainTractionBattery->getTimeRemaining();
testVehiclePowertrainTractionBattery->setTimeRemaining(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryThreadSafeDecorator : public IVehiclePowertrainTractionBattery
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBattery> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setId(const std::string& Id) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    const std::string& getId() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setProductionDate(const std::string& ProductionDate) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    const std::string& getProductionDate() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setErrorCodes(const std::list<std::string>& ErrorCodes) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    const std::list<std::string>& getErrorCodes() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setIsPowerConnected(bool IsPowerConnected) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    bool getIsPowerConnected() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setIsGroundConnected(bool IsGroundConnected) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    bool getIsGroundConnected() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setGrossCapacity(int32_t GrossCapacity) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    int32_t getGrossCapacity() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setNetCapacity(int32_t NetCapacity) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    int32_t getNetCapacity() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setStateOfHealth(float StateOfHealth) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getStateOfHealth() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setNominalVoltage(int32_t NominalVoltage) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    int32_t getNominalVoltage() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setMaxVoltage(int32_t MaxVoltage) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    int32_t getMaxVoltage() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setCurrentVoltage(float CurrentVoltage) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getCurrentVoltage() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setCurrentCurrent(float CurrentCurrent) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getCurrentCurrent() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setCurrentPower(float CurrentPower) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getCurrentPower() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setAccumulatedChargedEnergy(float AccumulatedChargedEnergy) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getAccumulatedChargedEnergy() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setAccumulatedConsumedEnergy(float AccumulatedConsumedEnergy) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getAccumulatedConsumedEnergy() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setAccumulatedChargedThroughput(float AccumulatedChargedThroughput) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getAccumulatedChargedThroughput() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setAccumulatedConsumedThroughput(float AccumulatedConsumedThroughput) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getAccumulatedConsumedThroughput() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setPowerLoss(float PowerLoss) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    float getPowerLoss() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setRange(int32_t Range) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    int32_t getRange() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    void setTimeRemaining(int32_t TimeRemaining) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery implementation. */
    int32_t getTimeRemaining() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery.
    */
    IVehiclePowertrainTractionBatteryPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBattery> m_impl;
    // Mutex for id property
    mutable std::shared_timed_mutex m_idMutex;
    // Mutex for productionDate property
    mutable std::shared_timed_mutex m_productionDateMutex;
    // Mutex for errorCodes property
    mutable std::shared_timed_mutex m_errorCodesMutex;
    // Mutex for isPowerConnected property
    mutable std::shared_timed_mutex m_isPowerConnectedMutex;
    // Mutex for isGroundConnected property
    mutable std::shared_timed_mutex m_isGroundConnectedMutex;
    // Mutex for grossCapacity property
    mutable std::shared_timed_mutex m_grossCapacityMutex;
    // Mutex for netCapacity property
    mutable std::shared_timed_mutex m_netCapacityMutex;
    // Mutex for stateOfHealth property
    mutable std::shared_timed_mutex m_stateOfHealthMutex;
    // Mutex for nominalVoltage property
    mutable std::shared_timed_mutex m_nominalVoltageMutex;
    // Mutex for maxVoltage property
    mutable std::shared_timed_mutex m_maxVoltageMutex;
    // Mutex for currentVoltage property
    mutable std::shared_timed_mutex m_currentVoltageMutex;
    // Mutex for currentCurrent property
    mutable std::shared_timed_mutex m_currentCurrentMutex;
    // Mutex for currentPower property
    mutable std::shared_timed_mutex m_currentPowerMutex;
    // Mutex for accumulatedChargedEnergy property
    mutable std::shared_timed_mutex m_accumulatedChargedEnergyMutex;
    // Mutex for accumulatedConsumedEnergy property
    mutable std::shared_timed_mutex m_accumulatedConsumedEnergyMutex;
    // Mutex for accumulatedChargedThroughput property
    mutable std::shared_timed_mutex m_accumulatedChargedThroughputMutex;
    // Mutex for accumulatedConsumedThroughput property
    mutable std::shared_timed_mutex m_accumulatedConsumedThroughputMutex;
    // Mutex for powerLoss property
    mutable std::shared_timed_mutex m_powerLossMutex;
    // Mutex for range property
    mutable std::shared_timed_mutex m_rangeMutex;
    // Mutex for timeRemaining property
    mutable std::shared_timed_mutex m_timeRemainingMutex;
};
} // namespace Vehicle
} // namespace Cpp
