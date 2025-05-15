

#include "vehicle/generated/core/vehiclepowertraintractionbattery.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryThreadSafeDecorator::VehiclePowertrainTractionBatteryThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBattery> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setId(const std::string& Id)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_idMutex);
    m_impl->setId(Id);
}

const std::string& VehiclePowertrainTractionBatteryThreadSafeDecorator::getId() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_idMutex);
    return m_impl->getId();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setProductionDate(const std::string& ProductionDate)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_productionDateMutex);
    m_impl->setProductionDate(ProductionDate);
}

const std::string& VehiclePowertrainTractionBatteryThreadSafeDecorator::getProductionDate() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_productionDateMutex);
    return m_impl->getProductionDate();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setErrorCodes(const std::list<std::string>& ErrorCodes)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_errorCodesMutex);
    m_impl->setErrorCodes(ErrorCodes);
}

const std::list<std::string>& VehiclePowertrainTractionBatteryThreadSafeDecorator::getErrorCodes() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_errorCodesMutex);
    return m_impl->getErrorCodes();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setIsPowerConnected(bool IsPowerConnected)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isPowerConnectedMutex);
    m_impl->setIsPowerConnected(IsPowerConnected);
}

bool VehiclePowertrainTractionBatteryThreadSafeDecorator::getIsPowerConnected() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isPowerConnectedMutex);
    return m_impl->getIsPowerConnected();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setIsGroundConnected(bool IsGroundConnected)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isGroundConnectedMutex);
    m_impl->setIsGroundConnected(IsGroundConnected);
}

bool VehiclePowertrainTractionBatteryThreadSafeDecorator::getIsGroundConnected() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isGroundConnectedMutex);
    return m_impl->getIsGroundConnected();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setGrossCapacity(int32_t GrossCapacity)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_grossCapacityMutex);
    m_impl->setGrossCapacity(GrossCapacity);
}

int32_t VehiclePowertrainTractionBatteryThreadSafeDecorator::getGrossCapacity() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_grossCapacityMutex);
    return m_impl->getGrossCapacity();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setNetCapacity(int32_t NetCapacity)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_netCapacityMutex);
    m_impl->setNetCapacity(NetCapacity);
}

int32_t VehiclePowertrainTractionBatteryThreadSafeDecorator::getNetCapacity() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_netCapacityMutex);
    return m_impl->getNetCapacity();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setStateOfHealth(float StateOfHealth)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_stateOfHealthMutex);
    m_impl->setStateOfHealth(StateOfHealth);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getStateOfHealth() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_stateOfHealthMutex);
    return m_impl->getStateOfHealth();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setNominalVoltage(int32_t NominalVoltage)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalVoltageMutex);
    m_impl->setNominalVoltage(NominalVoltage);
}

int32_t VehiclePowertrainTractionBatteryThreadSafeDecorator::getNominalVoltage() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_nominalVoltageMutex);
    return m_impl->getNominalVoltage();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setMaxVoltage(int32_t MaxVoltage)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxVoltageMutex);
    m_impl->setMaxVoltage(MaxVoltage);
}

int32_t VehiclePowertrainTractionBatteryThreadSafeDecorator::getMaxVoltage() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_maxVoltageMutex);
    return m_impl->getMaxVoltage();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setCurrentVoltage(float CurrentVoltage)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentVoltageMutex);
    m_impl->setCurrentVoltage(CurrentVoltage);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getCurrentVoltage() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_currentVoltageMutex);
    return m_impl->getCurrentVoltage();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setCurrentCurrent(float CurrentCurrent)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentCurrentMutex);
    m_impl->setCurrentCurrent(CurrentCurrent);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getCurrentCurrent() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_currentCurrentMutex);
    return m_impl->getCurrentCurrent();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setCurrentPower(float CurrentPower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentPowerMutex);
    m_impl->setCurrentPower(CurrentPower);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getCurrentPower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_currentPowerMutex);
    return m_impl->getCurrentPower();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setAccumulatedChargedEnergy(float AccumulatedChargedEnergy)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedChargedEnergyMutex);
    m_impl->setAccumulatedChargedEnergy(AccumulatedChargedEnergy);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getAccumulatedChargedEnergy() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_accumulatedChargedEnergyMutex);
    return m_impl->getAccumulatedChargedEnergy();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setAccumulatedConsumedEnergy(float AccumulatedConsumedEnergy)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedConsumedEnergyMutex);
    m_impl->setAccumulatedConsumedEnergy(AccumulatedConsumedEnergy);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getAccumulatedConsumedEnergy() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_accumulatedConsumedEnergyMutex);
    return m_impl->getAccumulatedConsumedEnergy();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setAccumulatedChargedThroughput(float AccumulatedChargedThroughput)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedChargedThroughputMutex);
    m_impl->setAccumulatedChargedThroughput(AccumulatedChargedThroughput);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getAccumulatedChargedThroughput() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_accumulatedChargedThroughputMutex);
    return m_impl->getAccumulatedChargedThroughput();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setAccumulatedConsumedThroughput(float AccumulatedConsumedThroughput)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedConsumedThroughputMutex);
    m_impl->setAccumulatedConsumedThroughput(AccumulatedConsumedThroughput);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getAccumulatedConsumedThroughput() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_accumulatedConsumedThroughputMutex);
    return m_impl->getAccumulatedConsumedThroughput();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setPowerLoss(float PowerLoss)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossMutex);
    m_impl->setPowerLoss(PowerLoss);
}

float VehiclePowertrainTractionBatteryThreadSafeDecorator::getPowerLoss() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_powerLossMutex);
    return m_impl->getPowerLoss();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setRange(int32_t Range)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rangeMutex);
    m_impl->setRange(Range);
}

int32_t VehiclePowertrainTractionBatteryThreadSafeDecorator::getRange() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rangeMutex);
    return m_impl->getRange();
}
void VehiclePowertrainTractionBatteryThreadSafeDecorator::setTimeRemaining(int32_t TimeRemaining)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeRemainingMutex);
    m_impl->setTimeRemaining(TimeRemaining);
}

int32_t VehiclePowertrainTractionBatteryThreadSafeDecorator::getTimeRemaining() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_timeRemainingMutex);
    return m_impl->getTimeRemaining();
}

IVehiclePowertrainTractionBatteryPublisher& VehiclePowertrainTractionBatteryThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}