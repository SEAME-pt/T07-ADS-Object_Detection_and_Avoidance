

#include "vehicle/implementation/vehiclepowertraintractionbattery.h"
#include "vehicle/generated/core/vehiclepowertraintractionbattery.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbattery.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBattery::VehiclePowertrainTractionBattery()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryPublisher>())
{
}
VehiclePowertrainTractionBattery::~VehiclePowertrainTractionBattery()
{
}

void VehiclePowertrainTractionBattery::setId(const std::string& Id)
{
    if (m_data.m_Id != Id) {
        m_data.m_Id = Id;
        m_publisher->publishIdChanged(Id);
    }
}

const std::string& VehiclePowertrainTractionBattery::getId() const
{
    return m_data.m_Id;
}

void VehiclePowertrainTractionBattery::setProductionDate(const std::string& ProductionDate)
{
    if (m_data.m_ProductionDate != ProductionDate) {
        m_data.m_ProductionDate = ProductionDate;
        m_publisher->publishProductionDateChanged(ProductionDate);
    }
}

const std::string& VehiclePowertrainTractionBattery::getProductionDate() const
{
    return m_data.m_ProductionDate;
}

void VehiclePowertrainTractionBattery::setErrorCodes(const std::list<std::string>& ErrorCodes)
{
    if (m_data.m_ErrorCodes != ErrorCodes) {
        m_data.m_ErrorCodes = ErrorCodes;
        m_publisher->publishErrorCodesChanged(ErrorCodes);
    }
}

const std::list<std::string>& VehiclePowertrainTractionBattery::getErrorCodes() const
{
    return m_data.m_ErrorCodes;
}

void VehiclePowertrainTractionBattery::setIsPowerConnected(bool IsPowerConnected)
{
    if (m_data.m_IsPowerConnected != IsPowerConnected) {
        m_data.m_IsPowerConnected = IsPowerConnected;
        m_publisher->publishIsPowerConnectedChanged(IsPowerConnected);
    }
}

bool VehiclePowertrainTractionBattery::getIsPowerConnected() const
{
    return m_data.m_IsPowerConnected;
}

void VehiclePowertrainTractionBattery::setIsGroundConnected(bool IsGroundConnected)
{
    if (m_data.m_IsGroundConnected != IsGroundConnected) {
        m_data.m_IsGroundConnected = IsGroundConnected;
        m_publisher->publishIsGroundConnectedChanged(IsGroundConnected);
    }
}

bool VehiclePowertrainTractionBattery::getIsGroundConnected() const
{
    return m_data.m_IsGroundConnected;
}

void VehiclePowertrainTractionBattery::setGrossCapacity(int32_t GrossCapacity)
{
    if (m_data.m_GrossCapacity != GrossCapacity) {
        m_data.m_GrossCapacity = GrossCapacity;
        m_publisher->publishGrossCapacityChanged(GrossCapacity);
    }
}

int32_t VehiclePowertrainTractionBattery::getGrossCapacity() const
{
    return m_data.m_GrossCapacity;
}

void VehiclePowertrainTractionBattery::setNetCapacity(int32_t NetCapacity)
{
    if (m_data.m_NetCapacity != NetCapacity) {
        m_data.m_NetCapacity = NetCapacity;
        m_publisher->publishNetCapacityChanged(NetCapacity);
    }
}

int32_t VehiclePowertrainTractionBattery::getNetCapacity() const
{
    return m_data.m_NetCapacity;
}

void VehiclePowertrainTractionBattery::setStateOfHealth(float StateOfHealth)
{
    if (m_data.m_StateOfHealth != StateOfHealth) {
        m_data.m_StateOfHealth = StateOfHealth;
        m_publisher->publishStateOfHealthChanged(StateOfHealth);
    }
}

float VehiclePowertrainTractionBattery::getStateOfHealth() const
{
    return m_data.m_StateOfHealth;
}

void VehiclePowertrainTractionBattery::setNominalVoltage(int32_t NominalVoltage)
{
    if (m_data.m_NominalVoltage != NominalVoltage) {
        m_data.m_NominalVoltage = NominalVoltage;
        m_publisher->publishNominalVoltageChanged(NominalVoltage);
    }
}

int32_t VehiclePowertrainTractionBattery::getNominalVoltage() const
{
    return m_data.m_NominalVoltage;
}

void VehiclePowertrainTractionBattery::setMaxVoltage(int32_t MaxVoltage)
{
    if (m_data.m_MaxVoltage != MaxVoltage) {
        m_data.m_MaxVoltage = MaxVoltage;
        m_publisher->publishMaxVoltageChanged(MaxVoltage);
    }
}

int32_t VehiclePowertrainTractionBattery::getMaxVoltage() const
{
    return m_data.m_MaxVoltage;
}

void VehiclePowertrainTractionBattery::setCurrentVoltage(float CurrentVoltage)
{
    if (m_data.m_CurrentVoltage != CurrentVoltage) {
        m_data.m_CurrentVoltage = CurrentVoltage;
        m_publisher->publishCurrentVoltageChanged(CurrentVoltage);
    }
}

float VehiclePowertrainTractionBattery::getCurrentVoltage() const
{
    return m_data.m_CurrentVoltage;
}

void VehiclePowertrainTractionBattery::setCurrentCurrent(float CurrentCurrent)
{
    if (m_data.m_CurrentCurrent != CurrentCurrent) {
        m_data.m_CurrentCurrent = CurrentCurrent;
        m_publisher->publishCurrentCurrentChanged(CurrentCurrent);
    }
}

float VehiclePowertrainTractionBattery::getCurrentCurrent() const
{
    return m_data.m_CurrentCurrent;
}

void VehiclePowertrainTractionBattery::setCurrentPower(float CurrentPower)
{
    if (m_data.m_CurrentPower != CurrentPower) {
        m_data.m_CurrentPower = CurrentPower;
        m_publisher->publishCurrentPowerChanged(CurrentPower);
    }
}

float VehiclePowertrainTractionBattery::getCurrentPower() const
{
    return m_data.m_CurrentPower;
}

void VehiclePowertrainTractionBattery::setAccumulatedChargedEnergy(float AccumulatedChargedEnergy)
{
    if (m_data.m_AccumulatedChargedEnergy != AccumulatedChargedEnergy) {
        m_data.m_AccumulatedChargedEnergy = AccumulatedChargedEnergy;
        m_publisher->publishAccumulatedChargedEnergyChanged(AccumulatedChargedEnergy);
    }
}

float VehiclePowertrainTractionBattery::getAccumulatedChargedEnergy() const
{
    return m_data.m_AccumulatedChargedEnergy;
}

void VehiclePowertrainTractionBattery::setAccumulatedConsumedEnergy(float AccumulatedConsumedEnergy)
{
    if (m_data.m_AccumulatedConsumedEnergy != AccumulatedConsumedEnergy) {
        m_data.m_AccumulatedConsumedEnergy = AccumulatedConsumedEnergy;
        m_publisher->publishAccumulatedConsumedEnergyChanged(AccumulatedConsumedEnergy);
    }
}

float VehiclePowertrainTractionBattery::getAccumulatedConsumedEnergy() const
{
    return m_data.m_AccumulatedConsumedEnergy;
}

void VehiclePowertrainTractionBattery::setAccumulatedChargedThroughput(float AccumulatedChargedThroughput)
{
    if (m_data.m_AccumulatedChargedThroughput != AccumulatedChargedThroughput) {
        m_data.m_AccumulatedChargedThroughput = AccumulatedChargedThroughput;
        m_publisher->publishAccumulatedChargedThroughputChanged(AccumulatedChargedThroughput);
    }
}

float VehiclePowertrainTractionBattery::getAccumulatedChargedThroughput() const
{
    return m_data.m_AccumulatedChargedThroughput;
}

void VehiclePowertrainTractionBattery::setAccumulatedConsumedThroughput(float AccumulatedConsumedThroughput)
{
    if (m_data.m_AccumulatedConsumedThroughput != AccumulatedConsumedThroughput) {
        m_data.m_AccumulatedConsumedThroughput = AccumulatedConsumedThroughput;
        m_publisher->publishAccumulatedConsumedThroughputChanged(AccumulatedConsumedThroughput);
    }
}

float VehiclePowertrainTractionBattery::getAccumulatedConsumedThroughput() const
{
    return m_data.m_AccumulatedConsumedThroughput;
}

void VehiclePowertrainTractionBattery::setPowerLoss(float PowerLoss)
{
    if (m_data.m_PowerLoss != PowerLoss) {
        m_data.m_PowerLoss = PowerLoss;
        m_publisher->publishPowerLossChanged(PowerLoss);
    }
}

float VehiclePowertrainTractionBattery::getPowerLoss() const
{
    return m_data.m_PowerLoss;
}

void VehiclePowertrainTractionBattery::setRange(int32_t Range)
{
    if (m_data.m_Range != Range) {
        m_data.m_Range = Range;
        m_publisher->publishRangeChanged(Range);
    }
}

int32_t VehiclePowertrainTractionBattery::getRange() const
{
    return m_data.m_Range;
}

void VehiclePowertrainTractionBattery::setTimeRemaining(int32_t TimeRemaining)
{
    if (m_data.m_TimeRemaining != TimeRemaining) {
        m_data.m_TimeRemaining = TimeRemaining;
        m_publisher->publishTimeRemainingChanged(TimeRemaining);
    }
}

int32_t VehiclePowertrainTractionBattery::getTimeRemaining() const
{
    return m_data.m_TimeRemaining;
}

IVehiclePowertrainTractionBatteryPublisher& VehiclePowertrainTractionBattery::_getPublisher() const
{
    return *m_publisher;
}
