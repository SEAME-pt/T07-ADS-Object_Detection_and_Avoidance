

#include "vehicle/generated/core/vehiclepowertrain.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainThreadSafeDecorator::VehiclePowertrainThreadSafeDecorator(std::shared_ptr<IVehiclePowertrain> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainThreadSafeDecorator::setAccumulatedBrakingEnergy(float AccumulatedBrakingEnergy)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedBrakingEnergyMutex);
    m_impl->setAccumulatedBrakingEnergy(AccumulatedBrakingEnergy);
}

float VehiclePowertrainThreadSafeDecorator::getAccumulatedBrakingEnergy() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_accumulatedBrakingEnergyMutex);
    return m_impl->getAccumulatedBrakingEnergy();
}
void VehiclePowertrainThreadSafeDecorator::setRange(int32_t Range)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rangeMutex);
    m_impl->setRange(Range);
}

int32_t VehiclePowertrainThreadSafeDecorator::getRange() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rangeMutex);
    return m_impl->getRange();
}
void VehiclePowertrainThreadSafeDecorator::setTimeRemaining(int32_t TimeRemaining)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeRemainingMutex);
    m_impl->setTimeRemaining(TimeRemaining);
}

int32_t VehiclePowertrainThreadSafeDecorator::getTimeRemaining() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_timeRemainingMutex);
    return m_impl->getTimeRemaining();
}
void VehiclePowertrainThreadSafeDecorator::setType(Vehicle_Powertrain_Type_ValueEnum Type)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_typeMutex);
    m_impl->setType(Type);
}

Vehicle_Powertrain_Type_ValueEnum VehiclePowertrainThreadSafeDecorator::getType() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_typeMutex);
    return m_impl->getType();
}
void VehiclePowertrainThreadSafeDecorator::setPowerOptimizeLevel(int32_t PowerOptimizeLevel)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelMutex);
    m_impl->setPowerOptimizeLevel(PowerOptimizeLevel);
}

int32_t VehiclePowertrainThreadSafeDecorator::getPowerOptimizeLevel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelMutex);
    return m_impl->getPowerOptimizeLevel();
}
void VehiclePowertrainThreadSafeDecorator::setIsAutoPowerOptimize(bool IsAutoPowerOptimize)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeMutex);
    m_impl->setIsAutoPowerOptimize(IsAutoPowerOptimize);
}

bool VehiclePowertrainThreadSafeDecorator::getIsAutoPowerOptimize() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeMutex);
    return m_impl->getIsAutoPowerOptimize();
}

IVehiclePowertrainPublisher& VehiclePowertrainThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}