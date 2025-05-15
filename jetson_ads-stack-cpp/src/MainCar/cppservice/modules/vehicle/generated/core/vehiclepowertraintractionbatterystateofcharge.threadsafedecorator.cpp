

#include "vehicle/generated/core/vehiclepowertraintractionbatterystateofcharge.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator::VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryStateOfCharge> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator::setCurrent(float Current)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentMutex);
    m_impl->setCurrent(Current);
}

float VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator::getCurrent() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_currentMutex);
    return m_impl->getCurrent();
}
void VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator::setDisplayed(float Displayed)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_displayedMutex);
    m_impl->setDisplayed(Displayed);
}

float VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator::getDisplayed() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_displayedMutex);
    return m_impl->getDisplayed();
}
void VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator::setCurrentEnergy(float CurrentEnergy)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentEnergyMutex);
    m_impl->setCurrentEnergy(CurrentEnergy);
}

float VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator::getCurrentEnergy() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_currentEnergyMutex);
    return m_impl->getCurrentEnergy();
}

IVehiclePowertrainTractionBatteryStateOfChargePublisher& VehiclePowertrainTractionBatteryStateOfChargeThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}