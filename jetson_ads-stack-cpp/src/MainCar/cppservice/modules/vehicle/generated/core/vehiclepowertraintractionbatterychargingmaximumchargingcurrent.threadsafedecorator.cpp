

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::setDc(float DC)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dcMutex);
    m_impl->setDc(DC);
}

float VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::getDc() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_dcMutex);
    return m_impl->getDc();
}
void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::setPhase1(float Phase1)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase1Mutex);
    m_impl->setPhase1(Phase1);
}

float VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::getPhase1() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase1Mutex);
    return m_impl->getPhase1();
}
void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::setPhase2(float Phase2)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase2Mutex);
    m_impl->setPhase2(Phase2);
}

float VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::getPhase2() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase2Mutex);
    return m_impl->getPhase2();
}
void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::setPhase3(float Phase3)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase3Mutex);
    m_impl->setPhase3(Phase3);
}

float VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::getPhase3() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase3Mutex);
    return m_impl->getPhase3();
}

IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher& VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}