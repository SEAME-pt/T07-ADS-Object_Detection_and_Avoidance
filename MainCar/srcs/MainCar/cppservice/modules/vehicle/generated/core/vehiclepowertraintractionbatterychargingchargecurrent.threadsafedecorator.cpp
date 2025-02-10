

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargecurrent.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargeCurrent> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::setDc(float DC)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dcMutex);
    m_impl->setDc(DC);
}

float VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::getDc() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_dcMutex);
    return m_impl->getDc();
}
void VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::setPhase1(float Phase1)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase1Mutex);
    m_impl->setPhase1(Phase1);
}

float VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::getPhase1() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase1Mutex);
    return m_impl->getPhase1();
}
void VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::setPhase2(float Phase2)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase2Mutex);
    m_impl->setPhase2(Phase2);
}

float VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::getPhase2() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase2Mutex);
    return m_impl->getPhase2();
}
void VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::setPhase3(float Phase3)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase3Mutex);
    m_impl->setPhase3(Phase3);
}

float VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::getPhase3() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase3Mutex);
    return m_impl->getPhase3();
}

IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher& VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}