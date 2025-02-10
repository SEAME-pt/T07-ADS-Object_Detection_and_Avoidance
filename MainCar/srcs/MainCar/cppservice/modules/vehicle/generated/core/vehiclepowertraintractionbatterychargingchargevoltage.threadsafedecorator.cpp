

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargevoltage.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargeVoltage> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::setDc(float DC)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dcMutex);
    m_impl->setDc(DC);
}

float VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::getDc() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_dcMutex);
    return m_impl->getDc();
}
void VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::setPhase1(float Phase1)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase1Mutex);
    m_impl->setPhase1(Phase1);
}

float VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::getPhase1() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase1Mutex);
    return m_impl->getPhase1();
}
void VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::setPhase2(float Phase2)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase2Mutex);
    m_impl->setPhase2(Phase2);
}

float VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::getPhase2() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase2Mutex);
    return m_impl->getPhase2();
}
void VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::setPhase3(float Phase3)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase3Mutex);
    m_impl->setPhase3(Phase3);
}

float VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::getPhase3() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_phase3Mutex);
    return m_impl->getPhase3();
}

IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher& VehiclePowertrainTractionBatteryChargingChargeVoltageThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}