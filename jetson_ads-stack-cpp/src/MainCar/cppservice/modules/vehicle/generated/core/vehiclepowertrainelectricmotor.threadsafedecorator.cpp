

#include "vehicle/generated/core/vehiclepowertrainelectricmotor.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainElectricMotorThreadSafeDecorator::VehiclePowertrainElectricMotorThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainElectricMotor> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setEngineCode(const std::string& EngineCode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_engineCodeMutex);
    m_impl->setEngineCode(EngineCode);
}

const std::string& VehiclePowertrainElectricMotorThreadSafeDecorator::getEngineCode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_engineCodeMutex);
    return m_impl->getEngineCode();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setMaxPower(int32_t MaxPower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxPowerMutex);
    m_impl->setMaxPower(MaxPower);
}

int32_t VehiclePowertrainElectricMotorThreadSafeDecorator::getMaxPower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_maxPowerMutex);
    return m_impl->getMaxPower();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setMaxTorque(int32_t MaxTorque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxTorqueMutex);
    m_impl->setMaxTorque(MaxTorque);
}

int32_t VehiclePowertrainElectricMotorThreadSafeDecorator::getMaxTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_maxTorqueMutex);
    return m_impl->getMaxTorque();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setMaxRegenPower(int32_t MaxRegenPower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxRegenPowerMutex);
    m_impl->setMaxRegenPower(MaxRegenPower);
}

int32_t VehiclePowertrainElectricMotorThreadSafeDecorator::getMaxRegenPower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_maxRegenPowerMutex);
    return m_impl->getMaxRegenPower();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setMaxRegenTorque(int32_t MaxRegenTorque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxRegenTorqueMutex);
    m_impl->setMaxRegenTorque(MaxRegenTorque);
}

int32_t VehiclePowertrainElectricMotorThreadSafeDecorator::getMaxRegenTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_maxRegenTorqueMutex);
    return m_impl->getMaxRegenTorque();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setSpeed(int32_t Speed)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_speedMutex);
    m_impl->setSpeed(Speed);
}

int32_t VehiclePowertrainElectricMotorThreadSafeDecorator::getSpeed() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_speedMutex);
    return m_impl->getSpeed();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setTimeInUse(float TimeInUse)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeInUseMutex);
    m_impl->setTimeInUse(TimeInUse);
}

float VehiclePowertrainElectricMotorThreadSafeDecorator::getTimeInUse() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_timeInUseMutex);
    return m_impl->getTimeInUse();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setTemperature(float Temperature)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    m_impl->setTemperature(Temperature);
}

float VehiclePowertrainElectricMotorThreadSafeDecorator::getTemperature() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    return m_impl->getTemperature();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setCoolantTemperatureDeprecated(float CoolantTemperature_Deprecated)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_coolantTemperatureDeprecatedMutex);
    m_impl->setCoolantTemperatureDeprecated(CoolantTemperature_Deprecated);
}

float VehiclePowertrainElectricMotorThreadSafeDecorator::getCoolantTemperatureDeprecated() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_coolantTemperatureDeprecatedMutex);
    return m_impl->getCoolantTemperatureDeprecated();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setPower(int32_t Power)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerMutex);
    m_impl->setPower(Power);
}

int32_t VehiclePowertrainElectricMotorThreadSafeDecorator::getPower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_powerMutex);
    return m_impl->getPower();
}
void VehiclePowertrainElectricMotorThreadSafeDecorator::setTorque(int32_t Torque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    m_impl->setTorque(Torque);
}

int32_t VehiclePowertrainElectricMotorThreadSafeDecorator::getTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    return m_impl->getTorque();
}

IVehiclePowertrainElectricMotorPublisher& VehiclePowertrainElectricMotorThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}