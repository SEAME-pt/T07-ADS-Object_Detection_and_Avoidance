

#include "vehicle/generated/core/vehiclepowertrainelectricmotorenginecoolant.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainElectricMotorEngineCoolant> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::setCapacity(float Capacity)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_capacityMutex);
    m_impl->setCapacity(Capacity);
}

float VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::getCapacity() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_capacityMutex);
    return m_impl->getCapacity();
}
void VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::setTemperature(float Temperature)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    m_impl->setTemperature(Temperature);
}

float VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::getTemperature() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_temperatureMutex);
    return m_impl->getTemperature();
}
void VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::setLevel(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_levelMutex);
    m_impl->setLevel(Level);
}

Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::getLevel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_levelMutex);
    return m_impl->getLevel();
}
void VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::setLifeRemaining(int32_t LifeRemaining)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_lifeRemainingMutex);
    m_impl->setLifeRemaining(LifeRemaining);
}

int32_t VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::getLifeRemaining() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_lifeRemainingMutex);
    return m_impl->getLifeRemaining();
}

IVehiclePowertrainElectricMotorEngineCoolantPublisher& VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}