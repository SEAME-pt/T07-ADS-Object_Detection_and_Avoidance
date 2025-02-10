

#include "vehicle/implementation/vehiclepowertrainelectricmotorenginecoolant.h"
#include "vehicle/generated/core/vehiclepowertrainelectricmotorenginecoolant.publisher.h"
#include "vehicle/generated/core/vehiclepowertrainelectricmotorenginecoolant.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainElectricMotorEngineCoolant::VehiclePowertrainElectricMotorEngineCoolant()
    : m_publisher(std::make_unique<VehiclePowertrainElectricMotorEngineCoolantPublisher>())
{
}
VehiclePowertrainElectricMotorEngineCoolant::~VehiclePowertrainElectricMotorEngineCoolant()
{
}

void VehiclePowertrainElectricMotorEngineCoolant::setCapacity(float Capacity)
{
    if (m_data.m_Capacity != Capacity) {
        m_data.m_Capacity = Capacity;
        m_publisher->publishCapacityChanged(Capacity);
    }
}

float VehiclePowertrainElectricMotorEngineCoolant::getCapacity() const
{
    return m_data.m_Capacity;
}

void VehiclePowertrainElectricMotorEngineCoolant::setTemperature(float Temperature)
{
    if (m_data.m_Temperature != Temperature) {
        m_data.m_Temperature = Temperature;
        m_publisher->publishTemperatureChanged(Temperature);
    }
}

float VehiclePowertrainElectricMotorEngineCoolant::getTemperature() const
{
    return m_data.m_Temperature;
}

void VehiclePowertrainElectricMotorEngineCoolant::setLevel(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level)
{
    if (m_data.m_Level != Level) {
        m_data.m_Level = Level;
        m_publisher->publishLevelChanged(Level);
    }
}

Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum VehiclePowertrainElectricMotorEngineCoolant::getLevel() const
{
    return m_data.m_Level;
}

void VehiclePowertrainElectricMotorEngineCoolant::setLifeRemaining(int32_t LifeRemaining)
{
    if (m_data.m_LifeRemaining != LifeRemaining) {
        m_data.m_LifeRemaining = LifeRemaining;
        m_publisher->publishLifeRemainingChanged(LifeRemaining);
    }
}

int32_t VehiclePowertrainElectricMotorEngineCoolant::getLifeRemaining() const
{
    return m_data.m_LifeRemaining;
}

IVehiclePowertrainElectricMotorEngineCoolantPublisher& VehiclePowertrainElectricMotorEngineCoolant::_getPublisher() const
{
    return *m_publisher;
}
