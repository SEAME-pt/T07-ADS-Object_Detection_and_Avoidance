

#include "vehicle/implementation/vehiclepowertrainelectricmotor.h"
#include "vehicle/generated/core/vehiclepowertrainelectricmotor.publisher.h"
#include "vehicle/generated/core/vehiclepowertrainelectricmotor.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainElectricMotor::VehiclePowertrainElectricMotor()
    : m_publisher(std::make_unique<VehiclePowertrainElectricMotorPublisher>())
{
}
VehiclePowertrainElectricMotor::~VehiclePowertrainElectricMotor()
{
}

void VehiclePowertrainElectricMotor::setEngineCode(const std::string& EngineCode)
{
    if (m_data.m_EngineCode != EngineCode) {
        m_data.m_EngineCode = EngineCode;
        m_publisher->publishEngineCodeChanged(EngineCode);
    }
}

const std::string& VehiclePowertrainElectricMotor::getEngineCode() const
{
    return m_data.m_EngineCode;
}

void VehiclePowertrainElectricMotor::setMaxPower(int32_t MaxPower)
{
    if (m_data.m_MaxPower != MaxPower) {
        m_data.m_MaxPower = MaxPower;
        m_publisher->publishMaxPowerChanged(MaxPower);
    }
}

int32_t VehiclePowertrainElectricMotor::getMaxPower() const
{
    return m_data.m_MaxPower;
}

void VehiclePowertrainElectricMotor::setMaxTorque(int32_t MaxTorque)
{
    if (m_data.m_MaxTorque != MaxTorque) {
        m_data.m_MaxTorque = MaxTorque;
        m_publisher->publishMaxTorqueChanged(MaxTorque);
    }
}

int32_t VehiclePowertrainElectricMotor::getMaxTorque() const
{
    return m_data.m_MaxTorque;
}

void VehiclePowertrainElectricMotor::setMaxRegenPower(int32_t MaxRegenPower)
{
    if (m_data.m_MaxRegenPower != MaxRegenPower) {
        m_data.m_MaxRegenPower = MaxRegenPower;
        m_publisher->publishMaxRegenPowerChanged(MaxRegenPower);
    }
}

int32_t VehiclePowertrainElectricMotor::getMaxRegenPower() const
{
    return m_data.m_MaxRegenPower;
}

void VehiclePowertrainElectricMotor::setMaxRegenTorque(int32_t MaxRegenTorque)
{
    if (m_data.m_MaxRegenTorque != MaxRegenTorque) {
        m_data.m_MaxRegenTorque = MaxRegenTorque;
        m_publisher->publishMaxRegenTorqueChanged(MaxRegenTorque);
    }
}

int32_t VehiclePowertrainElectricMotor::getMaxRegenTorque() const
{
    return m_data.m_MaxRegenTorque;
}

void VehiclePowertrainElectricMotor::setSpeed(int32_t Speed)
{
    if (m_data.m_Speed != Speed) {
        m_data.m_Speed = Speed;
        m_publisher->publishSpeedChanged(Speed);
    }
}

int32_t VehiclePowertrainElectricMotor::getSpeed() const
{
    return m_data.m_Speed;
}

void VehiclePowertrainElectricMotor::setTimeInUse(float TimeInUse)
{
    if (m_data.m_TimeInUse != TimeInUse) {
        m_data.m_TimeInUse = TimeInUse;
        m_publisher->publishTimeInUseChanged(TimeInUse);
    }
}

float VehiclePowertrainElectricMotor::getTimeInUse() const
{
    return m_data.m_TimeInUse;
}

void VehiclePowertrainElectricMotor::setTemperature(float Temperature)
{
    if (m_data.m_Temperature != Temperature) {
        m_data.m_Temperature = Temperature;
        m_publisher->publishTemperatureChanged(Temperature);
    }
}

float VehiclePowertrainElectricMotor::getTemperature() const
{
    return m_data.m_Temperature;
}

void VehiclePowertrainElectricMotor::setCoolantTemperatureDeprecated(float CoolantTemperature_Deprecated)
{
    if (m_data.m_CoolantTemperature_Deprecated != CoolantTemperature_Deprecated) {
        m_data.m_CoolantTemperature_Deprecated = CoolantTemperature_Deprecated;
        m_publisher->publishCoolantTemperatureDeprecatedChanged(CoolantTemperature_Deprecated);
    }
}

float VehiclePowertrainElectricMotor::getCoolantTemperatureDeprecated() const
{
    return m_data.m_CoolantTemperature_Deprecated;
}

void VehiclePowertrainElectricMotor::setPower(int32_t Power)
{
    if (m_data.m_Power != Power) {
        m_data.m_Power = Power;
        m_publisher->publishPowerChanged(Power);
    }
}

int32_t VehiclePowertrainElectricMotor::getPower() const
{
    return m_data.m_Power;
}

void VehiclePowertrainElectricMotor::setTorque(int32_t Torque)
{
    if (m_data.m_Torque != Torque) {
        m_data.m_Torque = Torque;
        m_publisher->publishTorqueChanged(Torque);
    }
}

int32_t VehiclePowertrainElectricMotor::getTorque() const
{
    return m_data.m_Torque;
}

IVehiclePowertrainElectricMotorPublisher& VehiclePowertrainElectricMotor::_getPublisher() const
{
    return *m_publisher;
}
