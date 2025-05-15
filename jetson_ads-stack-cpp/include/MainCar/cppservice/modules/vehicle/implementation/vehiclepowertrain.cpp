

#include "vehicle/implementation/vehiclepowertrain.h"
#include "vehicle/generated/core/vehiclepowertrain.publisher.h"
#include "vehicle/generated/core/vehiclepowertrain.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrain::VehiclePowertrain()
    : m_publisher(std::make_unique<VehiclePowertrainPublisher>())
{
}
VehiclePowertrain::~VehiclePowertrain()
{
}

void VehiclePowertrain::setAccumulatedBrakingEnergy(float AccumulatedBrakingEnergy)
{
    if (m_data.m_AccumulatedBrakingEnergy != AccumulatedBrakingEnergy) {
        m_data.m_AccumulatedBrakingEnergy = AccumulatedBrakingEnergy;
        m_publisher->publishAccumulatedBrakingEnergyChanged(AccumulatedBrakingEnergy);
    }
}

float VehiclePowertrain::getAccumulatedBrakingEnergy() const
{
    return m_data.m_AccumulatedBrakingEnergy;
}

void VehiclePowertrain::setRange(int32_t Range)
{
    if (m_data.m_Range != Range) {
        m_data.m_Range = Range;
        m_publisher->publishRangeChanged(Range);
    }
}

int32_t VehiclePowertrain::getRange() const
{
    return m_data.m_Range;
}

void VehiclePowertrain::setTimeRemaining(int32_t TimeRemaining)
{
    if (m_data.m_TimeRemaining != TimeRemaining) {
        m_data.m_TimeRemaining = TimeRemaining;
        m_publisher->publishTimeRemainingChanged(TimeRemaining);
    }
}

int32_t VehiclePowertrain::getTimeRemaining() const
{
    return m_data.m_TimeRemaining;
}

void VehiclePowertrain::setType(Vehicle_Powertrain_Type_ValueEnum Type)
{
    if (m_data.m_Type != Type) {
        m_data.m_Type = Type;
        m_publisher->publishTypeChanged(Type);
    }
}

Vehicle_Powertrain_Type_ValueEnum VehiclePowertrain::getType() const
{
    return m_data.m_Type;
}

void VehiclePowertrain::setPowerOptimizeLevel(int32_t PowerOptimizeLevel)
{
    if (m_data.m_PowerOptimizeLevel != PowerOptimizeLevel) {
        m_data.m_PowerOptimizeLevel = PowerOptimizeLevel;
        m_publisher->publishPowerOptimizeLevelChanged(PowerOptimizeLevel);
    }
}

int32_t VehiclePowertrain::getPowerOptimizeLevel() const
{
    return m_data.m_PowerOptimizeLevel;
}

void VehiclePowertrain::setIsAutoPowerOptimize(bool IsAutoPowerOptimize)
{
    if (m_data.m_IsAutoPowerOptimize != IsAutoPowerOptimize) {
        m_data.m_IsAutoPowerOptimize = IsAutoPowerOptimize;
        m_publisher->publishIsAutoPowerOptimizeChanged(IsAutoPowerOptimize);
    }
}

bool VehiclePowertrain::getIsAutoPowerOptimize() const
{
    return m_data.m_IsAutoPowerOptimize;
}

IVehiclePowertrainPublisher& VehiclePowertrain::_getPublisher() const
{
    return *m_publisher;
}
