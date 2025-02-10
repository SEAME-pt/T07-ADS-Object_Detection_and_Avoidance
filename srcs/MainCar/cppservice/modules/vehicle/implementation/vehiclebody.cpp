

#include "vehicle/implementation/vehiclebody.h"
#include "vehicle/generated/core/vehiclebody.publisher.h"
#include "vehicle/generated/core/vehiclebody.data.h"

using namespace Cpp::Vehicle;

VehicleBody::VehicleBody()
    : m_publisher(std::make_unique<VehicleBodyPublisher>())
{
}
VehicleBody::~VehicleBody()
{
}

void VehicleBody::setPowerOptimizeLevel(int32_t PowerOptimizeLevel)
{
    if (m_data.m_PowerOptimizeLevel != PowerOptimizeLevel) {
        m_data.m_PowerOptimizeLevel = PowerOptimizeLevel;
        m_publisher->publishPowerOptimizeLevelChanged(PowerOptimizeLevel);
    }
}

int32_t VehicleBody::getPowerOptimizeLevel() const
{
    return m_data.m_PowerOptimizeLevel;
}

void VehicleBody::setIsAutoPowerOptimize(bool IsAutoPowerOptimize)
{
    if (m_data.m_IsAutoPowerOptimize != IsAutoPowerOptimize) {
        m_data.m_IsAutoPowerOptimize = IsAutoPowerOptimize;
        m_publisher->publishIsAutoPowerOptimizeChanged(IsAutoPowerOptimize);
    }
}

bool VehicleBody::getIsAutoPowerOptimize() const
{
    return m_data.m_IsAutoPowerOptimize;
}

IVehicleBodyPublisher& VehicleBody::_getPublisher() const
{
    return *m_publisher;
}
