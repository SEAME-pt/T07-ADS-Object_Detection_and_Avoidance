

#include "vehicle/implementation/vehiclebodylightshazard.h"
#include "vehicle/generated/core/vehiclebodylightshazard.publisher.h"
#include "vehicle/generated/core/vehiclebodylightshazard.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsHazard::VehicleBodyLightsHazard()
    : m_publisher(std::make_unique<VehicleBodyLightsHazardPublisher>())
{
}
VehicleBodyLightsHazard::~VehicleBodyLightsHazard()
{
}

void VehicleBodyLightsHazard::setIsSignaling(bool IsSignaling)
{
    if (m_data.m_IsSignaling != IsSignaling) {
        m_data.m_IsSignaling = IsSignaling;
        m_publisher->publishIsSignalingChanged(IsSignaling);
    }
}

bool VehicleBodyLightsHazard::getIsSignaling() const
{
    return m_data.m_IsSignaling;
}

void VehicleBodyLightsHazard::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsHazard::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsHazardPublisher& VehicleBodyLightsHazard::_getPublisher() const
{
    return *m_publisher;
}
