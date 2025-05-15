

#include "vehicle/implementation/vehiclebodylightsdirectionindicatorright.h"
#include "vehicle/generated/core/vehiclebodylightsdirectionindicatorright.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsdirectionindicatorright.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsDirectionIndicatorRight::VehicleBodyLightsDirectionIndicatorRight()
    : m_publisher(std::make_unique<VehicleBodyLightsDirectionIndicatorRightPublisher>())
{
}
VehicleBodyLightsDirectionIndicatorRight::~VehicleBodyLightsDirectionIndicatorRight()
{
}

void VehicleBodyLightsDirectionIndicatorRight::setIsSignaling(bool IsSignaling)
{
    if (m_data.m_IsSignaling != IsSignaling) {
        m_data.m_IsSignaling = IsSignaling;
        m_publisher->publishIsSignalingChanged(IsSignaling);
    }
}

bool VehicleBodyLightsDirectionIndicatorRight::getIsSignaling() const
{
    return m_data.m_IsSignaling;
}

void VehicleBodyLightsDirectionIndicatorRight::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsDirectionIndicatorRight::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsDirectionIndicatorRightPublisher& VehicleBodyLightsDirectionIndicatorRight::_getPublisher() const
{
    return *m_publisher;
}
