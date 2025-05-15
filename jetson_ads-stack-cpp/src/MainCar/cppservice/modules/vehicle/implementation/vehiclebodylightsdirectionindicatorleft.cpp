

#include "vehicle/implementation/vehiclebodylightsdirectionindicatorleft.h"
#include "vehicle/generated/core/vehiclebodylightsdirectionindicatorleft.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsdirectionindicatorleft.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsDirectionIndicatorLeft::VehicleBodyLightsDirectionIndicatorLeft()
    : m_publisher(std::make_unique<VehicleBodyLightsDirectionIndicatorLeftPublisher>())
{
}
VehicleBodyLightsDirectionIndicatorLeft::~VehicleBodyLightsDirectionIndicatorLeft()
{
}

void VehicleBodyLightsDirectionIndicatorLeft::setIsSignaling(bool IsSignaling)
{
    if (m_data.m_IsSignaling != IsSignaling) {
        m_data.m_IsSignaling = IsSignaling;
        m_publisher->publishIsSignalingChanged(IsSignaling);
    }
}

bool VehicleBodyLightsDirectionIndicatorLeft::getIsSignaling() const
{
    return m_data.m_IsSignaling;
}

void VehicleBodyLightsDirectionIndicatorLeft::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsDirectionIndicatorLeft::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsDirectionIndicatorLeftPublisher& VehicleBodyLightsDirectionIndicatorLeft::_getPublisher() const
{
    return *m_publisher;
}
