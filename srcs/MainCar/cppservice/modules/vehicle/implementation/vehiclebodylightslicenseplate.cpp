

#include "vehicle/implementation/vehiclebodylightslicenseplate.h"
#include "vehicle/generated/core/vehiclebodylightslicenseplate.publisher.h"
#include "vehicle/generated/core/vehiclebodylightslicenseplate.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsLicensePlate::VehicleBodyLightsLicensePlate()
    : m_publisher(std::make_unique<VehicleBodyLightsLicensePlatePublisher>())
{
}
VehicleBodyLightsLicensePlate::~VehicleBodyLightsLicensePlate()
{
}

void VehicleBodyLightsLicensePlate::setIsOn(bool IsOn)
{
    if (m_data.m_IsOn != IsOn) {
        m_data.m_IsOn = IsOn;
        m_publisher->publishIsOnChanged(IsOn);
    }
}

bool VehicleBodyLightsLicensePlate::getIsOn() const
{
    return m_data.m_IsOn;
}

void VehicleBodyLightsLicensePlate::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsLicensePlate::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsLicensePlatePublisher& VehicleBodyLightsLicensePlate::_getPublisher() const
{
    return *m_publisher;
}
