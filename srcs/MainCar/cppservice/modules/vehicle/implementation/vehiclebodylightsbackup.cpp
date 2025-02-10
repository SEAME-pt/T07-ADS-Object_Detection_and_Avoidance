

#include "vehicle/implementation/vehiclebodylightsbackup.h"
#include "vehicle/generated/core/vehiclebodylightsbackup.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsbackup.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsBackup::VehicleBodyLightsBackup()
    : m_publisher(std::make_unique<VehicleBodyLightsBackupPublisher>())
{
}
VehicleBodyLightsBackup::~VehicleBodyLightsBackup()
{
}

void VehicleBodyLightsBackup::setIsOn(bool IsOn)
{
    if (m_data.m_IsOn != IsOn) {
        m_data.m_IsOn = IsOn;
        m_publisher->publishIsOnChanged(IsOn);
    }
}

bool VehicleBodyLightsBackup::getIsOn() const
{
    return m_data.m_IsOn;
}

void VehicleBodyLightsBackup::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsBackup::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsBackupPublisher& VehicleBodyLightsBackup::_getPublisher() const
{
    return *m_publisher;
}
