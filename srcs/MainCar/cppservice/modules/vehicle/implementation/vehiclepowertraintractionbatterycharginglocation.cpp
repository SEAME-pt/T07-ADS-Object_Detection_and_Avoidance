

#include "vehicle/implementation/vehiclepowertraintractionbatterycharginglocation.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycharginglocation.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycharginglocation.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTractionBatteryChargingLocation::VehiclePowertrainTractionBatteryChargingLocation()
    : m_publisher(std::make_unique<VehiclePowertrainTractionBatteryChargingLocationPublisher>())
{
}
VehiclePowertrainTractionBatteryChargingLocation::~VehiclePowertrainTractionBatteryChargingLocation()
{
}

void VehiclePowertrainTractionBatteryChargingLocation::setLatitude(double Latitude)
{
    if (m_data.m_Latitude != Latitude) {
        m_data.m_Latitude = Latitude;
        m_publisher->publishLatitudeChanged(Latitude);
    }
}

double VehiclePowertrainTractionBatteryChargingLocation::getLatitude() const
{
    return m_data.m_Latitude;
}

void VehiclePowertrainTractionBatteryChargingLocation::setLongitude(double Longitude)
{
    if (m_data.m_Longitude != Longitude) {
        m_data.m_Longitude = Longitude;
        m_publisher->publishLongitudeChanged(Longitude);
    }
}

double VehiclePowertrainTractionBatteryChargingLocation::getLongitude() const
{
    return m_data.m_Longitude;
}

void VehiclePowertrainTractionBatteryChargingLocation::setAltitude(double Altitude)
{
    if (m_data.m_Altitude != Altitude) {
        m_data.m_Altitude = Altitude;
        m_publisher->publishAltitudeChanged(Altitude);
    }
}

double VehiclePowertrainTractionBatteryChargingLocation::getAltitude() const
{
    return m_data.m_Altitude;
}

IVehiclePowertrainTractionBatteryChargingLocationPublisher& VehiclePowertrainTractionBatteryChargingLocation::_getPublisher() const
{
    return *m_publisher;
}
