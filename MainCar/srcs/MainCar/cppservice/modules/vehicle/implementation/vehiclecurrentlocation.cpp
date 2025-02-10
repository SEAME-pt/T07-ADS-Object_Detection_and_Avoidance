

#include "vehicle/implementation/vehiclecurrentlocation.h"
#include "vehicle/generated/core/vehiclecurrentlocation.publisher.h"
#include "vehicle/generated/core/vehiclecurrentlocation.data.h"

using namespace Cpp::Vehicle;

VehicleCurrentLocation::VehicleCurrentLocation()
    : m_publisher(std::make_unique<VehicleCurrentLocationPublisher>())
{
}
VehicleCurrentLocation::~VehicleCurrentLocation()
{
}

void VehicleCurrentLocation::setTimestamp(const std::string& Timestamp)
{
    if (m_data.m_Timestamp != Timestamp) {
        m_data.m_Timestamp = Timestamp;
        m_publisher->publishTimestampChanged(Timestamp);
    }
}

const std::string& VehicleCurrentLocation::getTimestamp() const
{
    return m_data.m_Timestamp;
}

void VehicleCurrentLocation::setLatitude(double Latitude)
{
    if (m_data.m_Latitude != Latitude) {
        m_data.m_Latitude = Latitude;
        m_publisher->publishLatitudeChanged(Latitude);
    }
}

double VehicleCurrentLocation::getLatitude() const
{
    return m_data.m_Latitude;
}

void VehicleCurrentLocation::setLongitude(double Longitude)
{
    if (m_data.m_Longitude != Longitude) {
        m_data.m_Longitude = Longitude;
        m_publisher->publishLongitudeChanged(Longitude);
    }
}

double VehicleCurrentLocation::getLongitude() const
{
    return m_data.m_Longitude;
}

void VehicleCurrentLocation::setHeading(double Heading)
{
    if (m_data.m_Heading != Heading) {
        m_data.m_Heading = Heading;
        m_publisher->publishHeadingChanged(Heading);
    }
}

double VehicleCurrentLocation::getHeading() const
{
    return m_data.m_Heading;
}

void VehicleCurrentLocation::setHorizontalAccuracy(double HorizontalAccuracy)
{
    if (m_data.m_HorizontalAccuracy != HorizontalAccuracy) {
        m_data.m_HorizontalAccuracy = HorizontalAccuracy;
        m_publisher->publishHorizontalAccuracyChanged(HorizontalAccuracy);
    }
}

double VehicleCurrentLocation::getHorizontalAccuracy() const
{
    return m_data.m_HorizontalAccuracy;
}

void VehicleCurrentLocation::setAltitude(double Altitude)
{
    if (m_data.m_Altitude != Altitude) {
        m_data.m_Altitude = Altitude;
        m_publisher->publishAltitudeChanged(Altitude);
    }
}

double VehicleCurrentLocation::getAltitude() const
{
    return m_data.m_Altitude;
}

void VehicleCurrentLocation::setVerticalAccuracy(double VerticalAccuracy)
{
    if (m_data.m_VerticalAccuracy != VerticalAccuracy) {
        m_data.m_VerticalAccuracy = VerticalAccuracy;
        m_publisher->publishVerticalAccuracyChanged(VerticalAccuracy);
    }
}

double VehicleCurrentLocation::getVerticalAccuracy() const
{
    return m_data.m_VerticalAccuracy;
}

IVehicleCurrentLocationPublisher& VehicleCurrentLocation::_getPublisher() const
{
    return *m_publisher;
}
