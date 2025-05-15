
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclecurrentlocation.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_CurrentLocation implementation.
*/
class CPP_VEHICLE_EXPORT VehicleCurrentLocation : public IVehicleCurrentLocation
{
public:
    explicit VehicleCurrentLocation();
    ~VehicleCurrentLocation();
public:
    /**
    * Timestamp Timestamp from GNSS system for current location, formatted according to ISO 8601 with UTC time zone.
    */
    void setTimestamp(const std::string& Timestamp) override;
    const std::string& getTimestamp() const override;
    
    /**
    * Latitude Current latitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    */
    void setLatitude(double Latitude) override;
    double getLatitude() const override;
    
    /**
    * Longitude Current longitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    */
    void setLongitude(double Longitude) override;
    double getLongitude() const override;
    
    /**
    * Heading Current heading relative to geographic north. 0 = North, 90 = East, 180 = South, 270 = West.
    */
    void setHeading(double Heading) override;
    double getHeading() const override;
    
    /**
    * HorizontalAccuracy Accuracy of the latitude and longitude coordinates.
    */
    void setHorizontalAccuracy(double HorizontalAccuracy) override;
    double getHorizontalAccuracy() const override;
    
    /**
    * Altitude Current altitude relative to WGS 84 reference ellipsoid, as measured at the position of GNSS receiver antenna.
    */
    void setAltitude(double Altitude) override;
    double getAltitude() const override;
    
    /**
    * VerticalAccuracy Accuracy of altitude.
    */
    void setVerticalAccuracy(double VerticalAccuracy) override;
    double getVerticalAccuracy() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation changes and signal emission.
    * @return The publisher for Vehicle_CurrentLocation.
    */
    IVehicleCurrentLocationPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_CurrentLocation. */
    std::unique_ptr<IVehicleCurrentLocationPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_CurrentLocation. */
    VehicleCurrentLocationData m_data;
};
} // namespace Vehicle
} // namespace Cpp
