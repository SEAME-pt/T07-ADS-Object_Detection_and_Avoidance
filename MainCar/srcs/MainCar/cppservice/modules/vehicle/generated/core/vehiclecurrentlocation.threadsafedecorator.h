
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleCurrentLocationThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_CurrentLocation interface.
* @see Vehicle_CurrentLocation
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleCurrentLocation> testVehicleCurrentLocation = std::make_unique<VehicleCurrentLocationThreadSafeDecorator>(std::make_shared<VehicleCurrentLocation>());

// Thread safe access
auto timestamp = testVehicleCurrentLocation->getTimestamp();
testVehicleCurrentLocation->setTimestamp(std::string());
auto latitude = testVehicleCurrentLocation->getLatitude();
testVehicleCurrentLocation->setLatitude(0.0);
auto longitude = testVehicleCurrentLocation->getLongitude();
testVehicleCurrentLocation->setLongitude(0.0);
auto heading = testVehicleCurrentLocation->getHeading();
testVehicleCurrentLocation->setHeading(0.0);
auto horizontalAccuracy = testVehicleCurrentLocation->getHorizontalAccuracy();
testVehicleCurrentLocation->setHorizontalAccuracy(0.0);
auto altitude = testVehicleCurrentLocation->getAltitude();
testVehicleCurrentLocation->setAltitude(0.0);
auto verticalAccuracy = testVehicleCurrentLocation->getVerticalAccuracy();
testVehicleCurrentLocation->setVerticalAccuracy(0.0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleCurrentLocationThreadSafeDecorator : public IVehicleCurrentLocation
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_CurrentLocation object to make thread safe.
    */
    explicit VehicleCurrentLocationThreadSafeDecorator(std::shared_ptr<IVehicleCurrentLocation> impl);

    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    void setTimestamp(const std::string& Timestamp) override;
    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    const std::string& getTimestamp() const override;

    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    void setLatitude(double Latitude) override;
    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    double getLatitude() const override;

    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    void setLongitude(double Longitude) override;
    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    double getLongitude() const override;

    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    void setHeading(double Heading) override;
    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    double getHeading() const override;

    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    void setHorizontalAccuracy(double HorizontalAccuracy) override;
    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    double getHorizontalAccuracy() const override;

    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    void setAltitude(double Altitude) override;
    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    double getAltitude() const override;

    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    void setVerticalAccuracy(double VerticalAccuracy) override;
    /** Guards and forwards call to Vehicle_CurrentLocation implementation. */
    double getVerticalAccuracy() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_CurrentLocation.
    */
    IVehicleCurrentLocationPublisher& _getPublisher() const override;
private:
    /** The Vehicle_CurrentLocation object which is guarded */
    std::shared_ptr<IVehicleCurrentLocation> m_impl;
    // Mutex for timestamp property
    mutable std::shared_timed_mutex m_timestampMutex;
    // Mutex for latitude property
    mutable std::shared_timed_mutex m_latitudeMutex;
    // Mutex for longitude property
    mutable std::shared_timed_mutex m_longitudeMutex;
    // Mutex for heading property
    mutable std::shared_timed_mutex m_headingMutex;
    // Mutex for horizontalAccuracy property
    mutable std::shared_timed_mutex m_horizontalAccuracyMutex;
    // Mutex for altitude property
    mutable std::shared_timed_mutex m_altitudeMutex;
    // Mutex for verticalAccuracy property
    mutable std::shared_timed_mutex m_verticalAccuracyMutex;
};
} // namespace Vehicle
} // namespace Cpp
