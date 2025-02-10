#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleCurrentLocationSubscriber;
class IVehicleCurrentLocationPublisher;

/**
*
* IVehicleCurrentLocation provides an interface for
 *  - methods defined for your Vehicle_CurrentLocation 
 *  - property setters and getters for defined properties
 * The IVehicleCurrentLocation also provides an interface to access a publisher IVehicleCurrentLocationPublisher, a class used by IVehicleCurrentLocationSubscriber clients.
 * The implementation should notify the publisher IVehicleCurrentLocationPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleCurrentLocationSubscriber, IVehicleCurrentLocationPublisher
 * and the example implementation Vehicle_CurrentLocation  or the
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocation
{
public:
    virtual ~IVehicleCurrentLocation() = default;

    /**
    * Sets the value of the Timestamp property.
    * @param Timestamp Timestamp from GNSS system for current location, formatted according to ISO 8601 with UTC time zone.
    */
    virtual void setTimestamp(const std::string& Timestamp) = 0;
    /**
    * Gets the value of the Timestamp property.
    * @return Timestamp from GNSS system for current location, formatted according to ISO 8601 with UTC time zone.
    */
    virtual const std::string& getTimestamp() const = 0;

    /**
    * Sets the value of the Latitude property.
    * @param Latitude Current latitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    */
    virtual void setLatitude(double Latitude) = 0;
    /**
    * Gets the value of the Latitude property.
    * @return Current latitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    */
    virtual double getLatitude() const = 0;

    /**
    * Sets the value of the Longitude property.
    * @param Longitude Current longitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    */
    virtual void setLongitude(double Longitude) = 0;
    /**
    * Gets the value of the Longitude property.
    * @return Current longitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    */
    virtual double getLongitude() const = 0;

    /**
    * Sets the value of the Heading property.
    * @param Heading Current heading relative to geographic north. 0 = North, 90 = East, 180 = South, 270 = West.
    */
    virtual void setHeading(double Heading) = 0;
    /**
    * Gets the value of the Heading property.
    * @return Current heading relative to geographic north. 0 = North, 90 = East, 180 = South, 270 = West.
    */
    virtual double getHeading() const = 0;

    /**
    * Sets the value of the HorizontalAccuracy property.
    * @param HorizontalAccuracy Accuracy of the latitude and longitude coordinates.
    */
    virtual void setHorizontalAccuracy(double HorizontalAccuracy) = 0;
    /**
    * Gets the value of the HorizontalAccuracy property.
    * @return Accuracy of the latitude and longitude coordinates.
    */
    virtual double getHorizontalAccuracy() const = 0;

    /**
    * Sets the value of the Altitude property.
    * @param Altitude Current altitude relative to WGS 84 reference ellipsoid, as measured at the position of GNSS receiver antenna.
    */
    virtual void setAltitude(double Altitude) = 0;
    /**
    * Gets the value of the Altitude property.
    * @return Current altitude relative to WGS 84 reference ellipsoid, as measured at the position of GNSS receiver antenna.
    */
    virtual double getAltitude() const = 0;

    /**
    * Sets the value of the VerticalAccuracy property.
    * @param VerticalAccuracy Accuracy of altitude.
    */
    virtual void setVerticalAccuracy(double VerticalAccuracy) = 0;
    /**
    * Gets the value of the VerticalAccuracy property.
    * @return Accuracy of altitude.
    */
    virtual double getVerticalAccuracy() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_CurrentLocation.
    */
    virtual IVehicleCurrentLocationPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleCurrentLocationSubscriber contains functions to allow informing about signals or property changes of the IVehicleCurrentLocation implementation.
 * The implementation for IVehicleCurrentLocation should provide mechanism for subscription of the IVehicleCurrentLocationSubscriber clients.
 * See IVehicleCurrentLocationPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleCurrentLocation should call the IVehicleCurrentLocationSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleCurrentLocationSubscriber class to implement clients of the IVehicleCurrentLocation or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocationSubscriber
{
public:
    virtual ~IVehicleCurrentLocationSubscriber() = default;
    /**
    * Called by the IVehicleCurrentLocationPublisher when Timestamp value has changed if subscribed for the Timestamp change.
    * @param Timestamp Timestamp from GNSS system for current location, formatted according to ISO 8601 with UTC time zone.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTimestampChanged(const std::string& Timestamp) = 0;
    /**
    * Called by the IVehicleCurrentLocationPublisher when Latitude value has changed if subscribed for the Latitude change.
    * @param Latitude Current latitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLatitudeChanged(double Latitude) = 0;
    /**
    * Called by the IVehicleCurrentLocationPublisher when Longitude value has changed if subscribed for the Longitude change.
    * @param Longitude Current longitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLongitudeChanged(double Longitude) = 0;
    /**
    * Called by the IVehicleCurrentLocationPublisher when Heading value has changed if subscribed for the Heading change.
    * @param Heading Current heading relative to geographic north. 0 = North, 90 = East, 180 = South, 270 = West.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onHeadingChanged(double Heading) = 0;
    /**
    * Called by the IVehicleCurrentLocationPublisher when HorizontalAccuracy value has changed if subscribed for the HorizontalAccuracy change.
    * @param HorizontalAccuracy Accuracy of the latitude and longitude coordinates.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onHorizontalAccuracyChanged(double HorizontalAccuracy) = 0;
    /**
    * Called by the IVehicleCurrentLocationPublisher when Altitude value has changed if subscribed for the Altitude change.
    * @param Altitude Current altitude relative to WGS 84 reference ellipsoid, as measured at the position of GNSS receiver antenna.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAltitudeChanged(double Altitude) = 0;
    /**
    * Called by the IVehicleCurrentLocationPublisher when VerticalAccuracy value has changed if subscribed for the VerticalAccuracy change.
    * @param VerticalAccuracy Accuracy of altitude.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVerticalAccuracyChanged(double VerticalAccuracy) = 0;
};

/** Callback for changes of Timestamp */
using VehicleCurrentLocationTimestampPropertyCb = std::function<void(const std::string& Timestamp)>;
/** Callback for changes of Latitude */
using VehicleCurrentLocationLatitudePropertyCb = std::function<void(double Latitude)>;
/** Callback for changes of Longitude */
using VehicleCurrentLocationLongitudePropertyCb = std::function<void(double Longitude)>;
/** Callback for changes of Heading */
using VehicleCurrentLocationHeadingPropertyCb = std::function<void(double Heading)>;
/** Callback for changes of HorizontalAccuracy */
using VehicleCurrentLocationHorizontalAccuracyPropertyCb = std::function<void(double HorizontalAccuracy)>;
/** Callback for changes of Altitude */
using VehicleCurrentLocationAltitudePropertyCb = std::function<void(double Altitude)>;
/** Callback for changes of VerticalAccuracy */
using VehicleCurrentLocationVerticalAccuracyPropertyCb = std::function<void(double VerticalAccuracy)>;


/**
 * The IVehicleCurrentLocationPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleCurrentLocation implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleCurrentLocationSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleCurrentLocation on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocationPublisher
{
public:
    virtual ~IVehicleCurrentLocationPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_CurrentLocation.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleCurrentLocationSubscriber which is subscribed in this function to any change of the Vehicle_CurrentLocation.
    */
    virtual void subscribeToAllChanges(IVehicleCurrentLocationSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_CurrentLocation.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleCurrentLocationSubscriber which subscription for any change of the Vehicle_CurrentLocation is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleCurrentLocationSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Timestamp value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationTimestampPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTimestampChanged(VehicleCurrentLocationTimestampPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Timestamp property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTimestampChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Latitude value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationLatitudePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLatitudeChanged(VehicleCurrentLocationLatitudePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Latitude property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLatitudeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Longitude value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationLongitudePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLongitudeChanged(VehicleCurrentLocationLongitudePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Longitude property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLongitudeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Heading value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationHeadingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToHeadingChanged(VehicleCurrentLocationHeadingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Heading property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromHeadingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for HorizontalAccuracy value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationHorizontalAccuracyPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToHorizontalAccuracyChanged(VehicleCurrentLocationHorizontalAccuracyPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from HorizontalAccuracy property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromHorizontalAccuracyChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Altitude value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationAltitudePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAltitudeChanged(VehicleCurrentLocationAltitudePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Altitude property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAltitudeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VerticalAccuracy value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationVerticalAccuracyPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVerticalAccuracyChanged(VehicleCurrentLocationVerticalAccuracyPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VerticalAccuracy property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVerticalAccuracyChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation implementation when property Timestamp changes.
    * @param The new value of Timestamp.
    * Timestamp from GNSS system for current location, formatted according to ISO 8601 with UTC time zone.
    */
    virtual void publishTimestampChanged(const std::string& Timestamp) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation implementation when property Latitude changes.
    * @param The new value of Latitude.
    * Current latitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    */
    virtual void publishLatitudeChanged(double Latitude) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation implementation when property Longitude changes.
    * @param The new value of Longitude.
    * Current longitude of vehicle in WGS 84 geodetic coordinates, as measured at the position of GNSS receiver antenna.
    */
    virtual void publishLongitudeChanged(double Longitude) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation implementation when property Heading changes.
    * @param The new value of Heading.
    * Current heading relative to geographic north. 0 = North, 90 = East, 180 = South, 270 = West.
    */
    virtual void publishHeadingChanged(double Heading) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation implementation when property HorizontalAccuracy changes.
    * @param The new value of HorizontalAccuracy.
    * Accuracy of the latitude and longitude coordinates.
    */
    virtual void publishHorizontalAccuracyChanged(double HorizontalAccuracy) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation implementation when property Altitude changes.
    * @param The new value of Altitude.
    * Current altitude relative to WGS 84 reference ellipsoid, as measured at the position of GNSS receiver antenna.
    */
    virtual void publishAltitudeChanged(double Altitude) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation implementation when property VerticalAccuracy changes.
    * @param The new value of VerticalAccuracy.
    * Accuracy of altitude.
    */
    virtual void publishVerticalAccuracyChanged(double VerticalAccuracy) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
