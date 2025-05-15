#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryChargingLocationSubscriber;
class IVehiclePowertrainTractionBatteryChargingLocationPublisher;

/**
*
* IVehiclePowertrainTractionBatteryChargingLocation provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Charging_Location 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryChargingLocation also provides an interface to access a publisher IVehiclePowertrainTractionBatteryChargingLocationPublisher, a class used by IVehiclePowertrainTractionBatteryChargingLocationSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryChargingLocationPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryChargingLocationSubscriber, IVehiclePowertrainTractionBatteryChargingLocationPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Charging_Location  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingLocation
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingLocation() = default;

    /**
    * Sets the value of the Latitude property.
    * @param Latitude Latitude of last or current charging event in WGS 84 geodetic coordinates.
    */
    virtual void setLatitude(double Latitude) = 0;
    /**
    * Gets the value of the Latitude property.
    * @return Latitude of last or current charging event in WGS 84 geodetic coordinates.
    */
    virtual double getLatitude() const = 0;

    /**
    * Sets the value of the Longitude property.
    * @param Longitude Longitude of last or current charging event in WGS 84 geodetic coordinates.
    */
    virtual void setLongitude(double Longitude) = 0;
    /**
    * Gets the value of the Longitude property.
    * @return Longitude of last or current charging event in WGS 84 geodetic coordinates.
    */
    virtual double getLongitude() const = 0;

    /**
    * Sets the value of the Altitude property.
    * @param Altitude Altitude relative to WGS 84 reference ellipsoid of last or current charging event.
    */
    virtual void setAltitude(double Altitude) = 0;
    /**
    * Gets the value of the Altitude property.
    * @return Altitude relative to WGS 84 reference ellipsoid of last or current charging event.
    */
    virtual double getAltitude() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_Location changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_Location.
    */
    virtual IVehiclePowertrainTractionBatteryChargingLocationPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryChargingLocationSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryChargingLocation implementation.
 * The implementation for IVehiclePowertrainTractionBatteryChargingLocation should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryChargingLocationSubscriber clients.
 * See IVehiclePowertrainTractionBatteryChargingLocationPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryChargingLocation should call the IVehiclePowertrainTractionBatteryChargingLocationSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryChargingLocationSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryChargingLocation or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingLocationSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingLocationSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingLocationPublisher when Latitude value has changed if subscribed for the Latitude change.
    * @param Latitude Latitude of last or current charging event in WGS 84 geodetic coordinates.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLatitudeChanged(double Latitude) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingLocationPublisher when Longitude value has changed if subscribed for the Longitude change.
    * @param Longitude Longitude of last or current charging event in WGS 84 geodetic coordinates.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLongitudeChanged(double Longitude) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingLocationPublisher when Altitude value has changed if subscribed for the Altitude change.
    * @param Altitude Altitude relative to WGS 84 reference ellipsoid of last or current charging event.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAltitudeChanged(double Altitude) = 0;
};

/** Callback for changes of Latitude */
using VehiclePowertrainTractionBatteryChargingLocationLatitudePropertyCb = std::function<void(double Latitude)>;
/** Callback for changes of Longitude */
using VehiclePowertrainTractionBatteryChargingLocationLongitudePropertyCb = std::function<void(double Longitude)>;
/** Callback for changes of Altitude */
using VehiclePowertrainTractionBatteryChargingLocationAltitudePropertyCb = std::function<void(double Altitude)>;


/**
 * The IVehiclePowertrainTractionBatteryChargingLocationPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryChargingLocation implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryChargingLocationSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryChargingLocation on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingLocationPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingLocationPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Charging_Location.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryChargingLocationSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Charging_Location.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingLocationSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Charging_Location.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryChargingLocationSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Charging_Location is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingLocationSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Latitude value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingLocationLatitudePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLatitudeChanged(VehiclePowertrainTractionBatteryChargingLocationLatitudePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Latitude property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLatitudeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Longitude value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingLocationLongitudePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLongitudeChanged(VehiclePowertrainTractionBatteryChargingLocationLongitudePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Longitude property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLongitudeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Altitude value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingLocationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingLocationAltitudePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAltitudeChanged(VehiclePowertrainTractionBatteryChargingLocationAltitudePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Altitude property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingLocationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAltitudeChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_Location implementation when property Latitude changes.
    * @param The new value of Latitude.
    * Latitude of last or current charging event in WGS 84 geodetic coordinates.
    */
    virtual void publishLatitudeChanged(double Latitude) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_Location implementation when property Longitude changes.
    * @param The new value of Longitude.
    * Longitude of last or current charging event in WGS 84 geodetic coordinates.
    */
    virtual void publishLongitudeChanged(double Longitude) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_Location implementation when property Altitude changes.
    * @param The new value of Altitude.
    * Altitude relative to WGS 84 reference ellipsoid of last or current charging event.
    */
    virtual void publishAltitudeChanged(double Altitude) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
