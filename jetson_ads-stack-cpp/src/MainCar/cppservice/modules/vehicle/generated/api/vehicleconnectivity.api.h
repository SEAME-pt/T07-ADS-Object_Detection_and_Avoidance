#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleConnectivitySubscriber;
class IVehicleConnectivityPublisher;

/**
*
* IVehicleConnectivity provides an interface for
 *  - methods defined for your Vehicle_Connectivity 
 *  - property setters and getters for defined properties
 * The IVehicleConnectivity also provides an interface to access a publisher IVehicleConnectivityPublisher, a class used by IVehicleConnectivitySubscriber clients.
 * The implementation should notify the publisher IVehicleConnectivityPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleConnectivitySubscriber, IVehicleConnectivityPublisher
 * and the example implementation Vehicle_Connectivity  or the
 */
class CPP_VEHICLE_EXPORT IVehicleConnectivity
{
public:
    virtual ~IVehicleConnectivity() = default;

    /**
    * Sets the value of the IsConnectivityAvailable property.
    * @param IsConnectivityAvailable Indicates if connectivity between vehicle and cloud is available. True = Connectivity is available. False = Connectivity is not available.
    */
    virtual void setIsConnectivityAvailable(bool IsConnectivityAvailable) = 0;
    /**
    * Gets the value of the IsConnectivityAvailable property.
    * @return Indicates if connectivity between vehicle and cloud is available. True = Connectivity is available. False = Connectivity is not available.
    */
    virtual bool getIsConnectivityAvailable() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Connectivity changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Connectivity.
    */
    virtual IVehicleConnectivityPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleConnectivitySubscriber contains functions to allow informing about signals or property changes of the IVehicleConnectivity implementation.
 * The implementation for IVehicleConnectivity should provide mechanism for subscription of the IVehicleConnectivitySubscriber clients.
 * See IVehicleConnectivityPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleConnectivity should call the IVehicleConnectivitySubscriber interface functions on either signal emit or property change.
 * You can use IVehicleConnectivitySubscriber class to implement clients of the IVehicleConnectivity or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleConnectivitySubscriber
{
public:
    virtual ~IVehicleConnectivitySubscriber() = default;
    /**
    * Called by the IVehicleConnectivityPublisher when IsConnectivityAvailable value has changed if subscribed for the IsConnectivityAvailable change.
    * @param IsConnectivityAvailable Indicates if connectivity between vehicle and cloud is available. True = Connectivity is available. False = Connectivity is not available.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsConnectivityAvailableChanged(bool IsConnectivityAvailable) = 0;
};

/** Callback for changes of IsConnectivityAvailable */
using VehicleConnectivityIsConnectivityAvailablePropertyCb = std::function<void(bool IsConnectivityAvailable)>;


/**
 * The IVehicleConnectivityPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleConnectivity implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleConnectivitySubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleConnectivity on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleConnectivityPublisher
{
public:
    virtual ~IVehicleConnectivityPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Connectivity.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleConnectivitySubscriber which is subscribed in this function to any change of the Vehicle_Connectivity.
    */
    virtual void subscribeToAllChanges(IVehicleConnectivitySubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Connectivity.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleConnectivitySubscriber which subscription for any change of the Vehicle_Connectivity is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleConnectivitySubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsConnectivityAvailable value changes.
    * If your subscriber uses subscription with IVehicleConnectivitySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleConnectivityIsConnectivityAvailablePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsConnectivityAvailableChanged(VehicleConnectivityIsConnectivityAvailablePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsConnectivityAvailable property changes.
    * If your subscriber uses subscription with IVehicleConnectivitySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsConnectivityAvailableChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Connectivity implementation when property IsConnectivityAvailable changes.
    * @param The new value of IsConnectivityAvailable.
    * Indicates if connectivity between vehicle and cloud is available. True = Connectivity is available. False = Connectivity is not available.
    */
    virtual void publishIsConnectivityAvailableChanged(bool IsConnectivityAvailable) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
