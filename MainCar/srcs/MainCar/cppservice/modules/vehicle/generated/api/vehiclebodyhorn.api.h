#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodyHornSubscriber;
class IVehicleBodyHornPublisher;

/**
*
* IVehicleBodyHorn provides an interface for
 *  - methods defined for your Vehicle_Body_Horn 
 *  - property setters and getters for defined properties
 * The IVehicleBodyHorn also provides an interface to access a publisher IVehicleBodyHornPublisher, a class used by IVehicleBodyHornSubscriber clients.
 * The implementation should notify the publisher IVehicleBodyHornPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodyHornSubscriber, IVehicleBodyHornPublisher
 * and the example implementation Vehicle_Body_Horn  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBodyHorn
{
public:
    virtual ~IVehicleBodyHorn() = default;

    /**
    * Sets the value of the IsActive property.
    * @param IsActive Horn active or inactive. True = Active. False = Inactive.
    */
    virtual void setIsActive(bool IsActive) = 0;
    /**
    * Gets the value of the IsActive property.
    * @return Horn active or inactive. True = Active. False = Inactive.
    */
    virtual bool getIsActive() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Horn changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body_Horn.
    */
    virtual IVehicleBodyHornPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodyHornSubscriber contains functions to allow informing about signals or property changes of the IVehicleBodyHorn implementation.
 * The implementation for IVehicleBodyHorn should provide mechanism for subscription of the IVehicleBodyHornSubscriber clients.
 * See IVehicleBodyHornPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBodyHorn should call the IVehicleBodyHornSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodyHornSubscriber class to implement clients of the IVehicleBodyHorn or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyHornSubscriber
{
public:
    virtual ~IVehicleBodyHornSubscriber() = default;
    /**
    * Called by the IVehicleBodyHornPublisher when IsActive value has changed if subscribed for the IsActive change.
    * @param IsActive Horn active or inactive. True = Active. False = Inactive.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsActiveChanged(bool IsActive) = 0;
};

/** Callback for changes of IsActive */
using VehicleBodyHornIsActivePropertyCb = std::function<void(bool IsActive)>;


/**
 * The IVehicleBodyHornPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBodyHorn implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodyHornSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBodyHorn on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyHornPublisher
{
public:
    virtual ~IVehicleBodyHornPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body_Horn.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodyHornSubscriber which is subscribed in this function to any change of the Vehicle_Body_Horn.
    */
    virtual void subscribeToAllChanges(IVehicleBodyHornSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body_Horn.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodyHornSubscriber which subscription for any change of the Vehicle_Body_Horn is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodyHornSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsActive value changes.
    * If your subscriber uses subscription with IVehicleBodyHornSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyHornIsActivePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsActiveChanged(VehicleBodyHornIsActivePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsActive property changes.
    * If your subscriber uses subscription with IVehicleBodyHornSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsActiveChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Horn implementation when property IsActive changes.
    * @param The new value of IsActive.
    * Horn active or inactive. True = Active. False = Inactive.
    */
    virtual void publishIsActiveChanged(bool IsActive) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
