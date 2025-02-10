#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleCurrentLocationGnssReceiverSubscriber;
class IVehicleCurrentLocationGnssReceiverPublisher;

/**
*
* IVehicleCurrentLocationGnssReceiver provides an interface for
 *  - methods defined for your Vehicle_CurrentLocation_GNSSReceiver 
 *  - property setters and getters for defined properties
 * The IVehicleCurrentLocationGnssReceiver also provides an interface to access a publisher IVehicleCurrentLocationGnssReceiverPublisher, a class used by IVehicleCurrentLocationGnssReceiverSubscriber clients.
 * The implementation should notify the publisher IVehicleCurrentLocationGnssReceiverPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleCurrentLocationGnssReceiverSubscriber, IVehicleCurrentLocationGnssReceiverPublisher
 * and the example implementation Vehicle_CurrentLocation_GNSSReceiver  or the
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocationGnssReceiver
{
public:
    virtual ~IVehicleCurrentLocationGnssReceiver() = default;

    /**
    * Sets the value of the FixType property.
    * @param FixType Fix status of GNSS receiver.
    */
    virtual void setFixType(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType) = 0;
    /**
    * Gets the value of the FixType property.
    * @return Fix status of GNSS receiver.
    */
    virtual Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum getFixType() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation_GNSSReceiver changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_CurrentLocation_GNSSReceiver.
    */
    virtual IVehicleCurrentLocationGnssReceiverPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleCurrentLocationGnssReceiverSubscriber contains functions to allow informing about signals or property changes of the IVehicleCurrentLocationGnssReceiver implementation.
 * The implementation for IVehicleCurrentLocationGnssReceiver should provide mechanism for subscription of the IVehicleCurrentLocationGnssReceiverSubscriber clients.
 * See IVehicleCurrentLocationGnssReceiverPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleCurrentLocationGnssReceiver should call the IVehicleCurrentLocationGnssReceiverSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleCurrentLocationGnssReceiverSubscriber class to implement clients of the IVehicleCurrentLocationGnssReceiver or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocationGnssReceiverSubscriber
{
public:
    virtual ~IVehicleCurrentLocationGnssReceiverSubscriber() = default;
    /**
    * Called by the IVehicleCurrentLocationGnssReceiverPublisher when FixType value has changed if subscribed for the FixType change.
    * @param FixType Fix status of GNSS receiver.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onFixTypeChanged(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType) = 0;
};

/** Callback for changes of FixType */
using VehicleCurrentLocationGnssReceiverFixTypePropertyCb = std::function<void(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType)>;


/**
 * The IVehicleCurrentLocationGnssReceiverPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleCurrentLocationGnssReceiver implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleCurrentLocationGnssReceiverSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleCurrentLocationGnssReceiver on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocationGnssReceiverPublisher
{
public:
    virtual ~IVehicleCurrentLocationGnssReceiverPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_CurrentLocation_GNSSReceiver.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleCurrentLocationGnssReceiverSubscriber which is subscribed in this function to any change of the Vehicle_CurrentLocation_GNSSReceiver.
    */
    virtual void subscribeToAllChanges(IVehicleCurrentLocationGnssReceiverSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_CurrentLocation_GNSSReceiver.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleCurrentLocationGnssReceiverSubscriber which subscription for any change of the Vehicle_CurrentLocation_GNSSReceiver is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleCurrentLocationGnssReceiverSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for FixType value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationGnssReceiverSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationGnssReceiverFixTypePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToFixTypeChanged(VehicleCurrentLocationGnssReceiverFixTypePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from FixType property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationGnssReceiverSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromFixTypeChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation_GNSSReceiver implementation when property FixType changes.
    * @param The new value of FixType.
    * Fix status of GNSS receiver.
    */
    virtual void publishFixTypeChanged(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
