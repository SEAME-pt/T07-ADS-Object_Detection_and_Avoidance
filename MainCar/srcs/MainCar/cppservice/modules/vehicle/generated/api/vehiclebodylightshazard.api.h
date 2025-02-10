#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodyLightsHazardSubscriber;
class IVehicleBodyLightsHazardPublisher;

/**
*
* IVehicleBodyLightsHazard provides an interface for
 *  - methods defined for your Vehicle_Body_Lights_Hazard 
 *  - property setters and getters for defined properties
 * The IVehicleBodyLightsHazard also provides an interface to access a publisher IVehicleBodyLightsHazardPublisher, a class used by IVehicleBodyLightsHazardSubscriber clients.
 * The implementation should notify the publisher IVehicleBodyLightsHazardPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodyLightsHazardSubscriber, IVehicleBodyLightsHazardPublisher
 * and the example implementation Vehicle_Body_Lights_Hazard  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsHazard
{
public:
    virtual ~IVehicleBodyLightsHazard() = default;

    /**
    * Sets the value of the IsSignaling property.
    * @param IsSignaling Indicates if light is signaling or off. True = signaling. False = Off.
    */
    virtual void setIsSignaling(bool IsSignaling) = 0;
    /**
    * Gets the value of the IsSignaling property.
    * @return Indicates if light is signaling or off. True = signaling. False = Off.
    */
    virtual bool getIsSignaling() const = 0;

    /**
    * Sets the value of the IsDefect property.
    * @param IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual void setIsDefect(bool IsDefect) = 0;
    /**
    * Gets the value of the IsDefect property.
    * @return Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual bool getIsDefect() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Hazard changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body_Lights_Hazard.
    */
    virtual IVehicleBodyLightsHazardPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodyLightsHazardSubscriber contains functions to allow informing about signals or property changes of the IVehicleBodyLightsHazard implementation.
 * The implementation for IVehicleBodyLightsHazard should provide mechanism for subscription of the IVehicleBodyLightsHazardSubscriber clients.
 * See IVehicleBodyLightsHazardPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBodyLightsHazard should call the IVehicleBodyLightsHazardSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodyLightsHazardSubscriber class to implement clients of the IVehicleBodyLightsHazard or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsHazardSubscriber
{
public:
    virtual ~IVehicleBodyLightsHazardSubscriber() = default;
    /**
    * Called by the IVehicleBodyLightsHazardPublisher when IsSignaling value has changed if subscribed for the IsSignaling change.
    * @param IsSignaling Indicates if light is signaling or off. True = signaling. False = Off.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsSignalingChanged(bool IsSignaling) = 0;
    /**
    * Called by the IVehicleBodyLightsHazardPublisher when IsDefect value has changed if subscribed for the IsDefect change.
    * @param IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsDefectChanged(bool IsDefect) = 0;
};

/** Callback for changes of IsSignaling */
using VehicleBodyLightsHazardIsSignalingPropertyCb = std::function<void(bool IsSignaling)>;
/** Callback for changes of IsDefect */
using VehicleBodyLightsHazardIsDefectPropertyCb = std::function<void(bool IsDefect)>;


/**
 * The IVehicleBodyLightsHazardPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBodyLightsHazard implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodyLightsHazardSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBodyLightsHazard on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsHazardPublisher
{
public:
    virtual ~IVehicleBodyLightsHazardPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body_Lights_Hazard.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodyLightsHazardSubscriber which is subscribed in this function to any change of the Vehicle_Body_Lights_Hazard.
    */
    virtual void subscribeToAllChanges(IVehicleBodyLightsHazardSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body_Lights_Hazard.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodyLightsHazardSubscriber which subscription for any change of the Vehicle_Body_Lights_Hazard is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodyLightsHazardSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsSignaling value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsHazardSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsHazardIsSignalingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsSignalingChanged(VehicleBodyLightsHazardIsSignalingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsSignaling property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsHazardSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsSignalingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsDefect value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsHazardSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsHazardIsDefectPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsDefectChanged(VehicleBodyLightsHazardIsDefectPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsDefect property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsHazardSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsDefectChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Hazard implementation when property IsSignaling changes.
    * @param The new value of IsSignaling.
    * Indicates if light is signaling or off. True = signaling. False = Off.
    */
    virtual void publishIsSignalingChanged(bool IsSignaling) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Hazard implementation when property IsDefect changes.
    * @param The new value of IsDefect.
    * Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual void publishIsDefectChanged(bool IsDefect) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
