#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodyLightsBrakeSubscriber;
class IVehicleBodyLightsBrakePublisher;

/**
*
* IVehicleBodyLightsBrake provides an interface for
 *  - methods defined for your Vehicle_Body_Lights_Brake 
 *  - property setters and getters for defined properties
 * The IVehicleBodyLightsBrake also provides an interface to access a publisher IVehicleBodyLightsBrakePublisher, a class used by IVehicleBodyLightsBrakeSubscriber clients.
 * The implementation should notify the publisher IVehicleBodyLightsBrakePublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodyLightsBrakeSubscriber, IVehicleBodyLightsBrakePublisher
 * and the example implementation Vehicle_Body_Lights_Brake  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBrake
{
public:
    virtual ~IVehicleBodyLightsBrake() = default;

    /**
    * Sets the value of the IsActive property.
    * @param IsActive Indicates if break-light is active. INACTIVE means lights are off. ACTIVE means lights are on. ADAPTIVE means that break-light is indicating emergency-breaking.
    */
    virtual void setIsActive(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive) = 0;
    /**
    * Gets the value of the IsActive property.
    * @return Indicates if break-light is active. INACTIVE means lights are off. ACTIVE means lights are on. ADAPTIVE means that break-light is indicating emergency-breaking.
    */
    virtual Vehicle_Body_Lights_Brake_IsActive_ValueEnum getIsActive() const = 0;

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
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Brake changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body_Lights_Brake.
    */
    virtual IVehicleBodyLightsBrakePublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodyLightsBrakeSubscriber contains functions to allow informing about signals or property changes of the IVehicleBodyLightsBrake implementation.
 * The implementation for IVehicleBodyLightsBrake should provide mechanism for subscription of the IVehicleBodyLightsBrakeSubscriber clients.
 * See IVehicleBodyLightsBrakePublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBodyLightsBrake should call the IVehicleBodyLightsBrakeSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodyLightsBrakeSubscriber class to implement clients of the IVehicleBodyLightsBrake or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBrakeSubscriber
{
public:
    virtual ~IVehicleBodyLightsBrakeSubscriber() = default;
    /**
    * Called by the IVehicleBodyLightsBrakePublisher when IsActive value has changed if subscribed for the IsActive change.
    * @param IsActive Indicates if break-light is active. INACTIVE means lights are off. ACTIVE means lights are on. ADAPTIVE means that break-light is indicating emergency-breaking.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsActiveChanged(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive) = 0;
    /**
    * Called by the IVehicleBodyLightsBrakePublisher when IsDefect value has changed if subscribed for the IsDefect change.
    * @param IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsDefectChanged(bool IsDefect) = 0;
};

/** Callback for changes of IsActive */
using VehicleBodyLightsBrakeIsActivePropertyCb = std::function<void(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive)>;
/** Callback for changes of IsDefect */
using VehicleBodyLightsBrakeIsDefectPropertyCb = std::function<void(bool IsDefect)>;


/**
 * The IVehicleBodyLightsBrakePublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBodyLightsBrake implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodyLightsBrakeSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBodyLightsBrake on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBrakePublisher
{
public:
    virtual ~IVehicleBodyLightsBrakePublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body_Lights_Brake.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodyLightsBrakeSubscriber which is subscribed in this function to any change of the Vehicle_Body_Lights_Brake.
    */
    virtual void subscribeToAllChanges(IVehicleBodyLightsBrakeSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body_Lights_Brake.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodyLightsBrakeSubscriber which subscription for any change of the Vehicle_Body_Lights_Brake is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodyLightsBrakeSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsActive value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBrakeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsBrakeIsActivePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsActiveChanged(VehicleBodyLightsBrakeIsActivePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsActive property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBrakeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsActiveChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsDefect value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBrakeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsBrakeIsDefectPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsDefectChanged(VehicleBodyLightsBrakeIsDefectPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsDefect property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBrakeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsDefectChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Brake implementation when property IsActive changes.
    * @param The new value of IsActive.
    * Indicates if break-light is active. INACTIVE means lights are off. ACTIVE means lights are on. ADAPTIVE means that break-light is indicating emergency-breaking.
    */
    virtual void publishIsActiveChanged(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Brake implementation when property IsDefect changes.
    * @param The new value of IsDefect.
    * Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual void publishIsDefectChanged(bool IsDefect) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
