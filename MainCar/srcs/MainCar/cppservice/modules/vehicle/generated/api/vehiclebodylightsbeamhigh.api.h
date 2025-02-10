#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodyLightsBeamHighSubscriber;
class IVehicleBodyLightsBeamHighPublisher;

/**
*
* IVehicleBodyLightsBeamHigh provides an interface for
 *  - methods defined for your Vehicle_Body_Lights_Beam_High 
 *  - property setters and getters for defined properties
 * The IVehicleBodyLightsBeamHigh also provides an interface to access a publisher IVehicleBodyLightsBeamHighPublisher, a class used by IVehicleBodyLightsBeamHighSubscriber clients.
 * The implementation should notify the publisher IVehicleBodyLightsBeamHighPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodyLightsBeamHighSubscriber, IVehicleBodyLightsBeamHighPublisher
 * and the example implementation Vehicle_Body_Lights_Beam_High  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBeamHigh
{
public:
    virtual ~IVehicleBodyLightsBeamHigh() = default;

    /**
    * Sets the value of the IsOn property.
    * @param IsOn Indicates if light is on or off. True = On. False = Off.
    */
    virtual void setIsOn(bool IsOn) = 0;
    /**
    * Gets the value of the IsOn property.
    * @return Indicates if light is on or off. True = On. False = Off.
    */
    virtual bool getIsOn() const = 0;

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
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Beam_High changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body_Lights_Beam_High.
    */
    virtual IVehicleBodyLightsBeamHighPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodyLightsBeamHighSubscriber contains functions to allow informing about signals or property changes of the IVehicleBodyLightsBeamHigh implementation.
 * The implementation for IVehicleBodyLightsBeamHigh should provide mechanism for subscription of the IVehicleBodyLightsBeamHighSubscriber clients.
 * See IVehicleBodyLightsBeamHighPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBodyLightsBeamHigh should call the IVehicleBodyLightsBeamHighSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodyLightsBeamHighSubscriber class to implement clients of the IVehicleBodyLightsBeamHigh or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBeamHighSubscriber
{
public:
    virtual ~IVehicleBodyLightsBeamHighSubscriber() = default;
    /**
    * Called by the IVehicleBodyLightsBeamHighPublisher when IsOn value has changed if subscribed for the IsOn change.
    * @param IsOn Indicates if light is on or off. True = On. False = Off.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsOnChanged(bool IsOn) = 0;
    /**
    * Called by the IVehicleBodyLightsBeamHighPublisher when IsDefect value has changed if subscribed for the IsDefect change.
    * @param IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsDefectChanged(bool IsDefect) = 0;
};

/** Callback for changes of IsOn */
using VehicleBodyLightsBeamHighIsOnPropertyCb = std::function<void(bool IsOn)>;
/** Callback for changes of IsDefect */
using VehicleBodyLightsBeamHighIsDefectPropertyCb = std::function<void(bool IsDefect)>;


/**
 * The IVehicleBodyLightsBeamHighPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBodyLightsBeamHigh implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodyLightsBeamHighSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBodyLightsBeamHigh on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBeamHighPublisher
{
public:
    virtual ~IVehicleBodyLightsBeamHighPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body_Lights_Beam_High.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodyLightsBeamHighSubscriber which is subscribed in this function to any change of the Vehicle_Body_Lights_Beam_High.
    */
    virtual void subscribeToAllChanges(IVehicleBodyLightsBeamHighSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body_Lights_Beam_High.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodyLightsBeamHighSubscriber which subscription for any change of the Vehicle_Body_Lights_Beam_High is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodyLightsBeamHighSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsOn value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBeamHighSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsBeamHighIsOnPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsOnChanged(VehicleBodyLightsBeamHighIsOnPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsOn property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBeamHighSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsOnChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsDefect value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBeamHighSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsBeamHighIsDefectPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsDefectChanged(VehicleBodyLightsBeamHighIsDefectPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsDefect property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBeamHighSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsDefectChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Beam_High implementation when property IsOn changes.
    * @param The new value of IsOn.
    * Indicates if light is on or off. True = On. False = Off.
    */
    virtual void publishIsOnChanged(bool IsOn) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Beam_High implementation when property IsDefect changes.
    * @param The new value of IsDefect.
    * Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual void publishIsDefectChanged(bool IsDefect) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
