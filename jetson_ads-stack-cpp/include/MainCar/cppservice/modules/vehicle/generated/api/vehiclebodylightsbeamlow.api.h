#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodyLightsBeamLowSubscriber;
class IVehicleBodyLightsBeamLowPublisher;

/**
*
* IVehicleBodyLightsBeamLow provides an interface for
 *  - methods defined for your Vehicle_Body_Lights_Beam_Low 
 *  - property setters and getters for defined properties
 * The IVehicleBodyLightsBeamLow also provides an interface to access a publisher IVehicleBodyLightsBeamLowPublisher, a class used by IVehicleBodyLightsBeamLowSubscriber clients.
 * The implementation should notify the publisher IVehicleBodyLightsBeamLowPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodyLightsBeamLowSubscriber, IVehicleBodyLightsBeamLowPublisher
 * and the example implementation Vehicle_Body_Lights_Beam_Low  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBeamLow
{
public:
    virtual ~IVehicleBodyLightsBeamLow() = default;

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
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Beam_Low changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body_Lights_Beam_Low.
    */
    virtual IVehicleBodyLightsBeamLowPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodyLightsBeamLowSubscriber contains functions to allow informing about signals or property changes of the IVehicleBodyLightsBeamLow implementation.
 * The implementation for IVehicleBodyLightsBeamLow should provide mechanism for subscription of the IVehicleBodyLightsBeamLowSubscriber clients.
 * See IVehicleBodyLightsBeamLowPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBodyLightsBeamLow should call the IVehicleBodyLightsBeamLowSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodyLightsBeamLowSubscriber class to implement clients of the IVehicleBodyLightsBeamLow or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBeamLowSubscriber
{
public:
    virtual ~IVehicleBodyLightsBeamLowSubscriber() = default;
    /**
    * Called by the IVehicleBodyLightsBeamLowPublisher when IsOn value has changed if subscribed for the IsOn change.
    * @param IsOn Indicates if light is on or off. True = On. False = Off.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsOnChanged(bool IsOn) = 0;
    /**
    * Called by the IVehicleBodyLightsBeamLowPublisher when IsDefect value has changed if subscribed for the IsDefect change.
    * @param IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsDefectChanged(bool IsDefect) = 0;
};

/** Callback for changes of IsOn */
using VehicleBodyLightsBeamLowIsOnPropertyCb = std::function<void(bool IsOn)>;
/** Callback for changes of IsDefect */
using VehicleBodyLightsBeamLowIsDefectPropertyCb = std::function<void(bool IsDefect)>;


/**
 * The IVehicleBodyLightsBeamLowPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBodyLightsBeamLow implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodyLightsBeamLowSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBodyLightsBeamLow on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBeamLowPublisher
{
public:
    virtual ~IVehicleBodyLightsBeamLowPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body_Lights_Beam_Low.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodyLightsBeamLowSubscriber which is subscribed in this function to any change of the Vehicle_Body_Lights_Beam_Low.
    */
    virtual void subscribeToAllChanges(IVehicleBodyLightsBeamLowSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body_Lights_Beam_Low.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodyLightsBeamLowSubscriber which subscription for any change of the Vehicle_Body_Lights_Beam_Low is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodyLightsBeamLowSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsOn value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBeamLowSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsBeamLowIsOnPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsOnChanged(VehicleBodyLightsBeamLowIsOnPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsOn property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBeamLowSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsOnChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsDefect value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBeamLowSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsBeamLowIsDefectPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsDefectChanged(VehicleBodyLightsBeamLowIsDefectPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsDefect property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBeamLowSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsDefectChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Beam_Low implementation when property IsOn changes.
    * @param The new value of IsOn.
    * Indicates if light is on or off. True = On. False = Off.
    */
    virtual void publishIsOnChanged(bool IsOn) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Beam_Low implementation when property IsDefect changes.
    * @param The new value of IsDefect.
    * Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual void publishIsDefectChanged(bool IsDefect) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
