#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodyLightsFogFrontSubscriber;
class IVehicleBodyLightsFogFrontPublisher;

/**
*
* IVehicleBodyLightsFogFront provides an interface for
 *  - methods defined for your Vehicle_Body_Lights_Fog_Front 
 *  - property setters and getters for defined properties
 * The IVehicleBodyLightsFogFront also provides an interface to access a publisher IVehicleBodyLightsFogFrontPublisher, a class used by IVehicleBodyLightsFogFrontSubscriber clients.
 * The implementation should notify the publisher IVehicleBodyLightsFogFrontPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodyLightsFogFrontSubscriber, IVehicleBodyLightsFogFrontPublisher
 * and the example implementation Vehicle_Body_Lights_Fog_Front  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsFogFront
{
public:
    virtual ~IVehicleBodyLightsFogFront() = default;

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
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Fog_Front changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body_Lights_Fog_Front.
    */
    virtual IVehicleBodyLightsFogFrontPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodyLightsFogFrontSubscriber contains functions to allow informing about signals or property changes of the IVehicleBodyLightsFogFront implementation.
 * The implementation for IVehicleBodyLightsFogFront should provide mechanism for subscription of the IVehicleBodyLightsFogFrontSubscriber clients.
 * See IVehicleBodyLightsFogFrontPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBodyLightsFogFront should call the IVehicleBodyLightsFogFrontSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodyLightsFogFrontSubscriber class to implement clients of the IVehicleBodyLightsFogFront or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsFogFrontSubscriber
{
public:
    virtual ~IVehicleBodyLightsFogFrontSubscriber() = default;
    /**
    * Called by the IVehicleBodyLightsFogFrontPublisher when IsOn value has changed if subscribed for the IsOn change.
    * @param IsOn Indicates if light is on or off. True = On. False = Off.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsOnChanged(bool IsOn) = 0;
    /**
    * Called by the IVehicleBodyLightsFogFrontPublisher when IsDefect value has changed if subscribed for the IsDefect change.
    * @param IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsDefectChanged(bool IsDefect) = 0;
};

/** Callback for changes of IsOn */
using VehicleBodyLightsFogFrontIsOnPropertyCb = std::function<void(bool IsOn)>;
/** Callback for changes of IsDefect */
using VehicleBodyLightsFogFrontIsDefectPropertyCb = std::function<void(bool IsDefect)>;


/**
 * The IVehicleBodyLightsFogFrontPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBodyLightsFogFront implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodyLightsFogFrontSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBodyLightsFogFront on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsFogFrontPublisher
{
public:
    virtual ~IVehicleBodyLightsFogFrontPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body_Lights_Fog_Front.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodyLightsFogFrontSubscriber which is subscribed in this function to any change of the Vehicle_Body_Lights_Fog_Front.
    */
    virtual void subscribeToAllChanges(IVehicleBodyLightsFogFrontSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body_Lights_Fog_Front.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodyLightsFogFrontSubscriber which subscription for any change of the Vehicle_Body_Lights_Fog_Front is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodyLightsFogFrontSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsOn value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsFogFrontSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsFogFrontIsOnPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsOnChanged(VehicleBodyLightsFogFrontIsOnPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsOn property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsFogFrontSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsOnChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsDefect value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsFogFrontSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsFogFrontIsDefectPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsDefectChanged(VehicleBodyLightsFogFrontIsDefectPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsDefect property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsFogFrontSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsDefectChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Fog_Front implementation when property IsOn changes.
    * @param The new value of IsOn.
    * Indicates if light is on or off. True = On. False = Off.
    */
    virtual void publishIsOnChanged(bool IsOn) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Fog_Front implementation when property IsDefect changes.
    * @param The new value of IsDefect.
    * Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual void publishIsDefectChanged(bool IsDefect) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
