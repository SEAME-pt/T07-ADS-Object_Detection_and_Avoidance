#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodyLightsSubscriber;
class IVehicleBodyLightsPublisher;

/**
*
* IVehicleBodyLights provides an interface for
 *  - methods defined for your Vehicle_Body_Lights 
 *  - property setters and getters for defined properties
 * The IVehicleBodyLights also provides an interface to access a publisher IVehicleBodyLightsPublisher, a class used by IVehicleBodyLightsSubscriber clients.
 * The implementation should notify the publisher IVehicleBodyLightsPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodyLightsSubscriber, IVehicleBodyLightsPublisher
 * and the example implementation Vehicle_Body_Lights  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLights
{
public:
    virtual ~IVehicleBodyLights() = default;

    /**
    * Sets the value of the LightSwitch property.
    * @param LightSwitch Status of the vehicle main light switch.
    */
    virtual void setLightSwitch(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch) = 0;
    /**
    * Gets the value of the LightSwitch property.
    * @return Status of the vehicle main light switch.
    */
    virtual Vehicle_Body_Lights_LightSwitch_ValueEnum getLightSwitch() const = 0;

    /**
    * Sets the value of the IsHighBeamSwitchOn property.
    * @param IsHighBeamSwitchOn Status of the high beam switch. True = high beam enabled. False = high beam not enabled.
    */
    virtual void setIsHighBeamSwitchOn(bool IsHighBeamSwitchOn) = 0;
    /**
    * Gets the value of the IsHighBeamSwitchOn property.
    * @return Status of the high beam switch. True = high beam enabled. False = high beam not enabled.
    */
    virtual bool getIsHighBeamSwitchOn() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body_Lights.
    */
    virtual IVehicleBodyLightsPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodyLightsSubscriber contains functions to allow informing about signals or property changes of the IVehicleBodyLights implementation.
 * The implementation for IVehicleBodyLights should provide mechanism for subscription of the IVehicleBodyLightsSubscriber clients.
 * See IVehicleBodyLightsPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBodyLights should call the IVehicleBodyLightsSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodyLightsSubscriber class to implement clients of the IVehicleBodyLights or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsSubscriber
{
public:
    virtual ~IVehicleBodyLightsSubscriber() = default;
    /**
    * Called by the IVehicleBodyLightsPublisher when LightSwitch value has changed if subscribed for the LightSwitch change.
    * @param LightSwitch Status of the vehicle main light switch.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLightSwitchChanged(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch) = 0;
    /**
    * Called by the IVehicleBodyLightsPublisher when IsHighBeamSwitchOn value has changed if subscribed for the IsHighBeamSwitchOn change.
    * @param IsHighBeamSwitchOn Status of the high beam switch. True = high beam enabled. False = high beam not enabled.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsHighBeamSwitchOnChanged(bool IsHighBeamSwitchOn) = 0;
};

/** Callback for changes of LightSwitch */
using VehicleBodyLightsLightSwitchPropertyCb = std::function<void(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch)>;
/** Callback for changes of IsHighBeamSwitchOn */
using VehicleBodyLightsIsHighBeamSwitchOnPropertyCb = std::function<void(bool IsHighBeamSwitchOn)>;


/**
 * The IVehicleBodyLightsPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBodyLights implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodyLightsSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBodyLights on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsPublisher
{
public:
    virtual ~IVehicleBodyLightsPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body_Lights.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodyLightsSubscriber which is subscribed in this function to any change of the Vehicle_Body_Lights.
    */
    virtual void subscribeToAllChanges(IVehicleBodyLightsSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body_Lights.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodyLightsSubscriber which subscription for any change of the Vehicle_Body_Lights is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodyLightsSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for LightSwitch value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsLightSwitchPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLightSwitchChanged(VehicleBodyLightsLightSwitchPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from LightSwitch property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLightSwitchChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsHighBeamSwitchOn value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsIsHighBeamSwitchOnPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsHighBeamSwitchOnChanged(VehicleBodyLightsIsHighBeamSwitchOnPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsHighBeamSwitchOn property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsHighBeamSwitchOnChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights implementation when property LightSwitch changes.
    * @param The new value of LightSwitch.
    * Status of the vehicle main light switch.
    */
    virtual void publishLightSwitchChanged(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights implementation when property IsHighBeamSwitchOn changes.
    * @param The new value of IsHighBeamSwitchOn.
    * Status of the high beam switch. True = high beam enabled. False = high beam not enabled.
    */
    virtual void publishIsHighBeamSwitchOnChanged(bool IsHighBeamSwitchOn) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
