#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleAngularVelocitySubscriber;
class IVehicleAngularVelocityPublisher;

/**
*
* IVehicleAngularVelocity provides an interface for
 *  - methods defined for your Vehicle_AngularVelocity 
 *  - property setters and getters for defined properties
 * The IVehicleAngularVelocity also provides an interface to access a publisher IVehicleAngularVelocityPublisher, a class used by IVehicleAngularVelocitySubscriber clients.
 * The implementation should notify the publisher IVehicleAngularVelocityPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleAngularVelocitySubscriber, IVehicleAngularVelocityPublisher
 * and the example implementation Vehicle_AngularVelocity  or the
 */
class CPP_VEHICLE_EXPORT IVehicleAngularVelocity
{
public:
    virtual ~IVehicleAngularVelocity() = default;

    /**
    * Sets the value of the Roll property.
    * @param Roll Vehicle rotation rate along X (longitudinal).
    */
    virtual void setRoll(float Roll) = 0;
    /**
    * Gets the value of the Roll property.
    * @return Vehicle rotation rate along X (longitudinal).
    */
    virtual float getRoll() const = 0;

    /**
    * Sets the value of the Pitch property.
    * @param Pitch Vehicle rotation rate along Y (lateral).
    */
    virtual void setPitch(float Pitch) = 0;
    /**
    * Gets the value of the Pitch property.
    * @return Vehicle rotation rate along Y (lateral).
    */
    virtual float getPitch() const = 0;

    /**
    * Sets the value of the Yaw property.
    * @param Yaw Vehicle rotation rate along Z (vertical).
    */
    virtual void setYaw(float Yaw) = 0;
    /**
    * Gets the value of the Yaw property.
    * @return Vehicle rotation rate along Z (vertical).
    */
    virtual float getYaw() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_AngularVelocity changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_AngularVelocity.
    */
    virtual IVehicleAngularVelocityPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleAngularVelocitySubscriber contains functions to allow informing about signals or property changes of the IVehicleAngularVelocity implementation.
 * The implementation for IVehicleAngularVelocity should provide mechanism for subscription of the IVehicleAngularVelocitySubscriber clients.
 * See IVehicleAngularVelocityPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleAngularVelocity should call the IVehicleAngularVelocitySubscriber interface functions on either signal emit or property change.
 * You can use IVehicleAngularVelocitySubscriber class to implement clients of the IVehicleAngularVelocity or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleAngularVelocitySubscriber
{
public:
    virtual ~IVehicleAngularVelocitySubscriber() = default;
    /**
    * Called by the IVehicleAngularVelocityPublisher when Roll value has changed if subscribed for the Roll change.
    * @param Roll Vehicle rotation rate along X (longitudinal).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRollChanged(float Roll) = 0;
    /**
    * Called by the IVehicleAngularVelocityPublisher when Pitch value has changed if subscribed for the Pitch change.
    * @param Pitch Vehicle rotation rate along Y (lateral).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPitchChanged(float Pitch) = 0;
    /**
    * Called by the IVehicleAngularVelocityPublisher when Yaw value has changed if subscribed for the Yaw change.
    * @param Yaw Vehicle rotation rate along Z (vertical).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onYawChanged(float Yaw) = 0;
};

/** Callback for changes of Roll */
using VehicleAngularVelocityRollPropertyCb = std::function<void(float Roll)>;
/** Callback for changes of Pitch */
using VehicleAngularVelocityPitchPropertyCb = std::function<void(float Pitch)>;
/** Callback for changes of Yaw */
using VehicleAngularVelocityYawPropertyCb = std::function<void(float Yaw)>;


/**
 * The IVehicleAngularVelocityPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleAngularVelocity implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleAngularVelocitySubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleAngularVelocity on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleAngularVelocityPublisher
{
public:
    virtual ~IVehicleAngularVelocityPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_AngularVelocity.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleAngularVelocitySubscriber which is subscribed in this function to any change of the Vehicle_AngularVelocity.
    */
    virtual void subscribeToAllChanges(IVehicleAngularVelocitySubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_AngularVelocity.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleAngularVelocitySubscriber which subscription for any change of the Vehicle_AngularVelocity is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleAngularVelocitySubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Roll value changes.
    * If your subscriber uses subscription with IVehicleAngularVelocitySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleAngularVelocityRollPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRollChanged(VehicleAngularVelocityRollPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Roll property changes.
    * If your subscriber uses subscription with IVehicleAngularVelocitySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRollChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Pitch value changes.
    * If your subscriber uses subscription with IVehicleAngularVelocitySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleAngularVelocityPitchPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPitchChanged(VehicleAngularVelocityPitchPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Pitch property changes.
    * If your subscriber uses subscription with IVehicleAngularVelocitySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPitchChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Yaw value changes.
    * If your subscriber uses subscription with IVehicleAngularVelocitySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleAngularVelocityYawPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToYawChanged(VehicleAngularVelocityYawPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Yaw property changes.
    * If your subscriber uses subscription with IVehicleAngularVelocitySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromYawChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_AngularVelocity implementation when property Roll changes.
    * @param The new value of Roll.
    * Vehicle rotation rate along X (longitudinal).
    */
    virtual void publishRollChanged(float Roll) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_AngularVelocity implementation when property Pitch changes.
    * @param The new value of Pitch.
    * Vehicle rotation rate along Y (lateral).
    */
    virtual void publishPitchChanged(float Pitch) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_AngularVelocity implementation when property Yaw changes.
    * @param The new value of Yaw.
    * Vehicle rotation rate along Z (vertical).
    */
    virtual void publishYawChanged(float Yaw) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
