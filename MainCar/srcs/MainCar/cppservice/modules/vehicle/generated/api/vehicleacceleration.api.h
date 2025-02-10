#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleAccelerationSubscriber;
class IVehicleAccelerationPublisher;

/**
*
* IVehicleAcceleration provides an interface for
 *  - methods defined for your Vehicle_Acceleration 
 *  - property setters and getters for defined properties
 * The IVehicleAcceleration also provides an interface to access a publisher IVehicleAccelerationPublisher, a class used by IVehicleAccelerationSubscriber clients.
 * The implementation should notify the publisher IVehicleAccelerationPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleAccelerationSubscriber, IVehicleAccelerationPublisher
 * and the example implementation Vehicle_Acceleration  or the
 */
class CPP_VEHICLE_EXPORT IVehicleAcceleration
{
public:
    virtual ~IVehicleAcceleration() = default;

    /**
    * Sets the value of the Longitudinal property.
    * @param Longitudinal Vehicle acceleration in X (longitudinal acceleration).
    */
    virtual void setLongitudinal(float Longitudinal) = 0;
    /**
    * Gets the value of the Longitudinal property.
    * @return Vehicle acceleration in X (longitudinal acceleration).
    */
    virtual float getLongitudinal() const = 0;

    /**
    * Sets the value of the Lateral property.
    * @param Lateral Vehicle acceleration in Y (lateral acceleration).
    */
    virtual void setLateral(float Lateral) = 0;
    /**
    * Gets the value of the Lateral property.
    * @return Vehicle acceleration in Y (lateral acceleration).
    */
    virtual float getLateral() const = 0;

    /**
    * Sets the value of the Vertical property.
    * @param Vertical Vehicle acceleration in Z (vertical acceleration).
    */
    virtual void setVertical(float Vertical) = 0;
    /**
    * Gets the value of the Vertical property.
    * @return Vehicle acceleration in Z (vertical acceleration).
    */
    virtual float getVertical() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Acceleration changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Acceleration.
    */
    virtual IVehicleAccelerationPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleAccelerationSubscriber contains functions to allow informing about signals or property changes of the IVehicleAcceleration implementation.
 * The implementation for IVehicleAcceleration should provide mechanism for subscription of the IVehicleAccelerationSubscriber clients.
 * See IVehicleAccelerationPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleAcceleration should call the IVehicleAccelerationSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleAccelerationSubscriber class to implement clients of the IVehicleAcceleration or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleAccelerationSubscriber
{
public:
    virtual ~IVehicleAccelerationSubscriber() = default;
    /**
    * Called by the IVehicleAccelerationPublisher when Longitudinal value has changed if subscribed for the Longitudinal change.
    * @param Longitudinal Vehicle acceleration in X (longitudinal acceleration).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLongitudinalChanged(float Longitudinal) = 0;
    /**
    * Called by the IVehicleAccelerationPublisher when Lateral value has changed if subscribed for the Lateral change.
    * @param Lateral Vehicle acceleration in Y (lateral acceleration).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLateralChanged(float Lateral) = 0;
    /**
    * Called by the IVehicleAccelerationPublisher when Vertical value has changed if subscribed for the Vertical change.
    * @param Vertical Vehicle acceleration in Z (vertical acceleration).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVerticalChanged(float Vertical) = 0;
};

/** Callback for changes of Longitudinal */
using VehicleAccelerationLongitudinalPropertyCb = std::function<void(float Longitudinal)>;
/** Callback for changes of Lateral */
using VehicleAccelerationLateralPropertyCb = std::function<void(float Lateral)>;
/** Callback for changes of Vertical */
using VehicleAccelerationVerticalPropertyCb = std::function<void(float Vertical)>;


/**
 * The IVehicleAccelerationPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleAcceleration implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleAccelerationSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleAcceleration on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleAccelerationPublisher
{
public:
    virtual ~IVehicleAccelerationPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Acceleration.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleAccelerationSubscriber which is subscribed in this function to any change of the Vehicle_Acceleration.
    */
    virtual void subscribeToAllChanges(IVehicleAccelerationSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Acceleration.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleAccelerationSubscriber which subscription for any change of the Vehicle_Acceleration is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleAccelerationSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Longitudinal value changes.
    * If your subscriber uses subscription with IVehicleAccelerationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleAccelerationLongitudinalPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLongitudinalChanged(VehicleAccelerationLongitudinalPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Longitudinal property changes.
    * If your subscriber uses subscription with IVehicleAccelerationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLongitudinalChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Lateral value changes.
    * If your subscriber uses subscription with IVehicleAccelerationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleAccelerationLateralPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLateralChanged(VehicleAccelerationLateralPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Lateral property changes.
    * If your subscriber uses subscription with IVehicleAccelerationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLateralChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Vertical value changes.
    * If your subscriber uses subscription with IVehicleAccelerationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleAccelerationVerticalPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVerticalChanged(VehicleAccelerationVerticalPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Vertical property changes.
    * If your subscriber uses subscription with IVehicleAccelerationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVerticalChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Acceleration implementation when property Longitudinal changes.
    * @param The new value of Longitudinal.
    * Vehicle acceleration in X (longitudinal acceleration).
    */
    virtual void publishLongitudinalChanged(float Longitudinal) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Acceleration implementation when property Lateral changes.
    * @param The new value of Lateral.
    * Vehicle acceleration in Y (lateral acceleration).
    */
    virtual void publishLateralChanged(float Lateral) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Acceleration implementation when property Vertical changes.
    * @param The new value of Vertical.
    * Vehicle acceleration in Z (vertical acceleration).
    */
    virtual void publishVerticalChanged(float Vertical) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
