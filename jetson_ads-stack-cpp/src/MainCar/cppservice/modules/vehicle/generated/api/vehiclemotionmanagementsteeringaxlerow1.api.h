#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleMotionManagementSteeringAxleRow1Subscriber;
class IVehicleMotionManagementSteeringAxleRow1Publisher;

/**
*
* IVehicleMotionManagementSteeringAxleRow1 provides an interface for
 *  - methods defined for your Vehicle_MotionManagement_Steering_Axle_Row1 
 *  - property setters and getters for defined properties
 * The IVehicleMotionManagementSteeringAxleRow1 also provides an interface to access a publisher IVehicleMotionManagementSteeringAxleRow1Publisher, a class used by IVehicleMotionManagementSteeringAxleRow1Subscriber clients.
 * The implementation should notify the publisher IVehicleMotionManagementSteeringAxleRow1Publisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleMotionManagementSteeringAxleRow1Subscriber, IVehicleMotionManagementSteeringAxleRow1Publisher
 * and the example implementation Vehicle_MotionManagement_Steering_Axle_Row1  or the
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementSteeringAxleRow1
{
public:
    virtual ~IVehicleMotionManagementSteeringAxleRow1() = default;

    /**
    * Sets the value of the RackPositionOffsetTarget property.
    * @param RackPositionOffsetTarget Rack position offset request to the axle steering actuator (for steer-by-wire), added to the actuator internal calculated set-point. Positive values without internal calculated set point change lead to a left movement of the vehicle (based on ISO8855).
    */
    virtual void setRackPositionOffsetTarget(int32_t RackPositionOffsetTarget) = 0;
    /**
    * Gets the value of the RackPositionOffsetTarget property.
    * @return Rack position offset request to the axle steering actuator (for steer-by-wire), added to the actuator internal calculated set-point. Positive values without internal calculated set point change lead to a left movement of the vehicle (based on ISO8855).
    */
    virtual int32_t getRackPositionOffsetTarget() const = 0;

    /**
    * Sets the value of the RackPositionOffsetTargetMode property.
    * @param RackPositionOffsetTargetMode Mode used for controlling rack position offset interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void setRackPositionOffsetTargetMode(int32_t RackPositionOffsetTargetMode) = 0;
    /**
    * Gets the value of the RackPositionOffsetTargetMode property.
    * @return Mode used for controlling rack position offset interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual int32_t getRackPositionOffsetTargetMode() const = 0;

    /**
    * Sets the value of the RackPosition property.
    * @param RackPosition Represents the current position of the steering rack on axle steering actuator. Positive values leads to a left turn of the vehicle (based on ISO8855).
    */
    virtual void setRackPosition(int32_t RackPosition) = 0;
    /**
    * Gets the value of the RackPosition property.
    * @return Represents the current position of the steering rack on axle steering actuator. Positive values leads to a left turn of the vehicle (based on ISO8855).
    */
    virtual int32_t getRackPosition() const = 0;

    /**
    * Sets the value of the RackPositionTarget property.
    * @param RackPositionTarget Rack position request to the axle steering actuator (external set-point). Positive values lead to a left turn of the vehicle (based on ISO8855).
    */
    virtual void setRackPositionTarget(int32_t RackPositionTarget) = 0;
    /**
    * Gets the value of the RackPositionTarget property.
    * @return Rack position request to the axle steering actuator (external set-point). Positive values lead to a left turn of the vehicle (based on ISO8855).
    */
    virtual int32_t getRackPositionTarget() const = 0;

    /**
    * Sets the value of the RackPositionTargetMode property.
    * @param RackPositionTargetMode Mode used for controlling rack position interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void setRackPositionTargetMode(int32_t RackPositionTargetMode) = 0;
    /**
    * Gets the value of the RackPositionTargetMode property.
    * @return Mode used for controlling rack position interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual int32_t getRackPositionTargetMode() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Steering_Axle_Row1 changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_MotionManagement_Steering_Axle_Row1.
    */
    virtual IVehicleMotionManagementSteeringAxleRow1Publisher& _getPublisher() const = 0;
};


/**
 * The IVehicleMotionManagementSteeringAxleRow1Subscriber contains functions to allow informing about signals or property changes of the IVehicleMotionManagementSteeringAxleRow1 implementation.
 * The implementation for IVehicleMotionManagementSteeringAxleRow1 should provide mechanism for subscription of the IVehicleMotionManagementSteeringAxleRow1Subscriber clients.
 * See IVehicleMotionManagementSteeringAxleRow1Publisher, which provides facilitation for this purpose.
 * The implementation for IVehicleMotionManagementSteeringAxleRow1 should call the IVehicleMotionManagementSteeringAxleRow1Subscriber interface functions on either signal emit or property change.
 * You can use IVehicleMotionManagementSteeringAxleRow1Subscriber class to implement clients of the IVehicleMotionManagementSteeringAxleRow1 or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementSteeringAxleRow1Subscriber
{
public:
    virtual ~IVehicleMotionManagementSteeringAxleRow1Subscriber() = default;
    /**
    * Called by the IVehicleMotionManagementSteeringAxleRow1Publisher when RackPositionOffsetTarget value has changed if subscribed for the RackPositionOffsetTarget change.
    * @param RackPositionOffsetTarget Rack position offset request to the axle steering actuator (for steer-by-wire), added to the actuator internal calculated set-point. Positive values without internal calculated set point change lead to a left movement of the vehicle (based on ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRackPositionOffsetTargetChanged(int32_t RackPositionOffsetTarget) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringAxleRow1Publisher when RackPositionOffsetTargetMode value has changed if subscribed for the RackPositionOffsetTargetMode change.
    * @param RackPositionOffsetTargetMode Mode used for controlling rack position offset interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRackPositionOffsetTargetModeChanged(int32_t RackPositionOffsetTargetMode) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringAxleRow1Publisher when RackPosition value has changed if subscribed for the RackPosition change.
    * @param RackPosition Represents the current position of the steering rack on axle steering actuator. Positive values leads to a left turn of the vehicle (based on ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRackPositionChanged(int32_t RackPosition) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringAxleRow1Publisher when RackPositionTarget value has changed if subscribed for the RackPositionTarget change.
    * @param RackPositionTarget Rack position request to the axle steering actuator (external set-point). Positive values lead to a left turn of the vehicle (based on ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRackPositionTargetChanged(int32_t RackPositionTarget) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringAxleRow1Publisher when RackPositionTargetMode value has changed if subscribed for the RackPositionTargetMode change.
    * @param RackPositionTargetMode Mode used for controlling rack position interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRackPositionTargetModeChanged(int32_t RackPositionTargetMode) = 0;
};

/** Callback for changes of RackPositionOffsetTarget */
using VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetPropertyCb = std::function<void(int32_t RackPositionOffsetTarget)>;
/** Callback for changes of RackPositionOffsetTargetMode */
using VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetModePropertyCb = std::function<void(int32_t RackPositionOffsetTargetMode)>;
/** Callback for changes of RackPosition */
using VehicleMotionManagementSteeringAxleRow1RackPositionPropertyCb = std::function<void(int32_t RackPosition)>;
/** Callback for changes of RackPositionTarget */
using VehicleMotionManagementSteeringAxleRow1RackPositionTargetPropertyCb = std::function<void(int32_t RackPositionTarget)>;
/** Callback for changes of RackPositionTargetMode */
using VehicleMotionManagementSteeringAxleRow1RackPositionTargetModePropertyCb = std::function<void(int32_t RackPositionTargetMode)>;


/**
 * The IVehicleMotionManagementSteeringAxleRow1Publisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleMotionManagementSteeringAxleRow1 implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleMotionManagementSteeringAxleRow1Subscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleMotionManagementSteeringAxleRow1 on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementSteeringAxleRow1Publisher
{
public:
    virtual ~IVehicleMotionManagementSteeringAxleRow1Publisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_MotionManagement_Steering_Axle_Row1.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleMotionManagementSteeringAxleRow1Subscriber which is subscribed in this function to any change of the Vehicle_MotionManagement_Steering_Axle_Row1.
    */
    virtual void subscribeToAllChanges(IVehicleMotionManagementSteeringAxleRow1Subscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_MotionManagement_Steering_Axle_Row1.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleMotionManagementSteeringAxleRow1Subscriber which subscription for any change of the Vehicle_MotionManagement_Steering_Axle_Row1 is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleMotionManagementSteeringAxleRow1Subscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for RackPositionOffsetTarget value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRackPositionOffsetTargetChanged(VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RackPositionOffsetTarget property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRackPositionOffsetTargetChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RackPositionOffsetTargetMode value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRackPositionOffsetTargetModeChanged(VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RackPositionOffsetTargetMode property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRackPositionOffsetTargetModeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RackPosition value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringAxleRow1RackPositionPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRackPositionChanged(VehicleMotionManagementSteeringAxleRow1RackPositionPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RackPosition property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRackPositionChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RackPositionTarget value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringAxleRow1RackPositionTargetPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRackPositionTargetChanged(VehicleMotionManagementSteeringAxleRow1RackPositionTargetPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RackPositionTarget property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRackPositionTargetChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RackPositionTargetMode value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringAxleRow1RackPositionTargetModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRackPositionTargetModeChanged(VehicleMotionManagementSteeringAxleRow1RackPositionTargetModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RackPositionTargetMode property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRackPositionTargetModeChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_Axle_Row1 implementation when property RackPositionOffsetTarget changes.
    * @param The new value of RackPositionOffsetTarget.
    * Rack position offset request to the axle steering actuator (for steer-by-wire), added to the actuator internal calculated set-point. Positive values without internal calculated set point change lead to a left movement of the vehicle (based on ISO8855).
    */
    virtual void publishRackPositionOffsetTargetChanged(int32_t RackPositionOffsetTarget) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_Axle_Row1 implementation when property RackPositionOffsetTargetMode changes.
    * @param The new value of RackPositionOffsetTargetMode.
    * Mode used for controlling rack position offset interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void publishRackPositionOffsetTargetModeChanged(int32_t RackPositionOffsetTargetMode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_Axle_Row1 implementation when property RackPosition changes.
    * @param The new value of RackPosition.
    * Represents the current position of the steering rack on axle steering actuator. Positive values leads to a left turn of the vehicle (based on ISO8855).
    */
    virtual void publishRackPositionChanged(int32_t RackPosition) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_Axle_Row1 implementation when property RackPositionTarget changes.
    * @param The new value of RackPositionTarget.
    * Rack position request to the axle steering actuator (external set-point). Positive values lead to a left turn of the vehicle (based on ISO8855).
    */
    virtual void publishRackPositionTargetChanged(int32_t RackPositionTarget) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_Axle_Row1 implementation when property RackPositionTargetMode changes.
    * @param The new value of RackPositionTargetMode.
    * Mode used for controlling rack position interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void publishRackPositionTargetModeChanged(int32_t RackPositionTargetMode) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
