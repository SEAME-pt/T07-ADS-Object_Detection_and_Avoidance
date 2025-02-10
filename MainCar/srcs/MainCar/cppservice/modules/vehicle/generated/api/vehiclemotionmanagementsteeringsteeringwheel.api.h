#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleMotionManagementSteeringSteeringWheelSubscriber;
class IVehicleMotionManagementSteeringSteeringWheelPublisher;

/**
*
* IVehicleMotionManagementSteeringSteeringWheel provides an interface for
 *  - methods defined for your Vehicle_MotionManagement_Steering_SteeringWheel 
 *  - property setters and getters for defined properties
 * The IVehicleMotionManagementSteeringSteeringWheel also provides an interface to access a publisher IVehicleMotionManagementSteeringSteeringWheelPublisher, a class used by IVehicleMotionManagementSteeringSteeringWheelSubscriber clients.
 * The implementation should notify the publisher IVehicleMotionManagementSteeringSteeringWheelPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleMotionManagementSteeringSteeringWheelSubscriber, IVehicleMotionManagementSteeringSteeringWheelPublisher
 * and the example implementation Vehicle_MotionManagement_Steering_SteeringWheel  or the
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementSteeringSteeringWheel
{
public:
    virtual ~IVehicleMotionManagementSteeringSteeringWheel() = default;

    /**
    * Sets the value of the Angle property.
    * @param Angle Represents the current input angle of the steering system, typically corresponds to angle applied by driver on the steering-wheel. Positive for angle in counterclockwise direction (based on ISO8855).
    */
    virtual void setAngle(int32_t Angle) = 0;
    /**
    * Gets the value of the Angle property.
    * @return Represents the current input angle of the steering system, typically corresponds to angle applied by driver on the steering-wheel. Positive for angle in counterclockwise direction (based on ISO8855).
    */
    virtual int32_t getAngle() const = 0;

    /**
    * Sets the value of the AngleTarget property.
    * @param AngleTarget Steering-wheel angle request to the steering actuator (external set-point for steer-by-wire). Positive sign for angle in counterclockwise direction (based on ISO8855).
    */
    virtual void setAngleTarget(int32_t AngleTarget) = 0;
    /**
    * Gets the value of the AngleTarget property.
    * @return Steering-wheel angle request to the steering actuator (external set-point for steer-by-wire). Positive sign for angle in counterclockwise direction (based on ISO8855).
    */
    virtual int32_t getAngleTarget() const = 0;

    /**
    * Sets the value of the AngleTargetMode property.
    * @param AngleTargetMode Mode used for controlling Steering-wheel angle interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void setAngleTargetMode(int32_t AngleTargetMode) = 0;
    /**
    * Gets the value of the AngleTargetMode property.
    * @return Mode used for controlling Steering-wheel angle interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual int32_t getAngleTargetMode() const = 0;

    /**
    * Sets the value of the TorqueOffsetTarget property.
    * @param TorqueOffsetTarget Steering-wheel torque offset request to the steering actuator, added to the actuator internal calculated target value. Positive sign for torque in counterclockwise direction (based on ISO8855).
    */
    virtual void setTorqueOffsetTarget(int32_t TorqueOffsetTarget) = 0;
    /**
    * Gets the value of the TorqueOffsetTarget property.
    * @return Steering-wheel torque offset request to the steering actuator, added to the actuator internal calculated target value. Positive sign for torque in counterclockwise direction (based on ISO8855).
    */
    virtual int32_t getTorqueOffsetTarget() const = 0;

    /**
    * Sets the value of the TorqueOffsetTargetMode property.
    * @param TorqueOffsetTargetMode Mode used for controlling steering-wheel torque offset interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void setTorqueOffsetTargetMode(int32_t TorqueOffsetTargetMode) = 0;
    /**
    * Gets the value of the TorqueOffsetTargetMode property.
    * @return Mode used for controlling steering-wheel torque offset interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual int32_t getTorqueOffsetTargetMode() const = 0;

    /**
    * Sets the value of the TorqueTarget property.
    * @param TorqueTarget Steering-wheel torque request to the steering actuator (external set-point for steer-by-wire). Positive sign for torque in counterclockwise direction (based on ISO8855).
    */
    virtual void setTorqueTarget(int32_t TorqueTarget) = 0;
    /**
    * Gets the value of the TorqueTarget property.
    * @return Steering-wheel torque request to the steering actuator (external set-point for steer-by-wire). Positive sign for torque in counterclockwise direction (based on ISO8855).
    */
    virtual int32_t getTorqueTarget() const = 0;

    /**
    * Sets the value of the TorqueTargetMode property.
    * @param TorqueTargetMode Mode used for controlling Steering-wheel torque interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void setTorqueTargetMode(int32_t TorqueTargetMode) = 0;
    /**
    * Gets the value of the TorqueTargetMode property.
    * @return Mode used for controlling Steering-wheel torque interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual int32_t getTorqueTargetMode() const = 0;

    /**
    * Sets the value of the Torque property.
    * @param Torque Represents the current input torque for steering system, typically corresponds to torque applied by driver on the steering wheel. Positive for torque in counterclockwise direction (based on ISO8855).
    */
    virtual void setTorque(int32_t Torque) = 0;
    /**
    * Gets the value of the Torque property.
    * @return Represents the current input torque for steering system, typically corresponds to torque applied by driver on the steering wheel. Positive for torque in counterclockwise direction (based on ISO8855).
    */
    virtual int32_t getTorque() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Steering_SteeringWheel changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_MotionManagement_Steering_SteeringWheel.
    */
    virtual IVehicleMotionManagementSteeringSteeringWheelPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleMotionManagementSteeringSteeringWheelSubscriber contains functions to allow informing about signals or property changes of the IVehicleMotionManagementSteeringSteeringWheel implementation.
 * The implementation for IVehicleMotionManagementSteeringSteeringWheel should provide mechanism for subscription of the IVehicleMotionManagementSteeringSteeringWheelSubscriber clients.
 * See IVehicleMotionManagementSteeringSteeringWheelPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleMotionManagementSteeringSteeringWheel should call the IVehicleMotionManagementSteeringSteeringWheelSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleMotionManagementSteeringSteeringWheelSubscriber class to implement clients of the IVehicleMotionManagementSteeringSteeringWheel or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementSteeringSteeringWheelSubscriber
{
public:
    virtual ~IVehicleMotionManagementSteeringSteeringWheelSubscriber() = default;
    /**
    * Called by the IVehicleMotionManagementSteeringSteeringWheelPublisher when Angle value has changed if subscribed for the Angle change.
    * @param Angle Represents the current input angle of the steering system, typically corresponds to angle applied by driver on the steering-wheel. Positive for angle in counterclockwise direction (based on ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAngleChanged(int32_t Angle) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringSteeringWheelPublisher when AngleTarget value has changed if subscribed for the AngleTarget change.
    * @param AngleTarget Steering-wheel angle request to the steering actuator (external set-point for steer-by-wire). Positive sign for angle in counterclockwise direction (based on ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAngleTargetChanged(int32_t AngleTarget) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringSteeringWheelPublisher when AngleTargetMode value has changed if subscribed for the AngleTargetMode change.
    * @param AngleTargetMode Mode used for controlling Steering-wheel angle interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAngleTargetModeChanged(int32_t AngleTargetMode) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringSteeringWheelPublisher when TorqueOffsetTarget value has changed if subscribed for the TorqueOffsetTarget change.
    * @param TorqueOffsetTarget Steering-wheel torque offset request to the steering actuator, added to the actuator internal calculated target value. Positive sign for torque in counterclockwise direction (based on ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueOffsetTargetChanged(int32_t TorqueOffsetTarget) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringSteeringWheelPublisher when TorqueOffsetTargetMode value has changed if subscribed for the TorqueOffsetTargetMode change.
    * @param TorqueOffsetTargetMode Mode used for controlling steering-wheel torque offset interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueOffsetTargetModeChanged(int32_t TorqueOffsetTargetMode) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringSteeringWheelPublisher when TorqueTarget value has changed if subscribed for the TorqueTarget change.
    * @param TorqueTarget Steering-wheel torque request to the steering actuator (external set-point for steer-by-wire). Positive sign for torque in counterclockwise direction (based on ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueTargetChanged(int32_t TorqueTarget) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringSteeringWheelPublisher when TorqueTargetMode value has changed if subscribed for the TorqueTargetMode change.
    * @param TorqueTargetMode Mode used for controlling Steering-wheel torque interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueTargetModeChanged(int32_t TorqueTargetMode) = 0;
    /**
    * Called by the IVehicleMotionManagementSteeringSteeringWheelPublisher when Torque value has changed if subscribed for the Torque change.
    * @param Torque Represents the current input torque for steering system, typically corresponds to torque applied by driver on the steering wheel. Positive for torque in counterclockwise direction (based on ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueChanged(int32_t Torque) = 0;
};

/** Callback for changes of Angle */
using VehicleMotionManagementSteeringSteeringWheelAnglePropertyCb = std::function<void(int32_t Angle)>;
/** Callback for changes of AngleTarget */
using VehicleMotionManagementSteeringSteeringWheelAngleTargetPropertyCb = std::function<void(int32_t AngleTarget)>;
/** Callback for changes of AngleTargetMode */
using VehicleMotionManagementSteeringSteeringWheelAngleTargetModePropertyCb = std::function<void(int32_t AngleTargetMode)>;
/** Callback for changes of TorqueOffsetTarget */
using VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetPropertyCb = std::function<void(int32_t TorqueOffsetTarget)>;
/** Callback for changes of TorqueOffsetTargetMode */
using VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetModePropertyCb = std::function<void(int32_t TorqueOffsetTargetMode)>;
/** Callback for changes of TorqueTarget */
using VehicleMotionManagementSteeringSteeringWheelTorqueTargetPropertyCb = std::function<void(int32_t TorqueTarget)>;
/** Callback for changes of TorqueTargetMode */
using VehicleMotionManagementSteeringSteeringWheelTorqueTargetModePropertyCb = std::function<void(int32_t TorqueTargetMode)>;
/** Callback for changes of Torque */
using VehicleMotionManagementSteeringSteeringWheelTorquePropertyCb = std::function<void(int32_t Torque)>;


/**
 * The IVehicleMotionManagementSteeringSteeringWheelPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleMotionManagementSteeringSteeringWheel implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleMotionManagementSteeringSteeringWheelSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleMotionManagementSteeringSteeringWheel on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementSteeringSteeringWheelPublisher
{
public:
    virtual ~IVehicleMotionManagementSteeringSteeringWheelPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_MotionManagement_Steering_SteeringWheel.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleMotionManagementSteeringSteeringWheelSubscriber which is subscribed in this function to any change of the Vehicle_MotionManagement_Steering_SteeringWheel.
    */
    virtual void subscribeToAllChanges(IVehicleMotionManagementSteeringSteeringWheelSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_MotionManagement_Steering_SteeringWheel.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleMotionManagementSteeringSteeringWheelSubscriber which subscription for any change of the Vehicle_MotionManagement_Steering_SteeringWheel is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleMotionManagementSteeringSteeringWheelSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Angle value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringSteeringWheelAnglePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAngleChanged(VehicleMotionManagementSteeringSteeringWheelAnglePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Angle property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAngleChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AngleTarget value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringSteeringWheelAngleTargetPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAngleTargetChanged(VehicleMotionManagementSteeringSteeringWheelAngleTargetPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AngleTarget property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAngleTargetChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AngleTargetMode value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringSteeringWheelAngleTargetModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAngleTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelAngleTargetModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AngleTargetMode property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAngleTargetModeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueOffsetTarget value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueOffsetTargetChanged(VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueOffsetTarget property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueOffsetTargetChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueOffsetTargetMode value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueOffsetTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueOffsetTargetMode property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueOffsetTargetModeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueTarget value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringSteeringWheelTorqueTargetPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueTargetChanged(VehicleMotionManagementSteeringSteeringWheelTorqueTargetPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueTarget property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueTargetChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueTargetMode value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringSteeringWheelTorqueTargetModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelTorqueTargetModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueTargetMode property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueTargetModeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Torque value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementSteeringSteeringWheelTorquePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueChanged(VehicleMotionManagementSteeringSteeringWheelTorquePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Torque property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementSteeringSteeringWheelSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_SteeringWheel implementation when property Angle changes.
    * @param The new value of Angle.
    * Represents the current input angle of the steering system, typically corresponds to angle applied by driver on the steering-wheel. Positive for angle in counterclockwise direction (based on ISO8855).
    */
    virtual void publishAngleChanged(int32_t Angle) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_SteeringWheel implementation when property AngleTarget changes.
    * @param The new value of AngleTarget.
    * Steering-wheel angle request to the steering actuator (external set-point for steer-by-wire). Positive sign for angle in counterclockwise direction (based on ISO8855).
    */
    virtual void publishAngleTargetChanged(int32_t AngleTarget) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_SteeringWheel implementation when property AngleTargetMode changes.
    * @param The new value of AngleTargetMode.
    * Mode used for controlling Steering-wheel angle interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void publishAngleTargetModeChanged(int32_t AngleTargetMode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_SteeringWheel implementation when property TorqueOffsetTarget changes.
    * @param The new value of TorqueOffsetTarget.
    * Steering-wheel torque offset request to the steering actuator, added to the actuator internal calculated target value. Positive sign for torque in counterclockwise direction (based on ISO8855).
    */
    virtual void publishTorqueOffsetTargetChanged(int32_t TorqueOffsetTarget) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_SteeringWheel implementation when property TorqueOffsetTargetMode changes.
    * @param The new value of TorqueOffsetTargetMode.
    * Mode used for controlling steering-wheel torque offset interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void publishTorqueOffsetTargetModeChanged(int32_t TorqueOffsetTargetMode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_SteeringWheel implementation when property TorqueTarget changes.
    * @param The new value of TorqueTarget.
    * Steering-wheel torque request to the steering actuator (external set-point for steer-by-wire). Positive sign for torque in counterclockwise direction (based on ISO8855).
    */
    virtual void publishTorqueTargetChanged(int32_t TorqueTarget) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_SteeringWheel implementation when property TorqueTargetMode changes.
    * @param The new value of TorqueTargetMode.
    * Mode used for controlling Steering-wheel torque interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    virtual void publishTorqueTargetModeChanged(int32_t TorqueTargetMode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Steering_SteeringWheel implementation when property Torque changes.
    * @param The new value of Torque.
    * Represents the current input torque for steering system, typically corresponds to torque applied by driver on the steering wheel. Positive for torque in counterclockwise direction (based on ISO8855).
    */
    virtual void publishTorqueChanged(int32_t Torque) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
