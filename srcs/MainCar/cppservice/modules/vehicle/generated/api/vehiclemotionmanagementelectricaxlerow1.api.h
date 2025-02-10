#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleMotionManagementElectricAxleRow1Subscriber;
class IVehicleMotionManagementElectricAxleRow1Publisher;

/**
*
* IVehicleMotionManagementElectricAxleRow1 provides an interface for
 *  - methods defined for your Vehicle_MotionManagement_ElectricAxle_Row1 
 *  - property setters and getters for defined properties
 * The IVehicleMotionManagementElectricAxleRow1 also provides an interface to access a publisher IVehicleMotionManagementElectricAxleRow1Publisher, a class used by IVehicleMotionManagementElectricAxleRow1Subscriber clients.
 * The implementation should notify the publisher IVehicleMotionManagementElectricAxleRow1Publisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleMotionManagementElectricAxleRow1Subscriber, IVehicleMotionManagementElectricAxleRow1Publisher
 * and the example implementation Vehicle_MotionManagement_ElectricAxle_Row1  or the
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementElectricAxleRow1
{
public:
    virtual ~IVehicleMotionManagementElectricAxleRow1() = default;

    /**
    * Sets the value of the Torque property.
    * @param Torque Axle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void setTorque(int32_t Torque) = 0;
    /**
    * Gets the value of the Torque property.
    * @return Axle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual int32_t getTorque() const = 0;

    /**
    * Sets the value of the TorqueTarget property.
    * @param TorqueTarget Target axle torque in torque control mode, positive sign for torque in forward direction, negative sign for torque in backward direction (ISO8855).
    */
    virtual void setTorqueTarget(int32_t TorqueTarget) = 0;
    /**
    * Gets the value of the TorqueTarget property.
    * @return Target axle torque in torque control mode, positive sign for torque in forward direction, negative sign for torque in backward direction (ISO8855).
    */
    virtual int32_t getTorqueTarget() const = 0;

    /**
    * Sets the value of the TorqueMaximum property.
    * @param TorqueMaximum Maximum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void setTorqueMaximum(int32_t TorqueMaximum) = 0;
    /**
    * Gets the value of the TorqueMaximum property.
    * @return Maximum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual int32_t getTorqueMaximum() const = 0;

    /**
    * Sets the value of the TorqueMinimum property.
    * @param TorqueMinimum Minimum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void setTorqueMinimum(int32_t TorqueMinimum) = 0;
    /**
    * Gets the value of the TorqueMinimum property.
    * @return Minimum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual int32_t getTorqueMinimum() const = 0;

    /**
    * Sets the value of the TorqueLongTermMaximum property.
    * @param TorqueLongTermMaximum Maximum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void setTorqueLongTermMaximum(int32_t TorqueLongTermMaximum) = 0;
    /**
    * Gets the value of the TorqueLongTermMaximum property.
    * @return Maximum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual int32_t getTorqueLongTermMaximum() const = 0;

    /**
    * Sets the value of the TorqueLongTermMinimum property.
    * @param TorqueLongTermMinimum Minimum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void setTorqueLongTermMinimum(int32_t TorqueLongTermMinimum) = 0;
    /**
    * Gets the value of the TorqueLongTermMinimum property.
    * @return Minimum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual int32_t getTorqueLongTermMinimum() const = 0;

    /**
    * Sets the value of the TorqueShortTermMaximum property.
    * @param TorqueShortTermMaximum Maximum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void setTorqueShortTermMaximum(int32_t TorqueShortTermMaximum) = 0;
    /**
    * Gets the value of the TorqueShortTermMaximum property.
    * @return Maximum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual int32_t getTorqueShortTermMaximum() const = 0;

    /**
    * Sets the value of the TorqueShortTermMinimum property.
    * @param TorqueShortTermMinimum Minimum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void setTorqueShortTermMinimum(int32_t TorqueShortTermMinimum) = 0;
    /**
    * Gets the value of the TorqueShortTermMinimum property.
    * @return Minimum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual int32_t getTorqueShortTermMinimum() const = 0;

    /**
    * Sets the value of the TorqueMaximumLimit property.
    * @param TorqueMaximumLimit Maximum allowed eAxle torque in rotation speed control mode, positive sign for torque in forward direction, negative sign unused.
    */
    virtual void setTorqueMaximumLimit(int32_t TorqueMaximumLimit) = 0;
    /**
    * Gets the value of the TorqueMaximumLimit property.
    * @return Maximum allowed eAxle torque in rotation speed control mode, positive sign for torque in forward direction, negative sign unused.
    */
    virtual int32_t getTorqueMaximumLimit() const = 0;

    /**
    * Sets the value of the TorqueMinimumLimit property.
    * @param TorqueMinimumLimit Minimum allowed axle torque in rotation speed control mode, positive sign unused, negative sign for torque in backward direction (ISO8855).
    */
    virtual void setTorqueMinimumLimit(int32_t TorqueMinimumLimit) = 0;
    /**
    * Gets the value of the TorqueMinimumLimit property.
    * @return Minimum allowed axle torque in rotation speed control mode, positive sign unused, negative sign for torque in backward direction (ISO8855).
    */
    virtual int32_t getTorqueMinimumLimit() const = 0;

    /**
    * Sets the value of the RotationalSpeed property.
    * @param RotationalSpeed Rotational speed for the specified axle, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual void setRotationalSpeed(int32_t RotationalSpeed) = 0;
    /**
    * Gets the value of the RotationalSpeed property.
    * @return Rotational speed for the specified axle, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual int32_t getRotationalSpeed() const = 0;

    /**
    * Sets the value of the RotationalSpeedTarget property.
    * @param RotationalSpeedTarget Target axle rotational speed in rotation speed control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual void setRotationalSpeedTarget(int32_t RotationalSpeedTarget) = 0;
    /**
    * Gets the value of the RotationalSpeedTarget property.
    * @return Target axle rotational speed in rotation speed control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual int32_t getRotationalSpeedTarget() const = 0;

    /**
    * Sets the value of the RotationalSpeedMaximumLimit property.
    * @param RotationalSpeedMaximumLimit Maximum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual void setRotationalSpeedMaximumLimit(int32_t RotationalSpeedMaximumLimit) = 0;
    /**
    * Gets the value of the RotationalSpeedMaximumLimit property.
    * @return Maximum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual int32_t getRotationalSpeedMaximumLimit() const = 0;

    /**
    * Sets the value of the RotationalSpeedMinimumLimit property.
    * @param RotationalSpeedMinimumLimit Minimum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual void setRotationalSpeedMinimumLimit(int32_t RotationalSpeedMinimumLimit) = 0;
    /**
    * Gets the value of the RotationalSpeedMinimumLimit property.
    * @return Minimum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual int32_t getRotationalSpeedMinimumLimit() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_ElectricAxle_Row1 changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_MotionManagement_ElectricAxle_Row1.
    */
    virtual IVehicleMotionManagementElectricAxleRow1Publisher& _getPublisher() const = 0;
};


/**
 * The IVehicleMotionManagementElectricAxleRow1Subscriber contains functions to allow informing about signals or property changes of the IVehicleMotionManagementElectricAxleRow1 implementation.
 * The implementation for IVehicleMotionManagementElectricAxleRow1 should provide mechanism for subscription of the IVehicleMotionManagementElectricAxleRow1Subscriber clients.
 * See IVehicleMotionManagementElectricAxleRow1Publisher, which provides facilitation for this purpose.
 * The implementation for IVehicleMotionManagementElectricAxleRow1 should call the IVehicleMotionManagementElectricAxleRow1Subscriber interface functions on either signal emit or property change.
 * You can use IVehicleMotionManagementElectricAxleRow1Subscriber class to implement clients of the IVehicleMotionManagementElectricAxleRow1 or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementElectricAxleRow1Subscriber
{
public:
    virtual ~IVehicleMotionManagementElectricAxleRow1Subscriber() = default;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when Torque value has changed if subscribed for the Torque change.
    * @param Torque Axle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueChanged(int32_t Torque) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueTarget value has changed if subscribed for the TorqueTarget change.
    * @param TorqueTarget Target axle torque in torque control mode, positive sign for torque in forward direction, negative sign for torque in backward direction (ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueTargetChanged(int32_t TorqueTarget) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueMaximum value has changed if subscribed for the TorqueMaximum change.
    * @param TorqueMaximum Maximum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueMaximumChanged(int32_t TorqueMaximum) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueMinimum value has changed if subscribed for the TorqueMinimum change.
    * @param TorqueMinimum Minimum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueMinimumChanged(int32_t TorqueMinimum) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueLongTermMaximum value has changed if subscribed for the TorqueLongTermMaximum change.
    * @param TorqueLongTermMaximum Maximum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueLongTermMaximumChanged(int32_t TorqueLongTermMaximum) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueLongTermMinimum value has changed if subscribed for the TorqueLongTermMinimum change.
    * @param TorqueLongTermMinimum Minimum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueLongTermMinimumChanged(int32_t TorqueLongTermMinimum) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueShortTermMaximum value has changed if subscribed for the TorqueShortTermMaximum change.
    * @param TorqueShortTermMaximum Maximum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueShortTermMaximumChanged(int32_t TorqueShortTermMaximum) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueShortTermMinimum value has changed if subscribed for the TorqueShortTermMinimum change.
    * @param TorqueShortTermMinimum Minimum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueShortTermMinimumChanged(int32_t TorqueShortTermMinimum) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueMaximumLimit value has changed if subscribed for the TorqueMaximumLimit change.
    * @param TorqueMaximumLimit Maximum allowed eAxle torque in rotation speed control mode, positive sign for torque in forward direction, negative sign unused.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueMaximumLimitChanged(int32_t TorqueMaximumLimit) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when TorqueMinimumLimit value has changed if subscribed for the TorqueMinimumLimit change.
    * @param TorqueMinimumLimit Minimum allowed axle torque in rotation speed control mode, positive sign unused, negative sign for torque in backward direction (ISO8855).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueMinimumLimitChanged(int32_t TorqueMinimumLimit) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when RotationalSpeed value has changed if subscribed for the RotationalSpeed change.
    * @param RotationalSpeed Rotational speed for the specified axle, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRotationalSpeedChanged(int32_t RotationalSpeed) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when RotationalSpeedTarget value has changed if subscribed for the RotationalSpeedTarget change.
    * @param RotationalSpeedTarget Target axle rotational speed in rotation speed control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRotationalSpeedTargetChanged(int32_t RotationalSpeedTarget) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when RotationalSpeedMaximumLimit value has changed if subscribed for the RotationalSpeedMaximumLimit change.
    * @param RotationalSpeedMaximumLimit Maximum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRotationalSpeedMaximumLimitChanged(int32_t RotationalSpeedMaximumLimit) = 0;
    /**
    * Called by the IVehicleMotionManagementElectricAxleRow1Publisher when RotationalSpeedMinimumLimit value has changed if subscribed for the RotationalSpeedMinimumLimit change.
    * @param RotationalSpeedMinimumLimit Minimum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRotationalSpeedMinimumLimitChanged(int32_t RotationalSpeedMinimumLimit) = 0;
};

/** Callback for changes of Torque */
using VehicleMotionManagementElectricAxleRow1TorquePropertyCb = std::function<void(int32_t Torque)>;
/** Callback for changes of TorqueTarget */
using VehicleMotionManagementElectricAxleRow1TorqueTargetPropertyCb = std::function<void(int32_t TorqueTarget)>;
/** Callback for changes of TorqueMaximum */
using VehicleMotionManagementElectricAxleRow1TorqueMaximumPropertyCb = std::function<void(int32_t TorqueMaximum)>;
/** Callback for changes of TorqueMinimum */
using VehicleMotionManagementElectricAxleRow1TorqueMinimumPropertyCb = std::function<void(int32_t TorqueMinimum)>;
/** Callback for changes of TorqueLongTermMaximum */
using VehicleMotionManagementElectricAxleRow1TorqueLongTermMaximumPropertyCb = std::function<void(int32_t TorqueLongTermMaximum)>;
/** Callback for changes of TorqueLongTermMinimum */
using VehicleMotionManagementElectricAxleRow1TorqueLongTermMinimumPropertyCb = std::function<void(int32_t TorqueLongTermMinimum)>;
/** Callback for changes of TorqueShortTermMaximum */
using VehicleMotionManagementElectricAxleRow1TorqueShortTermMaximumPropertyCb = std::function<void(int32_t TorqueShortTermMaximum)>;
/** Callback for changes of TorqueShortTermMinimum */
using VehicleMotionManagementElectricAxleRow1TorqueShortTermMinimumPropertyCb = std::function<void(int32_t TorqueShortTermMinimum)>;
/** Callback for changes of TorqueMaximumLimit */
using VehicleMotionManagementElectricAxleRow1TorqueMaximumLimitPropertyCb = std::function<void(int32_t TorqueMaximumLimit)>;
/** Callback for changes of TorqueMinimumLimit */
using VehicleMotionManagementElectricAxleRow1TorqueMinimumLimitPropertyCb = std::function<void(int32_t TorqueMinimumLimit)>;
/** Callback for changes of RotationalSpeed */
using VehicleMotionManagementElectricAxleRow1RotationalSpeedPropertyCb = std::function<void(int32_t RotationalSpeed)>;
/** Callback for changes of RotationalSpeedTarget */
using VehicleMotionManagementElectricAxleRow1RotationalSpeedTargetPropertyCb = std::function<void(int32_t RotationalSpeedTarget)>;
/** Callback for changes of RotationalSpeedMaximumLimit */
using VehicleMotionManagementElectricAxleRow1RotationalSpeedMaximumLimitPropertyCb = std::function<void(int32_t RotationalSpeedMaximumLimit)>;
/** Callback for changes of RotationalSpeedMinimumLimit */
using VehicleMotionManagementElectricAxleRow1RotationalSpeedMinimumLimitPropertyCb = std::function<void(int32_t RotationalSpeedMinimumLimit)>;


/**
 * The IVehicleMotionManagementElectricAxleRow1Publisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleMotionManagementElectricAxleRow1 implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleMotionManagementElectricAxleRow1Subscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleMotionManagementElectricAxleRow1 on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementElectricAxleRow1Publisher
{
public:
    virtual ~IVehicleMotionManagementElectricAxleRow1Publisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_MotionManagement_ElectricAxle_Row1.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleMotionManagementElectricAxleRow1Subscriber which is subscribed in this function to any change of the Vehicle_MotionManagement_ElectricAxle_Row1.
    */
    virtual void subscribeToAllChanges(IVehicleMotionManagementElectricAxleRow1Subscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_MotionManagement_ElectricAxle_Row1.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleMotionManagementElectricAxleRow1Subscriber which subscription for any change of the Vehicle_MotionManagement_ElectricAxle_Row1 is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleMotionManagementElectricAxleRow1Subscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Torque value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorquePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueChanged(VehicleMotionManagementElectricAxleRow1TorquePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Torque property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueTarget value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueTargetPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueTargetChanged(VehicleMotionManagementElectricAxleRow1TorqueTargetPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueTarget property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueTargetChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueMaximum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueMaximumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueMaximumChanged(VehicleMotionManagementElectricAxleRow1TorqueMaximumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueMaximum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueMaximumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueMinimum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueMinimumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueMinimumChanged(VehicleMotionManagementElectricAxleRow1TorqueMinimumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueMinimum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueMinimumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueLongTermMaximum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueLongTermMaximumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueLongTermMaximumChanged(VehicleMotionManagementElectricAxleRow1TorqueLongTermMaximumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueLongTermMaximum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueLongTermMaximumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueLongTermMinimum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueLongTermMinimumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueLongTermMinimumChanged(VehicleMotionManagementElectricAxleRow1TorqueLongTermMinimumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueLongTermMinimum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueLongTermMinimumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueShortTermMaximum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueShortTermMaximumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueShortTermMaximumChanged(VehicleMotionManagementElectricAxleRow1TorqueShortTermMaximumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueShortTermMaximum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueShortTermMaximumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueShortTermMinimum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueShortTermMinimumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueShortTermMinimumChanged(VehicleMotionManagementElectricAxleRow1TorqueShortTermMinimumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueShortTermMinimum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueShortTermMinimumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueMaximumLimit value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueMaximumLimitPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueMaximumLimitChanged(VehicleMotionManagementElectricAxleRow1TorqueMaximumLimitPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueMaximumLimit property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueMaximumLimitChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueMinimumLimit value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1TorqueMinimumLimitPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueMinimumLimitChanged(VehicleMotionManagementElectricAxleRow1TorqueMinimumLimitPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueMinimumLimit property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueMinimumLimitChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RotationalSpeed value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1RotationalSpeedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRotationalSpeedChanged(VehicleMotionManagementElectricAxleRow1RotationalSpeedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RotationalSpeed property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRotationalSpeedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RotationalSpeedTarget value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1RotationalSpeedTargetPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRotationalSpeedTargetChanged(VehicleMotionManagementElectricAxleRow1RotationalSpeedTargetPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RotationalSpeedTarget property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRotationalSpeedTargetChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RotationalSpeedMaximumLimit value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1RotationalSpeedMaximumLimitPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRotationalSpeedMaximumLimitChanged(VehicleMotionManagementElectricAxleRow1RotationalSpeedMaximumLimitPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RotationalSpeedMaximumLimit property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRotationalSpeedMaximumLimitChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RotationalSpeedMinimumLimit value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementElectricAxleRow1RotationalSpeedMinimumLimitPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRotationalSpeedMinimumLimitChanged(VehicleMotionManagementElectricAxleRow1RotationalSpeedMinimumLimitPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RotationalSpeedMinimumLimit property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementElectricAxleRow1Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRotationalSpeedMinimumLimitChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property Torque changes.
    * @param The new value of Torque.
    * Axle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void publishTorqueChanged(int32_t Torque) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueTarget changes.
    * @param The new value of TorqueTarget.
    * Target axle torque in torque control mode, positive sign for torque in forward direction, negative sign for torque in backward direction (ISO8855).
    */
    virtual void publishTorqueTargetChanged(int32_t TorqueTarget) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueMaximum changes.
    * @param The new value of TorqueMaximum.
    * Maximum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void publishTorqueMaximumChanged(int32_t TorqueMaximum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueMinimum changes.
    * @param The new value of TorqueMinimum.
    * Minimum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void publishTorqueMinimumChanged(int32_t TorqueMinimum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueLongTermMaximum changes.
    * @param The new value of TorqueLongTermMaximum.
    * Maximum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void publishTorqueLongTermMaximumChanged(int32_t TorqueLongTermMaximum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueLongTermMinimum changes.
    * @param The new value of TorqueLongTermMinimum.
    * Minimum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void publishTorqueLongTermMinimumChanged(int32_t TorqueLongTermMinimum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueShortTermMaximum changes.
    * @param The new value of TorqueShortTermMaximum.
    * Maximum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void publishTorqueShortTermMaximumChanged(int32_t TorqueShortTermMaximum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueShortTermMinimum changes.
    * @param The new value of TorqueShortTermMinimum.
    * Minimum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    virtual void publishTorqueShortTermMinimumChanged(int32_t TorqueShortTermMinimum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueMaximumLimit changes.
    * @param The new value of TorqueMaximumLimit.
    * Maximum allowed eAxle torque in rotation speed control mode, positive sign for torque in forward direction, negative sign unused.
    */
    virtual void publishTorqueMaximumLimitChanged(int32_t TorqueMaximumLimit) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property TorqueMinimumLimit changes.
    * @param The new value of TorqueMinimumLimit.
    * Minimum allowed axle torque in rotation speed control mode, positive sign unused, negative sign for torque in backward direction (ISO8855).
    */
    virtual void publishTorqueMinimumLimitChanged(int32_t TorqueMinimumLimit) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property RotationalSpeed changes.
    * @param The new value of RotationalSpeed.
    * Rotational speed for the specified axle, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual void publishRotationalSpeedChanged(int32_t RotationalSpeed) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property RotationalSpeedTarget changes.
    * @param The new value of RotationalSpeedTarget.
    * Target axle rotational speed in rotation speed control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual void publishRotationalSpeedTargetChanged(int32_t RotationalSpeedTarget) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property RotationalSpeedMaximumLimit changes.
    * @param The new value of RotationalSpeedMaximumLimit.
    * Maximum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual void publishRotationalSpeedMaximumLimitChanged(int32_t RotationalSpeedMaximumLimit) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_ElectricAxle_Row1 implementation when property RotationalSpeedMinimumLimit changes.
    * @param The new value of RotationalSpeedMinimumLimit.
    * Minimum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    virtual void publishRotationalSpeedMinimumLimitChanged(int32_t RotationalSpeedMinimumLimit) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
