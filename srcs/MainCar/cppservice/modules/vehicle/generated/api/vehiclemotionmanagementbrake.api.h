#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleMotionManagementBrakeSubscriber;
class IVehicleMotionManagementBrakePublisher;

/**
*
* IVehicleMotionManagementBrake provides an interface for
 *  - methods defined for your Vehicle_MotionManagement_Brake 
 *  - property setters and getters for defined properties
 * The IVehicleMotionManagementBrake also provides an interface to access a publisher IVehicleMotionManagementBrakePublisher, a class used by IVehicleMotionManagementBrakeSubscriber clients.
 * The implementation should notify the publisher IVehicleMotionManagementBrakePublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleMotionManagementBrakeSubscriber, IVehicleMotionManagementBrakePublisher
 * and the example implementation Vehicle_MotionManagement_Brake  or the
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementBrake
{
public:
    virtual ~IVehicleMotionManagementBrake() = default;

    /**
    * Sets the value of the VehicleForceMaximum property.
    * @param VehicleForceMaximum Maximum longitudinal brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    */
    virtual void setVehicleForceMaximum(int32_t VehicleForceMaximum) = 0;
    /**
    * Gets the value of the VehicleForceMaximum property.
    * @return Maximum longitudinal brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    */
    virtual int32_t getVehicleForceMaximum() const = 0;

    /**
    * Sets the value of the VehicleForceElectricMinimumArbitrated property.
    * @param VehicleForceElectricMinimumArbitrated Brake system internally calculated regenerative force limit at vehicle level for eAxle actuation. Brake Force < 0N.
    */
    virtual void setVehicleForceElectricMinimumArbitrated(int32_t VehicleForceElectricMinimumArbitrated) = 0;
    /**
    * Gets the value of the VehicleForceElectricMinimumArbitrated property.
    * @return Brake system internally calculated regenerative force limit at vehicle level for eAxle actuation. Brake Force < 0N.
    */
    virtual int32_t getVehicleForceElectricMinimumArbitrated() const = 0;

    /**
    * Sets the value of the VehicleForceElectric property.
    * @param VehicleForceElectric Regenerative brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    */
    virtual void setVehicleForceElectric(int32_t VehicleForceElectric) = 0;
    /**
    * Gets the value of the VehicleForceElectric property.
    * @return Regenerative brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    */
    virtual int32_t getVehicleForceElectric() const = 0;

    /**
    * Sets the value of the VehicleForceDistributionFrontMaximum property.
    * @param VehicleForceDistributionFrontMaximum Maximum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force shall be shifted to rear axle. 50% = At most 50% shall be shifted to front axle. 100%  = Complete longitudinal brake force may be shifted to front axle.
    */
    virtual void setVehicleForceDistributionFrontMaximum(int32_t VehicleForceDistributionFrontMaximum) = 0;
    /**
    * Gets the value of the VehicleForceDistributionFrontMaximum property.
    * @return Maximum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force shall be shifted to rear axle. 50% = At most 50% shall be shifted to front axle. 100%  = Complete longitudinal brake force may be shifted to front axle.
    */
    virtual int32_t getVehicleForceDistributionFrontMaximum() const = 0;

    /**
    * Sets the value of the VehicleForceDistributionFrontMinimum property.
    * @param VehicleForceDistributionFrontMinimum Minimum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force may be shifted to rear axle. 50% = At least 50% shall be shifted to front axle. 100% = Complete longitudinal brake force shall be shifted to front axle.
    */
    virtual void setVehicleForceDistributionFrontMinimum(int32_t VehicleForceDistributionFrontMinimum) = 0;
    /**
    * Gets the value of the VehicleForceDistributionFrontMinimum property.
    * @return Minimum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force may be shifted to rear axle. 50% = At least 50% shall be shifted to front axle. 100% = Complete longitudinal brake force shall be shifted to front axle.
    */
    virtual int32_t getVehicleForceDistributionFrontMinimum() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_MotionManagement_Brake.
    */
    virtual IVehicleMotionManagementBrakePublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleMotionManagementBrakeSubscriber contains functions to allow informing about signals or property changes of the IVehicleMotionManagementBrake implementation.
 * The implementation for IVehicleMotionManagementBrake should provide mechanism for subscription of the IVehicleMotionManagementBrakeSubscriber clients.
 * See IVehicleMotionManagementBrakePublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleMotionManagementBrake should call the IVehicleMotionManagementBrakeSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleMotionManagementBrakeSubscriber class to implement clients of the IVehicleMotionManagementBrake or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementBrakeSubscriber
{
public:
    virtual ~IVehicleMotionManagementBrakeSubscriber() = default;
    /**
    * Called by the IVehicleMotionManagementBrakePublisher when VehicleForceMaximum value has changed if subscribed for the VehicleForceMaximum change.
    * @param VehicleForceMaximum Maximum longitudinal brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleForceMaximumChanged(int32_t VehicleForceMaximum) = 0;
    /**
    * Called by the IVehicleMotionManagementBrakePublisher when VehicleForceElectricMinimumArbitrated value has changed if subscribed for the VehicleForceElectricMinimumArbitrated change.
    * @param VehicleForceElectricMinimumArbitrated Brake system internally calculated regenerative force limit at vehicle level for eAxle actuation. Brake Force < 0N.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleForceElectricMinimumArbitratedChanged(int32_t VehicleForceElectricMinimumArbitrated) = 0;
    /**
    * Called by the IVehicleMotionManagementBrakePublisher when VehicleForceElectric value has changed if subscribed for the VehicleForceElectric change.
    * @param VehicleForceElectric Regenerative brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleForceElectricChanged(int32_t VehicleForceElectric) = 0;
    /**
    * Called by the IVehicleMotionManagementBrakePublisher when VehicleForceDistributionFrontMaximum value has changed if subscribed for the VehicleForceDistributionFrontMaximum change.
    * @param VehicleForceDistributionFrontMaximum Maximum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force shall be shifted to rear axle. 50% = At most 50% shall be shifted to front axle. 100%  = Complete longitudinal brake force may be shifted to front axle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleForceDistributionFrontMaximumChanged(int32_t VehicleForceDistributionFrontMaximum) = 0;
    /**
    * Called by the IVehicleMotionManagementBrakePublisher when VehicleForceDistributionFrontMinimum value has changed if subscribed for the VehicleForceDistributionFrontMinimum change.
    * @param VehicleForceDistributionFrontMinimum Minimum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force may be shifted to rear axle. 50% = At least 50% shall be shifted to front axle. 100% = Complete longitudinal brake force shall be shifted to front axle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleForceDistributionFrontMinimumChanged(int32_t VehicleForceDistributionFrontMinimum) = 0;
};

/** Callback for changes of VehicleForceMaximum */
using VehicleMotionManagementBrakeVehicleForceMaximumPropertyCb = std::function<void(int32_t VehicleForceMaximum)>;
/** Callback for changes of VehicleForceElectricMinimumArbitrated */
using VehicleMotionManagementBrakeVehicleForceElectricMinimumArbitratedPropertyCb = std::function<void(int32_t VehicleForceElectricMinimumArbitrated)>;
/** Callback for changes of VehicleForceElectric */
using VehicleMotionManagementBrakeVehicleForceElectricPropertyCb = std::function<void(int32_t VehicleForceElectric)>;
/** Callback for changes of VehicleForceDistributionFrontMaximum */
using VehicleMotionManagementBrakeVehicleForceDistributionFrontMaximumPropertyCb = std::function<void(int32_t VehicleForceDistributionFrontMaximum)>;
/** Callback for changes of VehicleForceDistributionFrontMinimum */
using VehicleMotionManagementBrakeVehicleForceDistributionFrontMinimumPropertyCb = std::function<void(int32_t VehicleForceDistributionFrontMinimum)>;


/**
 * The IVehicleMotionManagementBrakePublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleMotionManagementBrake implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleMotionManagementBrakeSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleMotionManagementBrake on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementBrakePublisher
{
public:
    virtual ~IVehicleMotionManagementBrakePublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_MotionManagement_Brake.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleMotionManagementBrakeSubscriber which is subscribed in this function to any change of the Vehicle_MotionManagement_Brake.
    */
    virtual void subscribeToAllChanges(IVehicleMotionManagementBrakeSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_MotionManagement_Brake.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleMotionManagementBrakeSubscriber which subscription for any change of the Vehicle_MotionManagement_Brake is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleMotionManagementBrakeSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for VehicleForceMaximum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeVehicleForceMaximumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleForceMaximumChanged(VehicleMotionManagementBrakeVehicleForceMaximumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleForceMaximum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleForceMaximumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleForceElectricMinimumArbitrated value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeVehicleForceElectricMinimumArbitratedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleForceElectricMinimumArbitratedChanged(VehicleMotionManagementBrakeVehicleForceElectricMinimumArbitratedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleForceElectricMinimumArbitrated property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleForceElectricMinimumArbitratedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleForceElectric value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeVehicleForceElectricPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleForceElectricChanged(VehicleMotionManagementBrakeVehicleForceElectricPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleForceElectric property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleForceElectricChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleForceDistributionFrontMaximum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeVehicleForceDistributionFrontMaximumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleForceDistributionFrontMaximumChanged(VehicleMotionManagementBrakeVehicleForceDistributionFrontMaximumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleForceDistributionFrontMaximum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleForceDistributionFrontMaximumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleForceDistributionFrontMinimum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeVehicleForceDistributionFrontMinimumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleForceDistributionFrontMinimumChanged(VehicleMotionManagementBrakeVehicleForceDistributionFrontMinimumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleForceDistributionFrontMinimum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleForceDistributionFrontMinimumChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake implementation when property VehicleForceMaximum changes.
    * @param The new value of VehicleForceMaximum.
    * Maximum longitudinal brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    */
    virtual void publishVehicleForceMaximumChanged(int32_t VehicleForceMaximum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake implementation when property VehicleForceElectricMinimumArbitrated changes.
    * @param The new value of VehicleForceElectricMinimumArbitrated.
    * Brake system internally calculated regenerative force limit at vehicle level for eAxle actuation. Brake Force < 0N.
    */
    virtual void publishVehicleForceElectricMinimumArbitratedChanged(int32_t VehicleForceElectricMinimumArbitrated) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake implementation when property VehicleForceElectric changes.
    * @param The new value of VehicleForceElectric.
    * Regenerative brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    */
    virtual void publishVehicleForceElectricChanged(int32_t VehicleForceElectric) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake implementation when property VehicleForceDistributionFrontMaximum changes.
    * @param The new value of VehicleForceDistributionFrontMaximum.
    * Maximum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force shall be shifted to rear axle. 50% = At most 50% shall be shifted to front axle. 100%  = Complete longitudinal brake force may be shifted to front axle.
    */
    virtual void publishVehicleForceDistributionFrontMaximumChanged(int32_t VehicleForceDistributionFrontMaximum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake implementation when property VehicleForceDistributionFrontMinimum changes.
    * @param The new value of VehicleForceDistributionFrontMinimum.
    * Minimum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force may be shifted to rear axle. 50% = At least 50% shall be shifted to front axle. 100% = Complete longitudinal brake force shall be shifted to front axle.
    */
    virtual void publishVehicleForceDistributionFrontMinimumChanged(int32_t VehicleForceDistributionFrontMinimum) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
