#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber;
class IVehiclePowertrainTractionBatteryBatteryConditioningPublisher;

/**
*
* IVehiclePowertrainTractionBatteryBatteryConditioning provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_BatteryConditioning 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryBatteryConditioning also provides an interface to access a publisher IVehiclePowertrainTractionBatteryBatteryConditioningPublisher, a class used by IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryBatteryConditioningPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber, IVehiclePowertrainTractionBatteryBatteryConditioningPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_BatteryConditioning  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryBatteryConditioning
{
public:
    virtual ~IVehiclePowertrainTractionBatteryBatteryConditioning() = default;

    /**
    * Sets the value of the IsActive property.
    * @param IsActive Indicates if battery conditioning is active (i.e. actively monitors battery temperature). True = Active. False = Inactive.
    */
    virtual void setIsActive(bool IsActive) = 0;
    /**
    * Gets the value of the IsActive property.
    * @return Indicates if battery conditioning is active (i.e. actively monitors battery temperature). True = Active. False = Inactive.
    */
    virtual bool getIsActive() const = 0;

    /**
    * Sets the value of the IsOngoing property.
    * @param IsOngoing Indicating if battery conditioning is currently ongoing. Battery conditioning is considered ongoing when the battery conditioning system is actively heating or cooling the battery, or requesting heating or cooling.
    */
    virtual void setIsOngoing(bool IsOngoing) = 0;
    /**
    * Gets the value of the IsOngoing property.
    * @return Indicating if battery conditioning is currently ongoing. Battery conditioning is considered ongoing when the battery conditioning system is actively heating or cooling the battery, or requesting heating or cooling.
    */
    virtual bool getIsOngoing() const = 0;

    /**
    * Sets the value of the RequestedMode property.
    * @param RequestedMode Defines requested mode for battery conditioning. INACTIVE - Battery conditioning inactive. FAST_CHARGING_PREPARATION - Battery conditioning for fast charging. DRIVING_PREPARATION - Battery conditioning for driving.
    */
    virtual void setRequestedMode(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode) = 0;
    /**
    * Gets the value of the RequestedMode property.
    * @return Defines requested mode for battery conditioning. INACTIVE - Battery conditioning inactive. FAST_CHARGING_PREPARATION - Battery conditioning for fast charging. DRIVING_PREPARATION - Battery conditioning for driving.
    */
    virtual Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum getRequestedMode() const = 0;

    /**
    * Sets the value of the StartTime property.
    * @param StartTime Start time for battery conditioning, formatted according to ISO 8601 with UTC time zone.
    */
    virtual void setStartTime(const std::string& StartTime) = 0;
    /**
    * Gets the value of the StartTime property.
    * @return Start time for battery conditioning, formatted according to ISO 8601 with UTC time zone.
    */
    virtual const std::string& getStartTime() const = 0;

    /**
    * Sets the value of the TargetTime property.
    * @param TargetTime Target time when conditioning shall be finished, formatted according to ISO 8601 with UTC time zone.
    */
    virtual void setTargetTime(const std::string& TargetTime) = 0;
    /**
    * Gets the value of the TargetTime property.
    * @return Target time when conditioning shall be finished, formatted according to ISO 8601 with UTC time zone.
    */
    virtual const std::string& getTargetTime() const = 0;

    /**
    * Sets the value of the TargetTemperature property.
    * @param TargetTemperature Target temperature for battery conditioning.
    */
    virtual void setTargetTemperature(float TargetTemperature) = 0;
    /**
    * Gets the value of the TargetTemperature property.
    * @return Target temperature for battery conditioning.
    */
    virtual float getTargetTemperature() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_BatteryConditioning changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_BatteryConditioning.
    */
    virtual IVehiclePowertrainTractionBatteryBatteryConditioningPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryBatteryConditioning implementation.
 * The implementation for IVehiclePowertrainTractionBatteryBatteryConditioning should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber clients.
 * See IVehiclePowertrainTractionBatteryBatteryConditioningPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryBatteryConditioning should call the IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryBatteryConditioning or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryBatteryConditioningPublisher when IsActive value has changed if subscribed for the IsActive change.
    * @param IsActive Indicates if battery conditioning is active (i.e. actively monitors battery temperature). True = Active. False = Inactive.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsActiveChanged(bool IsActive) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryBatteryConditioningPublisher when IsOngoing value has changed if subscribed for the IsOngoing change.
    * @param IsOngoing Indicating if battery conditioning is currently ongoing. Battery conditioning is considered ongoing when the battery conditioning system is actively heating or cooling the battery, or requesting heating or cooling.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsOngoingChanged(bool IsOngoing) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryBatteryConditioningPublisher when RequestedMode value has changed if subscribed for the RequestedMode change.
    * @param RequestedMode Defines requested mode for battery conditioning. INACTIVE - Battery conditioning inactive. FAST_CHARGING_PREPARATION - Battery conditioning for fast charging. DRIVING_PREPARATION - Battery conditioning for driving.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRequestedModeChanged(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryBatteryConditioningPublisher when StartTime value has changed if subscribed for the StartTime change.
    * @param StartTime Start time for battery conditioning, formatted according to ISO 8601 with UTC time zone.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onStartTimeChanged(const std::string& StartTime) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryBatteryConditioningPublisher when TargetTime value has changed if subscribed for the TargetTime change.
    * @param TargetTime Target time when conditioning shall be finished, formatted according to ISO 8601 with UTC time zone.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTargetTimeChanged(const std::string& TargetTime) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryBatteryConditioningPublisher when TargetTemperature value has changed if subscribed for the TargetTemperature change.
    * @param TargetTemperature Target temperature for battery conditioning.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTargetTemperatureChanged(float TargetTemperature) = 0;
};

/** Callback for changes of IsActive */
using VehiclePowertrainTractionBatteryBatteryConditioningIsActivePropertyCb = std::function<void(bool IsActive)>;
/** Callback for changes of IsOngoing */
using VehiclePowertrainTractionBatteryBatteryConditioningIsOngoingPropertyCb = std::function<void(bool IsOngoing)>;
/** Callback for changes of RequestedMode */
using VehiclePowertrainTractionBatteryBatteryConditioningRequestedModePropertyCb = std::function<void(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode)>;
/** Callback for changes of StartTime */
using VehiclePowertrainTractionBatteryBatteryConditioningStartTimePropertyCb = std::function<void(const std::string& StartTime)>;
/** Callback for changes of TargetTime */
using VehiclePowertrainTractionBatteryBatteryConditioningTargetTimePropertyCb = std::function<void(const std::string& TargetTime)>;
/** Callback for changes of TargetTemperature */
using VehiclePowertrainTractionBatteryBatteryConditioningTargetTemperaturePropertyCb = std::function<void(float TargetTemperature)>;


/**
 * The IVehiclePowertrainTractionBatteryBatteryConditioningPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryBatteryConditioning implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryBatteryConditioning on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryBatteryConditioningPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryBatteryConditioningPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_BatteryConditioning.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_BatteryConditioning.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_BatteryConditioning.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_BatteryConditioning is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsActive value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryBatteryConditioningIsActivePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsActiveChanged(VehiclePowertrainTractionBatteryBatteryConditioningIsActivePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsActive property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsActiveChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsOngoing value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryBatteryConditioningIsOngoingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsOngoingChanged(VehiclePowertrainTractionBatteryBatteryConditioningIsOngoingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsOngoing property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsOngoingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for RequestedMode value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryBatteryConditioningRequestedModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRequestedModeChanged(VehiclePowertrainTractionBatteryBatteryConditioningRequestedModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from RequestedMode property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRequestedModeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for StartTime value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryBatteryConditioningStartTimePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToStartTimeChanged(VehiclePowertrainTractionBatteryBatteryConditioningStartTimePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from StartTime property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromStartTimeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TargetTime value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryBatteryConditioningTargetTimePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTargetTimeChanged(VehiclePowertrainTractionBatteryBatteryConditioningTargetTimePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TargetTime property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTargetTimeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TargetTemperature value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryBatteryConditioningTargetTemperaturePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTargetTemperatureChanged(VehiclePowertrainTractionBatteryBatteryConditioningTargetTemperaturePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TargetTemperature property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTargetTemperatureChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation when property IsActive changes.
    * @param The new value of IsActive.
    * Indicates if battery conditioning is active (i.e. actively monitors battery temperature). True = Active. False = Inactive.
    */
    virtual void publishIsActiveChanged(bool IsActive) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation when property IsOngoing changes.
    * @param The new value of IsOngoing.
    * Indicating if battery conditioning is currently ongoing. Battery conditioning is considered ongoing when the battery conditioning system is actively heating or cooling the battery, or requesting heating or cooling.
    */
    virtual void publishIsOngoingChanged(bool IsOngoing) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation when property RequestedMode changes.
    * @param The new value of RequestedMode.
    * Defines requested mode for battery conditioning. INACTIVE - Battery conditioning inactive. FAST_CHARGING_PREPARATION - Battery conditioning for fast charging. DRIVING_PREPARATION - Battery conditioning for driving.
    */
    virtual void publishRequestedModeChanged(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation when property StartTime changes.
    * @param The new value of StartTime.
    * Start time for battery conditioning, formatted according to ISO 8601 with UTC time zone.
    */
    virtual void publishStartTimeChanged(const std::string& StartTime) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation when property TargetTime changes.
    * @param The new value of TargetTime.
    * Target time when conditioning shall be finished, formatted according to ISO 8601 with UTC time zone.
    */
    virtual void publishTargetTimeChanged(const std::string& TargetTime) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation when property TargetTemperature changes.
    * @param The new value of TargetTemperature.
    * Target temperature for battery conditioning.
    */
    virtual void publishTargetTemperatureChanged(float TargetTemperature) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
