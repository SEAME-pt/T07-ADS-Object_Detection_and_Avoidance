#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainSubscriber;
class IVehiclePowertrainPublisher;

/**
*
* IVehiclePowertrain provides an interface for
 *  - methods defined for your Vehicle_Powertrain 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrain also provides an interface to access a publisher IVehiclePowertrainPublisher, a class used by IVehiclePowertrainSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainSubscriber, IVehiclePowertrainPublisher
 * and the example implementation Vehicle_Powertrain  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrain
{
public:
    virtual ~IVehiclePowertrain() = default;

    /**
    * Sets the value of the AccumulatedBrakingEnergy property.
    * @param AccumulatedBrakingEnergy The accumulated energy from regenerative braking over lifetime.
    */
    virtual void setAccumulatedBrakingEnergy(float AccumulatedBrakingEnergy) = 0;
    /**
    * Gets the value of the AccumulatedBrakingEnergy property.
    * @return The accumulated energy from regenerative braking over lifetime.
    */
    virtual float getAccumulatedBrakingEnergy() const = 0;

    /**
    * Sets the value of the Range property.
    * @param Range Remaining range in meters using all energy sources available in the vehicle.
    */
    virtual void setRange(int32_t Range) = 0;
    /**
    * Gets the value of the Range property.
    * @return Remaining range in meters using all energy sources available in the vehicle.
    */
    virtual int32_t getRange() const = 0;

    /**
    * Sets the value of the TimeRemaining property.
    * @param TimeRemaining Time remaining in seconds before all energy sources available in the vehicle are empty.
    */
    virtual void setTimeRemaining(int32_t TimeRemaining) = 0;
    /**
    * Gets the value of the TimeRemaining property.
    * @return Time remaining in seconds before all energy sources available in the vehicle are empty.
    */
    virtual int32_t getTimeRemaining() const = 0;

    /**
    * Sets the value of the Type property.
    * @param Type Defines the powertrain type of the vehicle.
    */
    virtual void setType(Vehicle_Powertrain_Type_ValueEnum Type) = 0;
    /**
    * Gets the value of the Type property.
    * @return Defines the powertrain type of the vehicle.
    */
    virtual Vehicle_Powertrain_Type_ValueEnum getType() const = 0;

    /**
    * Sets the value of the PowerOptimizeLevel property.
    * @param PowerOptimizeLevel Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    virtual void setPowerOptimizeLevel(int32_t PowerOptimizeLevel) = 0;
    /**
    * Gets the value of the PowerOptimizeLevel property.
    * @return Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    virtual int32_t getPowerOptimizeLevel() const = 0;

    /**
    * Sets the value of the IsAutoPowerOptimize property.
    * @param IsAutoPowerOptimize Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    virtual void setIsAutoPowerOptimize(bool IsAutoPowerOptimize) = 0;
    /**
    * Gets the value of the IsAutoPowerOptimize property.
    * @return Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    virtual bool getIsAutoPowerOptimize() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain.
    */
    virtual IVehiclePowertrainPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrain implementation.
 * The implementation for IVehiclePowertrain should provide mechanism for subscription of the IVehiclePowertrainSubscriber clients.
 * See IVehiclePowertrainPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrain should call the IVehiclePowertrainSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainSubscriber class to implement clients of the IVehiclePowertrain or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainSubscriber
{
public:
    virtual ~IVehiclePowertrainSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainPublisher when AccumulatedBrakingEnergy value has changed if subscribed for the AccumulatedBrakingEnergy change.
    * @param AccumulatedBrakingEnergy The accumulated energy from regenerative braking over lifetime.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAccumulatedBrakingEnergyChanged(float AccumulatedBrakingEnergy) = 0;
    /**
    * Called by the IVehiclePowertrainPublisher when Range value has changed if subscribed for the Range change.
    * @param Range Remaining range in meters using all energy sources available in the vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRangeChanged(int32_t Range) = 0;
    /**
    * Called by the IVehiclePowertrainPublisher when TimeRemaining value has changed if subscribed for the TimeRemaining change.
    * @param TimeRemaining Time remaining in seconds before all energy sources available in the vehicle are empty.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTimeRemainingChanged(int32_t TimeRemaining) = 0;
    /**
    * Called by the IVehiclePowertrainPublisher when Type value has changed if subscribed for the Type change.
    * @param Type Defines the powertrain type of the vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTypeChanged(Vehicle_Powertrain_Type_ValueEnum Type) = 0;
    /**
    * Called by the IVehiclePowertrainPublisher when PowerOptimizeLevel value has changed if subscribed for the PowerOptimizeLevel change.
    * @param PowerOptimizeLevel Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) = 0;
    /**
    * Called by the IVehiclePowertrainPublisher when IsAutoPowerOptimize value has changed if subscribed for the IsAutoPowerOptimize change.
    * @param IsAutoPowerOptimize Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) = 0;
};

/** Callback for changes of AccumulatedBrakingEnergy */
using VehiclePowertrainAccumulatedBrakingEnergyPropertyCb = std::function<void(float AccumulatedBrakingEnergy)>;
/** Callback for changes of Range */
using VehiclePowertrainRangePropertyCb = std::function<void(int32_t Range)>;
/** Callback for changes of TimeRemaining */
using VehiclePowertrainTimeRemainingPropertyCb = std::function<void(int32_t TimeRemaining)>;
/** Callback for changes of Type */
using VehiclePowertrainTypePropertyCb = std::function<void(Vehicle_Powertrain_Type_ValueEnum Type)>;
/** Callback for changes of PowerOptimizeLevel */
using VehiclePowertrainPowerOptimizeLevelPropertyCb = std::function<void(int32_t PowerOptimizeLevel)>;
/** Callback for changes of IsAutoPowerOptimize */
using VehiclePowertrainIsAutoPowerOptimizePropertyCb = std::function<void(bool IsAutoPowerOptimize)>;


/**
 * The IVehiclePowertrainPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrain implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrain on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainPublisher
{
public:
    virtual ~IVehiclePowertrainPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainSubscriber which subscription for any change of the Vehicle_Powertrain is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for AccumulatedBrakingEnergy value changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainAccumulatedBrakingEnergyPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAccumulatedBrakingEnergyChanged(VehiclePowertrainAccumulatedBrakingEnergyPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AccumulatedBrakingEnergy property changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAccumulatedBrakingEnergyChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Range value changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainRangePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRangeChanged(VehiclePowertrainRangePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Range property changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRangeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TimeRemaining value changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTimeRemainingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTimeRemainingChanged(VehiclePowertrainTimeRemainingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TimeRemaining property changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTimeRemainingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Type value changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTypePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTypeChanged(VehiclePowertrainTypePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Type property changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTypeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for PowerOptimizeLevel value changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainPowerOptimizeLevelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPowerOptimizeLevelChanged(VehiclePowertrainPowerOptimizeLevelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from PowerOptimizeLevel property changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPowerOptimizeLevelChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsAutoPowerOptimize value changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainIsAutoPowerOptimizePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsAutoPowerOptimizeChanged(VehiclePowertrainIsAutoPowerOptimizePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsAutoPowerOptimize property changes.
    * If your subscriber uses subscription with IVehiclePowertrainSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsAutoPowerOptimizeChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain implementation when property AccumulatedBrakingEnergy changes.
    * @param The new value of AccumulatedBrakingEnergy.
    * The accumulated energy from regenerative braking over lifetime.
    */
    virtual void publishAccumulatedBrakingEnergyChanged(float AccumulatedBrakingEnergy) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain implementation when property Range changes.
    * @param The new value of Range.
    * Remaining range in meters using all energy sources available in the vehicle.
    */
    virtual void publishRangeChanged(int32_t Range) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain implementation when property TimeRemaining changes.
    * @param The new value of TimeRemaining.
    * Time remaining in seconds before all energy sources available in the vehicle are empty.
    */
    virtual void publishTimeRemainingChanged(int32_t TimeRemaining) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain implementation when property Type changes.
    * @param The new value of Type.
    * Defines the powertrain type of the vehicle.
    */
    virtual void publishTypeChanged(Vehicle_Powertrain_Type_ValueEnum Type) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain implementation when property PowerOptimizeLevel changes.
    * @param The new value of PowerOptimizeLevel.
    * Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    virtual void publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain implementation when property IsAutoPowerOptimize changes.
    * @param The new value of IsAutoPowerOptimize.
    * Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    virtual void publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
