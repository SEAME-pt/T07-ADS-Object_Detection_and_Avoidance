#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryChargingTimerSubscriber;
class IVehiclePowertrainTractionBatteryChargingTimerPublisher;

/**
*
* IVehiclePowertrainTractionBatteryChargingTimer provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Charging_Timer 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryChargingTimer also provides an interface to access a publisher IVehiclePowertrainTractionBatteryChargingTimerPublisher, a class used by IVehiclePowertrainTractionBatteryChargingTimerSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryChargingTimerPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryChargingTimerSubscriber, IVehiclePowertrainTractionBatteryChargingTimerPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Charging_Timer  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingTimer
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingTimer() = default;

    /**
    * Sets the value of the Mode property.
    * @param Mode Defines timer mode for charging: INACTIVE - no timer set, charging may start as soon as battery is connected to a charger. START_TIME - charging shall start at Charging.Timer.Time. END_TIME - charging shall be finished (reach Charging.ChargeLimit) at Charging.Timer.Time. When charging is completed the vehicle shall change mode to 'inactive' or set a new Charging.Timer.Time. Charging shall start immediately if mode is 'starttime' or 'endtime' and Charging.Timer.Time is a time in the past.
    */
    virtual void setMode(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode) = 0;
    /**
    * Gets the value of the Mode property.
    * @return Defines timer mode for charging: INACTIVE - no timer set, charging may start as soon as battery is connected to a charger. START_TIME - charging shall start at Charging.Timer.Time. END_TIME - charging shall be finished (reach Charging.ChargeLimit) at Charging.Timer.Time. When charging is completed the vehicle shall change mode to 'inactive' or set a new Charging.Timer.Time. Charging shall start immediately if mode is 'starttime' or 'endtime' and Charging.Timer.Time is a time in the past.
    */
    virtual Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum getMode() const = 0;

    /**
    * Sets the value of the Time property.
    * @param Time Time for next charging-related action, formatted according to ISO 8601 with UTC time zone. Value has no significance if Charging.Timer.Mode is 'inactive'.
    */
    virtual void setTime(const std::string& Time) = 0;
    /**
    * Gets the value of the Time property.
    * @return Time for next charging-related action, formatted according to ISO 8601 with UTC time zone. Value has no significance if Charging.Timer.Mode is 'inactive'.
    */
    virtual const std::string& getTime() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_Timer changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_Timer.
    */
    virtual IVehiclePowertrainTractionBatteryChargingTimerPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryChargingTimerSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryChargingTimer implementation.
 * The implementation for IVehiclePowertrainTractionBatteryChargingTimer should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryChargingTimerSubscriber clients.
 * See IVehiclePowertrainTractionBatteryChargingTimerPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryChargingTimer should call the IVehiclePowertrainTractionBatteryChargingTimerSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryChargingTimerSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryChargingTimer or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingTimerSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingTimerSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingTimerPublisher when Mode value has changed if subscribed for the Mode change.
    * @param Mode Defines timer mode for charging: INACTIVE - no timer set, charging may start as soon as battery is connected to a charger. START_TIME - charging shall start at Charging.Timer.Time. END_TIME - charging shall be finished (reach Charging.ChargeLimit) at Charging.Timer.Time. When charging is completed the vehicle shall change mode to 'inactive' or set a new Charging.Timer.Time. Charging shall start immediately if mode is 'starttime' or 'endtime' and Charging.Timer.Time is a time in the past.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onModeChanged(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingTimerPublisher when Time value has changed if subscribed for the Time change.
    * @param Time Time for next charging-related action, formatted according to ISO 8601 with UTC time zone. Value has no significance if Charging.Timer.Mode is 'inactive'.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTimeChanged(const std::string& Time) = 0;
};

/** Callback for changes of Mode */
using VehiclePowertrainTractionBatteryChargingTimerModePropertyCb = std::function<void(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode)>;
/** Callback for changes of Time */
using VehiclePowertrainTractionBatteryChargingTimerTimePropertyCb = std::function<void(const std::string& Time)>;


/**
 * The IVehiclePowertrainTractionBatteryChargingTimerPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryChargingTimer implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryChargingTimerSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryChargingTimer on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingTimerPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingTimerPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Charging_Timer.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryChargingTimerSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Charging_Timer.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingTimerSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Charging_Timer.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryChargingTimerSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Charging_Timer is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingTimerSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Mode value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingTimerSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingTimerModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToModeChanged(VehiclePowertrainTractionBatteryChargingTimerModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Mode property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingTimerSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromModeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Time value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingTimerSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingTimerTimePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTimeChanged(VehiclePowertrainTractionBatteryChargingTimerTimePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Time property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingTimerSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTimeChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_Timer implementation when property Mode changes.
    * @param The new value of Mode.
    * Defines timer mode for charging: INACTIVE - no timer set, charging may start as soon as battery is connected to a charger. START_TIME - charging shall start at Charging.Timer.Time. END_TIME - charging shall be finished (reach Charging.ChargeLimit) at Charging.Timer.Time. When charging is completed the vehicle shall change mode to 'inactive' or set a new Charging.Timer.Time. Charging shall start immediately if mode is 'starttime' or 'endtime' and Charging.Timer.Time is a time in the past.
    */
    virtual void publishModeChanged(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_Timer implementation when property Time changes.
    * @param The new value of Time.
    * Time for next charging-related action, formatted according to ISO 8601 with UTC time zone. Value has no significance if Charging.Timer.Mode is 'inactive'.
    */
    virtual void publishTimeChanged(const std::string& Time) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
