#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryChargingSubscriber;
class IVehiclePowertrainTractionBatteryChargingPublisher;

/**
*
* IVehiclePowertrainTractionBatteryCharging provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Charging 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryCharging also provides an interface to access a publisher IVehiclePowertrainTractionBatteryChargingPublisher, a class used by IVehiclePowertrainTractionBatteryChargingSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryChargingPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryChargingSubscriber, IVehiclePowertrainTractionBatteryChargingPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Charging  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryCharging
{
public:
    virtual ~IVehiclePowertrainTractionBatteryCharging() = default;

    /**
    * Sets the value of the ChargeLimit property.
    * @param ChargeLimit Target charge limit (state of charge) for battery.
    */
    virtual void setChargeLimit(int32_t ChargeLimit) = 0;
    /**
    * Gets the value of the ChargeLimit property.
    * @return Target charge limit (state of charge) for battery.
    */
    virtual int32_t getChargeLimit() const = 0;

    /**
    * Sets the value of the IsCharging property.
    * @param IsCharging True if charging is ongoing. Charging is considered to be ongoing if energy is flowing from charger to vehicle.
    */
    virtual void setIsCharging(bool IsCharging) = 0;
    /**
    * Gets the value of the IsCharging property.
    * @return True if charging is ongoing. Charging is considered to be ongoing if energy is flowing from charger to vehicle.
    */
    virtual bool getIsCharging() const = 0;

    /**
    * Sets the value of the IsDischarging property.
    * @param IsDischarging True if discharging (vehicle to grid) is ongoing. Discharging is considered to be ongoing if energy is flowing from vehicle to charger/grid.
    */
    virtual void setIsDischarging(bool IsDischarging) = 0;
    /**
    * Gets the value of the IsDischarging property.
    * @return True if discharging (vehicle to grid) is ongoing. Discharging is considered to be ongoing if energy is flowing from vehicle to charger/grid.
    */
    virtual bool getIsDischarging() const = 0;

    /**
    * Sets the value of the StartStopCharging property.
    * @param StartStopCharging Start or stop the charging process.
    */
    virtual void setStartStopCharging(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging) = 0;
    /**
    * Gets the value of the StartStopCharging property.
    * @return Start or stop the charging process.
    */
    virtual Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum getStartStopCharging() const = 0;

    /**
    * Sets the value of the AveragePower property.
    * @param AveragePower Average charging power of last or current charging event.
    */
    virtual void setAveragePower(float AveragePower) = 0;
    /**
    * Gets the value of the AveragePower property.
    * @return Average charging power of last or current charging event.
    */
    virtual float getAveragePower() const = 0;

    /**
    * Sets the value of the MaxPower property.
    * @param MaxPower Maximum charging power of last or current charging event.
    */
    virtual void setMaxPower(float MaxPower) = 0;
    /**
    * Gets the value of the MaxPower property.
    * @return Maximum charging power of last or current charging event.
    */
    virtual float getMaxPower() const = 0;

    /**
    * Sets the value of the EvseId property.
    * @param EvseId EVSE charging point ID (without separators) of last or current charging event according to ISO 15118-2 Annex H.
    */
    virtual void setEvseId(const std::string& EvseId) = 0;
    /**
    * Gets the value of the EvseId property.
    * @return EVSE charging point ID (without separators) of last or current charging event according to ISO 15118-2 Annex H.
    */
    virtual const std::string& getEvseId() const = 0;

    /**
    * Sets the value of the PowerLoss property.
    * @param PowerLoss Electrical energy lost by power dissipation to heat inside the AC/DC converter.
    */
    virtual void setPowerLoss(float PowerLoss) = 0;
    /**
    * Gets the value of the PowerLoss property.
    * @return Electrical energy lost by power dissipation to heat inside the AC/DC converter.
    */
    virtual float getPowerLoss() const = 0;

    /**
    * Sets the value of the Temperature property.
    * @param Temperature Current temperature of AC/DC converter converting grid voltage to battery voltage.
    */
    virtual void setTemperature(float Temperature) = 0;
    /**
    * Gets the value of the Temperature property.
    * @return Current temperature of AC/DC converter converting grid voltage to battery voltage.
    */
    virtual float getTemperature() const = 0;

    /**
    * Sets the value of the ChargeRate property.
    * @param ChargeRate Current charging rate, as in kilometers of range added per hour.
    */
    virtual void setChargeRate(float ChargeRate) = 0;
    /**
    * Gets the value of the ChargeRate property.
    * @return Current charging rate, as in kilometers of range added per hour.
    */
    virtual float getChargeRate() const = 0;

    /**
    * Sets the value of the TimeToComplete property.
    * @param TimeToComplete The time needed for the current charging process to reach Charging.ChargeLimit. 0 if charging is complete or no charging process is active or planned.
    */
    virtual void setTimeToComplete(int32_t TimeToComplete) = 0;
    /**
    * Gets the value of the TimeToComplete property.
    * @return The time needed for the current charging process to reach Charging.ChargeLimit. 0 if charging is complete or no charging process is active or planned.
    */
    virtual int32_t getTimeToComplete() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging.
    */
    virtual IVehiclePowertrainTractionBatteryChargingPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryChargingSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryCharging implementation.
 * The implementation for IVehiclePowertrainTractionBatteryCharging should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryChargingSubscriber clients.
 * See IVehiclePowertrainTractionBatteryChargingPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryCharging should call the IVehiclePowertrainTractionBatteryChargingSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryChargingSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryCharging or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when ChargeLimit value has changed if subscribed for the ChargeLimit change.
    * @param ChargeLimit Target charge limit (state of charge) for battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onChargeLimitChanged(int32_t ChargeLimit) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when IsCharging value has changed if subscribed for the IsCharging change.
    * @param IsCharging True if charging is ongoing. Charging is considered to be ongoing if energy is flowing from charger to vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsChargingChanged(bool IsCharging) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when IsDischarging value has changed if subscribed for the IsDischarging change.
    * @param IsDischarging True if discharging (vehicle to grid) is ongoing. Discharging is considered to be ongoing if energy is flowing from vehicle to charger/grid.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsDischargingChanged(bool IsDischarging) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when StartStopCharging value has changed if subscribed for the StartStopCharging change.
    * @param StartStopCharging Start or stop the charging process.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onStartStopChargingChanged(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when AveragePower value has changed if subscribed for the AveragePower change.
    * @param AveragePower Average charging power of last or current charging event.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAveragePowerChanged(float AveragePower) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when MaxPower value has changed if subscribed for the MaxPower change.
    * @param MaxPower Maximum charging power of last or current charging event.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMaxPowerChanged(float MaxPower) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when EvseId value has changed if subscribed for the EvseId change.
    * @param EvseId EVSE charging point ID (without separators) of last or current charging event according to ISO 15118-2 Annex H.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onEvseIdChanged(const std::string& EvseId) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when PowerLoss value has changed if subscribed for the PowerLoss change.
    * @param PowerLoss Electrical energy lost by power dissipation to heat inside the AC/DC converter.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPowerLossChanged(float PowerLoss) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when Temperature value has changed if subscribed for the Temperature change.
    * @param Temperature Current temperature of AC/DC converter converting grid voltage to battery voltage.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTemperatureChanged(float Temperature) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when ChargeRate value has changed if subscribed for the ChargeRate change.
    * @param ChargeRate Current charging rate, as in kilometers of range added per hour.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onChargeRateChanged(float ChargeRate) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingPublisher when TimeToComplete value has changed if subscribed for the TimeToComplete change.
    * @param TimeToComplete The time needed for the current charging process to reach Charging.ChargeLimit. 0 if charging is complete or no charging process is active or planned.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTimeToCompleteChanged(int32_t TimeToComplete) = 0;
};

/** Callback for changes of ChargeLimit */
using VehiclePowertrainTractionBatteryChargingChargeLimitPropertyCb = std::function<void(int32_t ChargeLimit)>;
/** Callback for changes of IsCharging */
using VehiclePowertrainTractionBatteryChargingIsChargingPropertyCb = std::function<void(bool IsCharging)>;
/** Callback for changes of IsDischarging */
using VehiclePowertrainTractionBatteryChargingIsDischargingPropertyCb = std::function<void(bool IsDischarging)>;
/** Callback for changes of StartStopCharging */
using VehiclePowertrainTractionBatteryChargingStartStopChargingPropertyCb = std::function<void(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging)>;
/** Callback for changes of AveragePower */
using VehiclePowertrainTractionBatteryChargingAveragePowerPropertyCb = std::function<void(float AveragePower)>;
/** Callback for changes of MaxPower */
using VehiclePowertrainTractionBatteryChargingMaxPowerPropertyCb = std::function<void(float MaxPower)>;
/** Callback for changes of EvseId */
using VehiclePowertrainTractionBatteryChargingEvseIdPropertyCb = std::function<void(const std::string& EvseId)>;
/** Callback for changes of PowerLoss */
using VehiclePowertrainTractionBatteryChargingPowerLossPropertyCb = std::function<void(float PowerLoss)>;
/** Callback for changes of Temperature */
using VehiclePowertrainTractionBatteryChargingTemperaturePropertyCb = std::function<void(float Temperature)>;
/** Callback for changes of ChargeRate */
using VehiclePowertrainTractionBatteryChargingChargeRatePropertyCb = std::function<void(float ChargeRate)>;
/** Callback for changes of TimeToComplete */
using VehiclePowertrainTractionBatteryChargingTimeToCompletePropertyCb = std::function<void(int32_t TimeToComplete)>;


/**
 * The IVehiclePowertrainTractionBatteryChargingPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryCharging implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryChargingSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryCharging on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Charging.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryChargingSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Charging.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Charging.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryChargingSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Charging is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for ChargeLimit value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeLimitPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToChargeLimitChanged(VehiclePowertrainTractionBatteryChargingChargeLimitPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from ChargeLimit property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromChargeLimitChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsCharging value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingIsChargingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsChargingChanged(VehiclePowertrainTractionBatteryChargingIsChargingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsCharging property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsChargingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsDischarging value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingIsDischargingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsDischargingChanged(VehiclePowertrainTractionBatteryChargingIsDischargingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsDischarging property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsDischargingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for StartStopCharging value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingStartStopChargingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToStartStopChargingChanged(VehiclePowertrainTractionBatteryChargingStartStopChargingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from StartStopCharging property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromStartStopChargingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AveragePower value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingAveragePowerPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAveragePowerChanged(VehiclePowertrainTractionBatteryChargingAveragePowerPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AveragePower property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAveragePowerChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for MaxPower value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingMaxPowerPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMaxPowerChanged(VehiclePowertrainTractionBatteryChargingMaxPowerPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from MaxPower property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMaxPowerChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for EvseId value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingEvseIdPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToEvseIdChanged(VehiclePowertrainTractionBatteryChargingEvseIdPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from EvseId property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromEvseIdChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for PowerLoss value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingPowerLossPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryChargingPowerLossPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from PowerLoss property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPowerLossChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Temperature value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingTemperaturePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTemperatureChanged(VehiclePowertrainTractionBatteryChargingTemperaturePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Temperature property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTemperatureChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for ChargeRate value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeRatePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToChargeRateChanged(VehiclePowertrainTractionBatteryChargingChargeRatePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from ChargeRate property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromChargeRateChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TimeToComplete value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingTimeToCompletePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTimeToCompleteChanged(VehiclePowertrainTractionBatteryChargingTimeToCompletePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TimeToComplete property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTimeToCompleteChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property ChargeLimit changes.
    * @param The new value of ChargeLimit.
    * Target charge limit (state of charge) for battery.
    */
    virtual void publishChargeLimitChanged(int32_t ChargeLimit) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property IsCharging changes.
    * @param The new value of IsCharging.
    * True if charging is ongoing. Charging is considered to be ongoing if energy is flowing from charger to vehicle.
    */
    virtual void publishIsChargingChanged(bool IsCharging) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property IsDischarging changes.
    * @param The new value of IsDischarging.
    * True if discharging (vehicle to grid) is ongoing. Discharging is considered to be ongoing if energy is flowing from vehicle to charger/grid.
    */
    virtual void publishIsDischargingChanged(bool IsDischarging) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property StartStopCharging changes.
    * @param The new value of StartStopCharging.
    * Start or stop the charging process.
    */
    virtual void publishStartStopChargingChanged(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property AveragePower changes.
    * @param The new value of AveragePower.
    * Average charging power of last or current charging event.
    */
    virtual void publishAveragePowerChanged(float AveragePower) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property MaxPower changes.
    * @param The new value of MaxPower.
    * Maximum charging power of last or current charging event.
    */
    virtual void publishMaxPowerChanged(float MaxPower) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property EvseId changes.
    * @param The new value of EvseId.
    * EVSE charging point ID (without separators) of last or current charging event according to ISO 15118-2 Annex H.
    */
    virtual void publishEvseIdChanged(const std::string& EvseId) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property PowerLoss changes.
    * @param The new value of PowerLoss.
    * Electrical energy lost by power dissipation to heat inside the AC/DC converter.
    */
    virtual void publishPowerLossChanged(float PowerLoss) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property Temperature changes.
    * @param The new value of Temperature.
    * Current temperature of AC/DC converter converting grid voltage to battery voltage.
    */
    virtual void publishTemperatureChanged(float Temperature) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property ChargeRate changes.
    * @param The new value of ChargeRate.
    * Current charging rate, as in kilometers of range added per hour.
    */
    virtual void publishChargeRateChanged(float ChargeRate) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging implementation when property TimeToComplete changes.
    * @param The new value of TimeToComplete.
    * The time needed for the current charging process to reach Charging.ChargeLimit. 0 if charging is complete or no charging process is active or planned.
    */
    virtual void publishTimeToCompleteChanged(int32_t TimeToComplete) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
