#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber;
class IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher;

/**
*
* IVehiclePowertrainTractionBatteryChargingChargeVoltage provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryChargingChargeVoltage also provides an interface to access a publisher IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher, a class used by IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber, IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargeVoltage
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargeVoltage() = default;

    /**
    * Sets the value of the DC property.
    * @param DC Current DC charging voltage at charging inlet.
    */
    virtual void setDc(float DC) = 0;
    /**
    * Gets the value of the DC property.
    * @return Current DC charging voltage at charging inlet.
    */
    virtual float getDc() const = 0;

    /**
    * Sets the value of the Phase1 property.
    * @param Phase1 Current AC charging voltage (rms) at inlet for Phase 1.
    */
    virtual void setPhase1(float Phase1) = 0;
    /**
    * Gets the value of the Phase1 property.
    * @return Current AC charging voltage (rms) at inlet for Phase 1.
    */
    virtual float getPhase1() const = 0;

    /**
    * Sets the value of the Phase2 property.
    * @param Phase2 Current AC charging voltage (rms) at inlet for Phase 2.
    */
    virtual void setPhase2(float Phase2) = 0;
    /**
    * Gets the value of the Phase2 property.
    * @return Current AC charging voltage (rms) at inlet for Phase 2.
    */
    virtual float getPhase2() const = 0;

    /**
    * Sets the value of the Phase3 property.
    * @param Phase3 Current AC charging voltage (rms) at inlet for Phase 3.
    */
    virtual void setPhase3(float Phase3) = 0;
    /**
    * Gets the value of the Phase3 property.
    * @return Current AC charging voltage (rms) at inlet for Phase 3.
    */
    virtual float getPhase3() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage.
    */
    virtual IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryChargingChargeVoltage implementation.
 * The implementation for IVehiclePowertrainTractionBatteryChargingChargeVoltage should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber clients.
 * See IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryChargingChargeVoltage should call the IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryChargingChargeVoltage or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher when DC value has changed if subscribed for the DC change.
    * @param DC Current DC charging voltage at charging inlet.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDcChanged(float DC) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher when Phase1 value has changed if subscribed for the Phase1 change.
    * @param Phase1 Current AC charging voltage (rms) at inlet for Phase 1.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase1Changed(float Phase1) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher when Phase2 value has changed if subscribed for the Phase2 change.
    * @param Phase2 Current AC charging voltage (rms) at inlet for Phase 2.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase2Changed(float Phase2) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher when Phase3 value has changed if subscribed for the Phase3 change.
    * @param Phase3 Current AC charging voltage (rms) at inlet for Phase 3.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase3Changed(float Phase3) = 0;
};

/** Callback for changes of DC */
using VehiclePowertrainTractionBatteryChargingChargeVoltageDcPropertyCb = std::function<void(float DC)>;
/** Callback for changes of Phase1 */
using VehiclePowertrainTractionBatteryChargingChargeVoltagePhase1PropertyCb = std::function<void(float Phase1)>;
/** Callback for changes of Phase2 */
using VehiclePowertrainTractionBatteryChargingChargeVoltagePhase2PropertyCb = std::function<void(float Phase2)>;
/** Callback for changes of Phase3 */
using VehiclePowertrainTractionBatteryChargingChargeVoltagePhase3PropertyCb = std::function<void(float Phase3)>;


/**
 * The IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryChargingChargeVoltage implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryChargingChargeVoltage on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for DC value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeVoltageDcPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDcChanged(VehiclePowertrainTractionBatteryChargingChargeVoltageDcPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DC property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDcChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase1 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeVoltagePhase1PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase1Changed(VehiclePowertrainTractionBatteryChargingChargeVoltagePhase1PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase1 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase1Changed(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase2 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeVoltagePhase2PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase2Changed(VehiclePowertrainTractionBatteryChargingChargeVoltagePhase2PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase2 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase2Changed(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase3 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeVoltagePhase3PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase3Changed(VehiclePowertrainTractionBatteryChargingChargeVoltagePhase3PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase3 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase3Changed(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage implementation when property DC changes.
    * @param The new value of DC.
    * Current DC charging voltage at charging inlet.
    */
    virtual void publishDcChanged(float DC) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage implementation when property Phase1 changes.
    * @param The new value of Phase1.
    * Current AC charging voltage (rms) at inlet for Phase 1.
    */
    virtual void publishPhase1Changed(float Phase1) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage implementation when property Phase2 changes.
    * @param The new value of Phase2.
    * Current AC charging voltage (rms) at inlet for Phase 2.
    */
    virtual void publishPhase2Changed(float Phase2) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage implementation when property Phase3 changes.
    * @param The new value of Phase3.
    * Current AC charging voltage (rms) at inlet for Phase 3.
    */
    virtual void publishPhase3Changed(float Phase3) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
