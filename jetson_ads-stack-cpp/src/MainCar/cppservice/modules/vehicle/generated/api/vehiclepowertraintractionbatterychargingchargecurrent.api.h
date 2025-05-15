#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber;
class IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher;

/**
*
* IVehiclePowertrainTractionBatteryChargingChargeCurrent provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryChargingChargeCurrent also provides an interface to access a publisher IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher, a class used by IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber, IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargeCurrent
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargeCurrent() = default;

    /**
    * Sets the value of the DC property.
    * @param DC Current DC charging current at inlet. Negative if returning energy to grid.
    */
    virtual void setDc(float DC) = 0;
    /**
    * Gets the value of the DC property.
    * @return Current DC charging current at inlet. Negative if returning energy to grid.
    */
    virtual float getDc() const = 0;

    /**
    * Sets the value of the Phase1 property.
    * @param Phase1 Current AC charging current (rms) at inlet for Phase 1. Negative if returning energy to grid.
    */
    virtual void setPhase1(float Phase1) = 0;
    /**
    * Gets the value of the Phase1 property.
    * @return Current AC charging current (rms) at inlet for Phase 1. Negative if returning energy to grid.
    */
    virtual float getPhase1() const = 0;

    /**
    * Sets the value of the Phase2 property.
    * @param Phase2 Current AC charging current (rms) at inlet for Phase 2. Negative if returning energy to grid.
    */
    virtual void setPhase2(float Phase2) = 0;
    /**
    * Gets the value of the Phase2 property.
    * @return Current AC charging current (rms) at inlet for Phase 2. Negative if returning energy to grid.
    */
    virtual float getPhase2() const = 0;

    /**
    * Sets the value of the Phase3 property.
    * @param Phase3 Current AC charging current (rms) at inlet for Phase 3. Negative if returning energy to grid.
    */
    virtual void setPhase3(float Phase3) = 0;
    /**
    * Gets the value of the Phase3 property.
    * @return Current AC charging current (rms) at inlet for Phase 3. Negative if returning energy to grid.
    */
    virtual float getPhase3() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent.
    */
    virtual IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryChargingChargeCurrent implementation.
 * The implementation for IVehiclePowertrainTractionBatteryChargingChargeCurrent should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber clients.
 * See IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryChargingChargeCurrent should call the IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryChargingChargeCurrent or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher when DC value has changed if subscribed for the DC change.
    * @param DC Current DC charging current at inlet. Negative if returning energy to grid.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDcChanged(float DC) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher when Phase1 value has changed if subscribed for the Phase1 change.
    * @param Phase1 Current AC charging current (rms) at inlet for Phase 1. Negative if returning energy to grid.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase1Changed(float Phase1) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher when Phase2 value has changed if subscribed for the Phase2 change.
    * @param Phase2 Current AC charging current (rms) at inlet for Phase 2. Negative if returning energy to grid.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase2Changed(float Phase2) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher when Phase3 value has changed if subscribed for the Phase3 change.
    * @param Phase3 Current AC charging current (rms) at inlet for Phase 3. Negative if returning energy to grid.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase3Changed(float Phase3) = 0;
};

/** Callback for changes of DC */
using VehiclePowertrainTractionBatteryChargingChargeCurrentDcPropertyCb = std::function<void(float DC)>;
/** Callback for changes of Phase1 */
using VehiclePowertrainTractionBatteryChargingChargeCurrentPhase1PropertyCb = std::function<void(float Phase1)>;
/** Callback for changes of Phase2 */
using VehiclePowertrainTractionBatteryChargingChargeCurrentPhase2PropertyCb = std::function<void(float Phase2)>;
/** Callback for changes of Phase3 */
using VehiclePowertrainTractionBatteryChargingChargeCurrentPhase3PropertyCb = std::function<void(float Phase3)>;


/**
 * The IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryChargingChargeCurrent implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryChargingChargeCurrent on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for DC value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeCurrentDcPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDcChanged(VehiclePowertrainTractionBatteryChargingChargeCurrentDcPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DC property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDcChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase1 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeCurrentPhase1PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase1Changed(VehiclePowertrainTractionBatteryChargingChargeCurrentPhase1PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase1 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase1Changed(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase2 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeCurrentPhase2PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase2Changed(VehiclePowertrainTractionBatteryChargingChargeCurrentPhase2PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase2 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase2Changed(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase3 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargeCurrentPhase3PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase3Changed(VehiclePowertrainTractionBatteryChargingChargeCurrentPhase3PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase3 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase3Changed(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation when property DC changes.
    * @param The new value of DC.
    * Current DC charging current at inlet. Negative if returning energy to grid.
    */
    virtual void publishDcChanged(float DC) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation when property Phase1 changes.
    * @param The new value of Phase1.
    * Current AC charging current (rms) at inlet for Phase 1. Negative if returning energy to grid.
    */
    virtual void publishPhase1Changed(float Phase1) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation when property Phase2 changes.
    * @param The new value of Phase2.
    * Current AC charging current (rms) at inlet for Phase 2. Negative if returning energy to grid.
    */
    virtual void publishPhase2Changed(float Phase2) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation when property Phase3 changes.
    * @param The new value of Phase3.
    * Current AC charging current (rms) at inlet for Phase 3. Negative if returning energy to grid.
    */
    virtual void publishPhase3Changed(float Phase3) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
