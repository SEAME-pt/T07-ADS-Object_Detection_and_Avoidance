#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber;
class IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher;

/**
*
* IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent also provides an interface to access a publisher IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher, a class used by IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber, IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent() = default;

    /**
    * Sets the value of the DC property.
    * @param DC Maximum DC charging current at inlet that can be accepted by the system.
    */
    virtual void setDc(float DC) = 0;
    /**
    * Gets the value of the DC property.
    * @return Maximum DC charging current at inlet that can be accepted by the system.
    */
    virtual float getDc() const = 0;

    /**
    * Sets the value of the Phase1 property.
    * @param Phase1 Maximum AC charging current (rms) at inlet for Phase 1 that can be accepted by the system.
    */
    virtual void setPhase1(float Phase1) = 0;
    /**
    * Gets the value of the Phase1 property.
    * @return Maximum AC charging current (rms) at inlet for Phase 1 that can be accepted by the system.
    */
    virtual float getPhase1() const = 0;

    /**
    * Sets the value of the Phase2 property.
    * @param Phase2 Maximum AC charging current (rms) at inlet for Phase 2 that can be accepted by the system.
    */
    virtual void setPhase2(float Phase2) = 0;
    /**
    * Gets the value of the Phase2 property.
    * @return Maximum AC charging current (rms) at inlet for Phase 2 that can be accepted by the system.
    */
    virtual float getPhase2() const = 0;

    /**
    * Sets the value of the Phase3 property.
    * @param Phase3 Maximum AC charging current (rms) at inlet for Phase 3 that can be accepted by the system.
    */
    virtual void setPhase3(float Phase3) = 0;
    /**
    * Gets the value of the Phase3 property.
    * @return Maximum AC charging current (rms) at inlet for Phase 3 that can be accepted by the system.
    */
    virtual float getPhase3() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent.
    */
    virtual IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent implementation.
 * The implementation for IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber clients.
 * See IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent should call the IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher when DC value has changed if subscribed for the DC change.
    * @param DC Maximum DC charging current at inlet that can be accepted by the system.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDcChanged(float DC) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher when Phase1 value has changed if subscribed for the Phase1 change.
    * @param Phase1 Maximum AC charging current (rms) at inlet for Phase 1 that can be accepted by the system.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase1Changed(float Phase1) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher when Phase2 value has changed if subscribed for the Phase2 change.
    * @param Phase2 Maximum AC charging current (rms) at inlet for Phase 2 that can be accepted by the system.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase2Changed(float Phase2) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher when Phase3 value has changed if subscribed for the Phase3 change.
    * @param Phase3 Maximum AC charging current (rms) at inlet for Phase 3 that can be accepted by the system.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPhase3Changed(float Phase3) = 0;
};

/** Callback for changes of DC */
using VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentDcPropertyCb = std::function<void(float DC)>;
/** Callback for changes of Phase1 */
using VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase1PropertyCb = std::function<void(float Phase1)>;
/** Callback for changes of Phase2 */
using VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase2PropertyCb = std::function<void(float Phase2)>;
/** Callback for changes of Phase3 */
using VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase3PropertyCb = std::function<void(float Phase3)>;


/**
 * The IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for DC value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentDcPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDcChanged(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentDcPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DC property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDcChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase1 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase1PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase1Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase1PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase1 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase1Changed(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase2 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase2PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase2Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase2PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase2 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase2Changed(long handleId) = 0;

    /**
    * Use this function to subscribe for Phase3 value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase3PropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPhase3Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase3PropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Phase3 property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPhase3Changed(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent implementation when property DC changes.
    * @param The new value of DC.
    * Maximum DC charging current at inlet that can be accepted by the system.
    */
    virtual void publishDcChanged(float DC) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent implementation when property Phase1 changes.
    * @param The new value of Phase1.
    * Maximum AC charging current (rms) at inlet for Phase 1 that can be accepted by the system.
    */
    virtual void publishPhase1Changed(float Phase1) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent implementation when property Phase2 changes.
    * @param The new value of Phase2.
    * Maximum AC charging current (rms) at inlet for Phase 2 that can be accepted by the system.
    */
    virtual void publishPhase2Changed(float Phase2) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent implementation when property Phase3 changes.
    * @param The new value of Phase3.
    * Maximum AC charging current (rms) at inlet for Phase 3 that can be accepted by the system.
    */
    virtual void publishPhase3Changed(float Phase3) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
