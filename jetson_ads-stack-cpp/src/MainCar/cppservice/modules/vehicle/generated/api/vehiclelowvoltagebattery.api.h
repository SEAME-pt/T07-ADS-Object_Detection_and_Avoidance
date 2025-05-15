#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleLowVoltageBatterySubscriber;
class IVehicleLowVoltageBatteryPublisher;

/**
*
* IVehicleLowVoltageBattery provides an interface for
 *  - methods defined for your Vehicle_LowVoltageBattery 
 *  - property setters and getters for defined properties
 * The IVehicleLowVoltageBattery also provides an interface to access a publisher IVehicleLowVoltageBatteryPublisher, a class used by IVehicleLowVoltageBatterySubscriber clients.
 * The implementation should notify the publisher IVehicleLowVoltageBatteryPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleLowVoltageBatterySubscriber, IVehicleLowVoltageBatteryPublisher
 * and the example implementation Vehicle_LowVoltageBattery  or the
 */
class CPP_VEHICLE_EXPORT IVehicleLowVoltageBattery
{
public:
    virtual ~IVehicleLowVoltageBattery() = default;

    /**
    * Sets the value of the NominalVoltage property.
    * @param NominalVoltage Nominal Voltage of the battery.
    */
    virtual void setNominalVoltage(int32_t NominalVoltage) = 0;
    /**
    * Gets the value of the NominalVoltage property.
    * @return Nominal Voltage of the battery.
    */
    virtual int32_t getNominalVoltage() const = 0;

    /**
    * Sets the value of the NominalCapacity property.
    * @param NominalCapacity Nominal capacity of the low voltage battery.
    */
    virtual void setNominalCapacity(int32_t NominalCapacity) = 0;
    /**
    * Gets the value of the NominalCapacity property.
    * @return Nominal capacity of the low voltage battery.
    */
    virtual int32_t getNominalCapacity() const = 0;

    /**
    * Sets the value of the CurrentVoltage property.
    * @param CurrentVoltage Current Voltage of the low voltage battery.
    */
    virtual void setCurrentVoltage(float CurrentVoltage) = 0;
    /**
    * Gets the value of the CurrentVoltage property.
    * @return Current Voltage of the low voltage battery.
    */
    virtual float getCurrentVoltage() const = 0;

    /**
    * Sets the value of the CurrentCurrent property.
    * @param CurrentCurrent Current current flowing in/out of the low voltage battery. Positive = Current flowing in to battery, e.g. during charging or driving. Negative = Current flowing out of battery, e.g. when using the battery to start a combustion engine.
    */
    virtual void setCurrentCurrent(float CurrentCurrent) = 0;
    /**
    * Gets the value of the CurrentCurrent property.
    * @return Current current flowing in/out of the low voltage battery. Positive = Current flowing in to battery, e.g. during charging or driving. Negative = Current flowing out of battery, e.g. when using the battery to start a combustion engine.
    */
    virtual float getCurrentCurrent() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_LowVoltageBattery changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_LowVoltageBattery.
    */
    virtual IVehicleLowVoltageBatteryPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleLowVoltageBatterySubscriber contains functions to allow informing about signals or property changes of the IVehicleLowVoltageBattery implementation.
 * The implementation for IVehicleLowVoltageBattery should provide mechanism for subscription of the IVehicleLowVoltageBatterySubscriber clients.
 * See IVehicleLowVoltageBatteryPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleLowVoltageBattery should call the IVehicleLowVoltageBatterySubscriber interface functions on either signal emit or property change.
 * You can use IVehicleLowVoltageBatterySubscriber class to implement clients of the IVehicleLowVoltageBattery or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleLowVoltageBatterySubscriber
{
public:
    virtual ~IVehicleLowVoltageBatterySubscriber() = default;
    /**
    * Called by the IVehicleLowVoltageBatteryPublisher when NominalVoltage value has changed if subscribed for the NominalVoltage change.
    * @param NominalVoltage Nominal Voltage of the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onNominalVoltageChanged(int32_t NominalVoltage) = 0;
    /**
    * Called by the IVehicleLowVoltageBatteryPublisher when NominalCapacity value has changed if subscribed for the NominalCapacity change.
    * @param NominalCapacity Nominal capacity of the low voltage battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onNominalCapacityChanged(int32_t NominalCapacity) = 0;
    /**
    * Called by the IVehicleLowVoltageBatteryPublisher when CurrentVoltage value has changed if subscribed for the CurrentVoltage change.
    * @param CurrentVoltage Current Voltage of the low voltage battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCurrentVoltageChanged(float CurrentVoltage) = 0;
    /**
    * Called by the IVehicleLowVoltageBatteryPublisher when CurrentCurrent value has changed if subscribed for the CurrentCurrent change.
    * @param CurrentCurrent Current current flowing in/out of the low voltage battery. Positive = Current flowing in to battery, e.g. during charging or driving. Negative = Current flowing out of battery, e.g. when using the battery to start a combustion engine.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCurrentCurrentChanged(float CurrentCurrent) = 0;
};

/** Callback for changes of NominalVoltage */
using VehicleLowVoltageBatteryNominalVoltagePropertyCb = std::function<void(int32_t NominalVoltage)>;
/** Callback for changes of NominalCapacity */
using VehicleLowVoltageBatteryNominalCapacityPropertyCb = std::function<void(int32_t NominalCapacity)>;
/** Callback for changes of CurrentVoltage */
using VehicleLowVoltageBatteryCurrentVoltagePropertyCb = std::function<void(float CurrentVoltage)>;
/** Callback for changes of CurrentCurrent */
using VehicleLowVoltageBatteryCurrentCurrentPropertyCb = std::function<void(float CurrentCurrent)>;


/**
 * The IVehicleLowVoltageBatteryPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleLowVoltageBattery implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleLowVoltageBatterySubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleLowVoltageBattery on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleLowVoltageBatteryPublisher
{
public:
    virtual ~IVehicleLowVoltageBatteryPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_LowVoltageBattery.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleLowVoltageBatterySubscriber which is subscribed in this function to any change of the Vehicle_LowVoltageBattery.
    */
    virtual void subscribeToAllChanges(IVehicleLowVoltageBatterySubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_LowVoltageBattery.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleLowVoltageBatterySubscriber which subscription for any change of the Vehicle_LowVoltageBattery is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleLowVoltageBatterySubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for NominalVoltage value changes.
    * If your subscriber uses subscription with IVehicleLowVoltageBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleLowVoltageBatteryNominalVoltagePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToNominalVoltageChanged(VehicleLowVoltageBatteryNominalVoltagePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from NominalVoltage property changes.
    * If your subscriber uses subscription with IVehicleLowVoltageBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromNominalVoltageChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for NominalCapacity value changes.
    * If your subscriber uses subscription with IVehicleLowVoltageBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleLowVoltageBatteryNominalCapacityPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToNominalCapacityChanged(VehicleLowVoltageBatteryNominalCapacityPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from NominalCapacity property changes.
    * If your subscriber uses subscription with IVehicleLowVoltageBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromNominalCapacityChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CurrentVoltage value changes.
    * If your subscriber uses subscription with IVehicleLowVoltageBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleLowVoltageBatteryCurrentVoltagePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCurrentVoltageChanged(VehicleLowVoltageBatteryCurrentVoltagePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CurrentVoltage property changes.
    * If your subscriber uses subscription with IVehicleLowVoltageBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCurrentVoltageChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CurrentCurrent value changes.
    * If your subscriber uses subscription with IVehicleLowVoltageBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleLowVoltageBatteryCurrentCurrentPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCurrentCurrentChanged(VehicleLowVoltageBatteryCurrentCurrentPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CurrentCurrent property changes.
    * If your subscriber uses subscription with IVehicleLowVoltageBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCurrentCurrentChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_LowVoltageBattery implementation when property NominalVoltage changes.
    * @param The new value of NominalVoltage.
    * Nominal Voltage of the battery.
    */
    virtual void publishNominalVoltageChanged(int32_t NominalVoltage) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_LowVoltageBattery implementation when property NominalCapacity changes.
    * @param The new value of NominalCapacity.
    * Nominal capacity of the low voltage battery.
    */
    virtual void publishNominalCapacityChanged(int32_t NominalCapacity) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_LowVoltageBattery implementation when property CurrentVoltage changes.
    * @param The new value of CurrentVoltage.
    * Current Voltage of the low voltage battery.
    */
    virtual void publishCurrentVoltageChanged(float CurrentVoltage) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_LowVoltageBattery implementation when property CurrentCurrent changes.
    * @param The new value of CurrentCurrent.
    * Current current flowing in/out of the low voltage battery. Positive = Current flowing in to battery, e.g. during charging or driving. Negative = Current flowing out of battery, e.g. when using the battery to start a combustion engine.
    */
    virtual void publishCurrentCurrentChanged(float CurrentCurrent) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
