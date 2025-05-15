#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryStateOfChargeSubscriber;
class IVehiclePowertrainTractionBatteryStateOfChargePublisher;

/**
*
* IVehiclePowertrainTractionBatteryStateOfCharge provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_StateOfCharge 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryStateOfCharge also provides an interface to access a publisher IVehiclePowertrainTractionBatteryStateOfChargePublisher, a class used by IVehiclePowertrainTractionBatteryStateOfChargeSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryStateOfChargePublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryStateOfChargeSubscriber, IVehiclePowertrainTractionBatteryStateOfChargePublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_StateOfCharge  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryStateOfCharge
{
public:
    virtual ~IVehiclePowertrainTractionBatteryStateOfCharge() = default;

    /**
    * Sets the value of the Current property.
    * @param Current Physical state of charge of the high voltage battery, relative to net capacity. This is not necessarily the state of charge being displayed to the customer.
    */
    virtual void setCurrent(float Current) = 0;
    /**
    * Gets the value of the Current property.
    * @return Physical state of charge of the high voltage battery, relative to net capacity. This is not necessarily the state of charge being displayed to the customer.
    */
    virtual float getCurrent() const = 0;

    /**
    * Sets the value of the Displayed property.
    * @param Displayed State of charge displayed to the customer.
    */
    virtual void setDisplayed(float Displayed) = 0;
    /**
    * Gets the value of the Displayed property.
    * @return State of charge displayed to the customer.
    */
    virtual float getDisplayed() const = 0;

    /**
    * Sets the value of the CurrentEnergy property.
    * @param CurrentEnergy Physical state of charge of high voltage battery expressed in kWh.
    */
    virtual void setCurrentEnergy(float CurrentEnergy) = 0;
    /**
    * Gets the value of the CurrentEnergy property.
    * @return Physical state of charge of high voltage battery expressed in kWh.
    */
    virtual float getCurrentEnergy() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_StateOfCharge changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_StateOfCharge.
    */
    virtual IVehiclePowertrainTractionBatteryStateOfChargePublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryStateOfChargeSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryStateOfCharge implementation.
 * The implementation for IVehiclePowertrainTractionBatteryStateOfCharge should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryStateOfChargeSubscriber clients.
 * See IVehiclePowertrainTractionBatteryStateOfChargePublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryStateOfCharge should call the IVehiclePowertrainTractionBatteryStateOfChargeSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryStateOfChargeSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryStateOfCharge or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryStateOfChargeSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryStateOfChargeSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryStateOfChargePublisher when Current value has changed if subscribed for the Current change.
    * @param Current Physical state of charge of the high voltage battery, relative to net capacity. This is not necessarily the state of charge being displayed to the customer.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCurrentChanged(float Current) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryStateOfChargePublisher when Displayed value has changed if subscribed for the Displayed change.
    * @param Displayed State of charge displayed to the customer.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDisplayedChanged(float Displayed) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryStateOfChargePublisher when CurrentEnergy value has changed if subscribed for the CurrentEnergy change.
    * @param CurrentEnergy Physical state of charge of high voltage battery expressed in kWh.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCurrentEnergyChanged(float CurrentEnergy) = 0;
};

/** Callback for changes of Current */
using VehiclePowertrainTractionBatteryStateOfChargeCurrentPropertyCb = std::function<void(float Current)>;
/** Callback for changes of Displayed */
using VehiclePowertrainTractionBatteryStateOfChargeDisplayedPropertyCb = std::function<void(float Displayed)>;
/** Callback for changes of CurrentEnergy */
using VehiclePowertrainTractionBatteryStateOfChargeCurrentEnergyPropertyCb = std::function<void(float CurrentEnergy)>;


/**
 * The IVehiclePowertrainTractionBatteryStateOfChargePublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryStateOfCharge implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryStateOfChargeSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryStateOfCharge on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryStateOfChargePublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryStateOfChargePublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_StateOfCharge.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryStateOfChargeSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_StateOfCharge.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryStateOfChargeSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_StateOfCharge.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryStateOfChargeSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_StateOfCharge is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryStateOfChargeSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Current value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryStateOfChargeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryStateOfChargeCurrentPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCurrentChanged(VehiclePowertrainTractionBatteryStateOfChargeCurrentPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Current property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryStateOfChargeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCurrentChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Displayed value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryStateOfChargeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryStateOfChargeDisplayedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDisplayedChanged(VehiclePowertrainTractionBatteryStateOfChargeDisplayedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Displayed property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryStateOfChargeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDisplayedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CurrentEnergy value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryStateOfChargeSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryStateOfChargeCurrentEnergyPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCurrentEnergyChanged(VehiclePowertrainTractionBatteryStateOfChargeCurrentEnergyPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CurrentEnergy property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryStateOfChargeSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCurrentEnergyChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_StateOfCharge implementation when property Current changes.
    * @param The new value of Current.
    * Physical state of charge of the high voltage battery, relative to net capacity. This is not necessarily the state of charge being displayed to the customer.
    */
    virtual void publishCurrentChanged(float Current) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_StateOfCharge implementation when property Displayed changes.
    * @param The new value of Displayed.
    * State of charge displayed to the customer.
    */
    virtual void publishDisplayedChanged(float Displayed) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_StateOfCharge implementation when property CurrentEnergy changes.
    * @param The new value of CurrentEnergy.
    * Physical state of charge of high voltage battery expressed in kWh.
    */
    virtual void publishCurrentEnergyChanged(float CurrentEnergy) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
