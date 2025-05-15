#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryDcdcSubscriber;
class IVehiclePowertrainTractionBatteryDcdcPublisher;

/**
*
* IVehiclePowertrainTractionBatteryDcdc provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_DCDC 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryDcdc also provides an interface to access a publisher IVehiclePowertrainTractionBatteryDcdcPublisher, a class used by IVehiclePowertrainTractionBatteryDcdcSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryDcdcPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryDcdcSubscriber, IVehiclePowertrainTractionBatteryDcdcPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_DCDC  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryDcdc
{
public:
    virtual ~IVehiclePowertrainTractionBatteryDcdc() = default;

    /**
    * Sets the value of the PowerLoss property.
    * @param PowerLoss Electrical energy lost by power dissipation to heat inside DC/DC converter.
    */
    virtual void setPowerLoss(float PowerLoss) = 0;
    /**
    * Gets the value of the PowerLoss property.
    * @return Electrical energy lost by power dissipation to heat inside DC/DC converter.
    */
    virtual float getPowerLoss() const = 0;

    /**
    * Sets the value of the Temperature property.
    * @param Temperature Current temperature of DC/DC converter converting battery high voltage to vehicle low voltage (typically 12 Volts).
    */
    virtual void setTemperature(float Temperature) = 0;
    /**
    * Gets the value of the Temperature property.
    * @return Current temperature of DC/DC converter converting battery high voltage to vehicle low voltage (typically 12 Volts).
    */
    virtual float getTemperature() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_DCDC changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_DCDC.
    */
    virtual IVehiclePowertrainTractionBatteryDcdcPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryDcdcSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryDcdc implementation.
 * The implementation for IVehiclePowertrainTractionBatteryDcdc should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryDcdcSubscriber clients.
 * See IVehiclePowertrainTractionBatteryDcdcPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryDcdc should call the IVehiclePowertrainTractionBatteryDcdcSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryDcdcSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryDcdc or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryDcdcSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryDcdcSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryDcdcPublisher when PowerLoss value has changed if subscribed for the PowerLoss change.
    * @param PowerLoss Electrical energy lost by power dissipation to heat inside DC/DC converter.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPowerLossChanged(float PowerLoss) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryDcdcPublisher when Temperature value has changed if subscribed for the Temperature change.
    * @param Temperature Current temperature of DC/DC converter converting battery high voltage to vehicle low voltage (typically 12 Volts).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTemperatureChanged(float Temperature) = 0;
};

/** Callback for changes of PowerLoss */
using VehiclePowertrainTractionBatteryDcdcPowerLossPropertyCb = std::function<void(float PowerLoss)>;
/** Callback for changes of Temperature */
using VehiclePowertrainTractionBatteryDcdcTemperaturePropertyCb = std::function<void(float Temperature)>;


/**
 * The IVehiclePowertrainTractionBatteryDcdcPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryDcdc implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryDcdcSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryDcdc on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryDcdcPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryDcdcPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_DCDC.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryDcdcSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_DCDC.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryDcdcSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_DCDC.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryDcdcSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_DCDC is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryDcdcSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for PowerLoss value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryDcdcSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryDcdcPowerLossPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryDcdcPowerLossPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from PowerLoss property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryDcdcSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPowerLossChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Temperature value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryDcdcSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryDcdcTemperaturePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTemperatureChanged(VehiclePowertrainTractionBatteryDcdcTemperaturePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Temperature property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryDcdcSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTemperatureChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_DCDC implementation when property PowerLoss changes.
    * @param The new value of PowerLoss.
    * Electrical energy lost by power dissipation to heat inside DC/DC converter.
    */
    virtual void publishPowerLossChanged(float PowerLoss) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_DCDC implementation when property Temperature changes.
    * @param The new value of Temperature.
    * Current temperature of DC/DC converter converting battery high voltage to vehicle low voltage (typically 12 Volts).
    */
    virtual void publishTemperatureChanged(float Temperature) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
