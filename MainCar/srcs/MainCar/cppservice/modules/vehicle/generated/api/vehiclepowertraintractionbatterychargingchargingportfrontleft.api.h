#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber;
class IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher;

/**
*
* IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft also provides an interface to access a publisher IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher, a class used by IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber, IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft() = default;

    /**
    * Sets the value of the SupportedInletTypes property.
    * @param SupportedInletTypes A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    */
    virtual void setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum SupportedInletTypes) = 0;
    /**
    * Gets the value of the SupportedInletTypes property.
    * @return A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    */
    virtual Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum getSupportedInletTypes() const = 0;

    /**
    * Sets the value of the IsFlapOpen property.
    * @param IsFlapOpen Status of the charging port flap(s).
    */
    virtual void setIsFlapOpen(bool IsFlapOpen) = 0;
    /**
    * Gets the value of the IsFlapOpen property.
    * @return Status of the charging port flap(s).
    */
    virtual bool getIsFlapOpen() const = 0;

    /**
    * Sets the value of the IsChargingCableConnected property.
    * @param IsChargingCableConnected Indicates whether a charging cable is physically connected to a particular charging port or not.
    */
    virtual void setIsChargingCableConnected(bool IsChargingCableConnected) = 0;
    /**
    * Gets the value of the IsChargingCableConnected property.
    * @return Indicates whether a charging cable is physically connected to a particular charging port or not.
    */
    virtual bool getIsChargingCableConnected() const = 0;

    /**
    * Sets the value of the IsChargingCableLocked property.
    * @param IsChargingCableLocked Is charging cable locked to prevent removal.
    */
    virtual void setIsChargingCableLocked(bool IsChargingCableLocked) = 0;
    /**
    * Gets the value of the IsChargingCableLocked property.
    * @return Is charging cable locked to prevent removal.
    */
    virtual bool getIsChargingCableLocked() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft.
    */
    virtual IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft implementation.
 * The implementation for IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber clients.
 * See IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft should call the IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher when SupportedInletTypes value has changed if subscribed for the SupportedInletTypes change.
    * @param SupportedInletTypes A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum SupportedInletTypes) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher when IsFlapOpen value has changed if subscribed for the IsFlapOpen change.
    * @param IsFlapOpen Status of the charging port flap(s).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsFlapOpenChanged(bool IsFlapOpen) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher when IsChargingCableConnected value has changed if subscribed for the IsChargingCableConnected change.
    * @param IsChargingCableConnected Indicates whether a charging cable is physically connected to a particular charging port or not.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsChargingCableConnectedChanged(bool IsChargingCableConnected) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher when IsChargingCableLocked value has changed if subscribed for the IsChargingCableLocked change.
    * @param IsChargingCableLocked Is charging cable locked to prevent removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsChargingCableLockedChanged(bool IsChargingCableLocked) = 0;
};

/** Callback for changes of SupportedInletTypes */
using VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSupportedInletTypesPropertyCb = std::function<void(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum SupportedInletTypes)>;
/** Callback for changes of IsFlapOpen */
using VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsFlapOpenPropertyCb = std::function<void(bool IsFlapOpen)>;
/** Callback for changes of IsChargingCableConnected */
using VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableConnectedPropertyCb = std::function<void(bool IsChargingCableConnected)>;
/** Callback for changes of IsChargingCableLocked */
using VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableLockedPropertyCb = std::function<void(bool IsChargingCableLocked)>;


/**
 * The IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryChargingChargingPortFrontLeft on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for SupportedInletTypes value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSupportedInletTypesPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToSupportedInletTypesChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSupportedInletTypesPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from SupportedInletTypes property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromSupportedInletTypesChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsFlapOpen value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsFlapOpenPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsFlapOpenChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsFlapOpenPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsFlapOpen property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsFlapOpenChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsChargingCableConnected value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableConnectedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsChargingCableConnectedChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableConnectedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsChargingCableConnected property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsChargingCableConnectedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsChargingCableLocked value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableLockedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsChargingCableLockedChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableLockedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsChargingCableLocked property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsChargingCableLockedChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft implementation when property SupportedInletTypes changes.
    * @param The new value of SupportedInletTypes.
    * A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    */
    virtual void publishSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum SupportedInletTypes) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft implementation when property IsFlapOpen changes.
    * @param The new value of IsFlapOpen.
    * Status of the charging port flap(s).
    */
    virtual void publishIsFlapOpenChanged(bool IsFlapOpen) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft implementation when property IsChargingCableConnected changes.
    * @param The new value of IsChargingCableConnected.
    * Indicates whether a charging cable is physically connected to a particular charging port or not.
    */
    virtual void publishIsChargingCableConnectedChanged(bool IsChargingCableConnected) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft implementation when property IsChargingCableLocked changes.
    * @param The new value of IsChargingCableLocked.
    * Is charging cable locked to prevent removal.
    */
    virtual void publishIsChargingCableLockedChanged(bool IsChargingCableLocked) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
