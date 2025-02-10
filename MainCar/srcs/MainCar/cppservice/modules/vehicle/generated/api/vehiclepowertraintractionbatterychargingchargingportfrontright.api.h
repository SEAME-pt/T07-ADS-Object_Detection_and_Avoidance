#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber;
class IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher;

/**
*
* IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight also provides an interface to access a publisher IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher, a class used by IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber, IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight() = default;

    /**
    * Sets the value of the SupportedInletTypes property.
    * @param SupportedInletTypes A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    */
    virtual void setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum SupportedInletTypes) = 0;
    /**
    * Gets the value of the SupportedInletTypes property.
    * @return A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    */
    virtual Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum getSupportedInletTypes() const = 0;

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
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight.
    */
    virtual IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight implementation.
 * The implementation for IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber clients.
 * See IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight should call the IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher when SupportedInletTypes value has changed if subscribed for the SupportedInletTypes change.
    * @param SupportedInletTypes A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum SupportedInletTypes) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher when IsFlapOpen value has changed if subscribed for the IsFlapOpen change.
    * @param IsFlapOpen Status of the charging port flap(s).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsFlapOpenChanged(bool IsFlapOpen) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher when IsChargingCableConnected value has changed if subscribed for the IsChargingCableConnected change.
    * @param IsChargingCableConnected Indicates whether a charging cable is physically connected to a particular charging port or not.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsChargingCableConnectedChanged(bool IsChargingCableConnected) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher when IsChargingCableLocked value has changed if subscribed for the IsChargingCableLocked change.
    * @param IsChargingCableLocked Is charging cable locked to prevent removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsChargingCableLockedChanged(bool IsChargingCableLocked) = 0;
};

/** Callback for changes of SupportedInletTypes */
using VehiclePowertrainTractionBatteryChargingChargingPortFrontRightSupportedInletTypesPropertyCb = std::function<void(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum SupportedInletTypes)>;
/** Callback for changes of IsFlapOpen */
using VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsFlapOpenPropertyCb = std::function<void(bool IsFlapOpen)>;
/** Callback for changes of IsChargingCableConnected */
using VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsChargingCableConnectedPropertyCb = std::function<void(bool IsChargingCableConnected)>;
/** Callback for changes of IsChargingCableLocked */
using VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsChargingCableLockedPropertyCb = std::function<void(bool IsChargingCableLocked)>;


/**
 * The IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryChargingChargingPortFrontRight implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryChargingChargingPortFrontRight on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for SupportedInletTypes value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargingPortFrontRightSupportedInletTypesPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToSupportedInletTypesChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontRightSupportedInletTypesPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from SupportedInletTypes property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromSupportedInletTypesChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsFlapOpen value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsFlapOpenPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsFlapOpenChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsFlapOpenPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsFlapOpen property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsFlapOpenChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsChargingCableConnected value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsChargingCableConnectedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsChargingCableConnectedChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsChargingCableConnectedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsChargingCableConnected property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsChargingCableConnectedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsChargingCableLocked value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsChargingCableLockedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsChargingCableLockedChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontRightIsChargingCableLockedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsChargingCableLocked property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsChargingCableLockedChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight implementation when property SupportedInletTypes changes.
    * @param The new value of SupportedInletTypes.
    * A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    */
    virtual void publishSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum SupportedInletTypes) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight implementation when property IsFlapOpen changes.
    * @param The new value of IsFlapOpen.
    * Status of the charging port flap(s).
    */
    virtual void publishIsFlapOpenChanged(bool IsFlapOpen) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight implementation when property IsChargingCableConnected changes.
    * @param The new value of IsChargingCableConnected.
    * Indicates whether a charging cable is physically connected to a particular charging port or not.
    */
    virtual void publishIsChargingCableConnectedChanged(bool IsChargingCableConnected) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight implementation when property IsChargingCableLocked changes.
    * @param The new value of IsChargingCableLocked.
    * Is charging cable locked to prevent removal.
    */
    virtual void publishIsChargingCableLockedChanged(bool IsChargingCableLocked) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
