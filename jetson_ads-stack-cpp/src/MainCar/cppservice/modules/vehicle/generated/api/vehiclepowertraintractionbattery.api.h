#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatterySubscriber;
class IVehiclePowertrainTractionBatteryPublisher;

/**
*
* IVehiclePowertrainTractionBattery provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBattery also provides an interface to access a publisher IVehiclePowertrainTractionBatteryPublisher, a class used by IVehiclePowertrainTractionBatterySubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatterySubscriber, IVehiclePowertrainTractionBatteryPublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBattery
{
public:
    virtual ~IVehiclePowertrainTractionBattery() = default;

    /**
    * Sets the value of the Id property.
    * @param Id Battery Identification Number as assigned by OEM.
    */
    virtual void setId(const std::string& Id) = 0;
    /**
    * Gets the value of the Id property.
    * @return Battery Identification Number as assigned by OEM.
    */
    virtual const std::string& getId() const = 0;

    /**
    * Sets the value of the ProductionDate property.
    * @param ProductionDate Production date of battery in ISO8601 format, e.g. YYYY-MM-DD.
    */
    virtual void setProductionDate(const std::string& ProductionDate) = 0;
    /**
    * Gets the value of the ProductionDate property.
    * @return Production date of battery in ISO8601 format, e.g. YYYY-MM-DD.
    */
    virtual const std::string& getProductionDate() const = 0;

    /**
    * Sets the value of the ErrorCodes property.
    * @param ErrorCodes Current error codes related to the battery, if any.
    */
    virtual void setErrorCodes(const std::list<std::string>& ErrorCodes) = 0;
    /**
    * Gets the value of the ErrorCodes property.
    * @return Current error codes related to the battery, if any.
    */
    virtual const std::list<std::string>& getErrorCodes() const = 0;

    /**
    * Sets the value of the IsPowerConnected property.
    * @param IsPowerConnected Indicating if the power (positive terminator) of the traction battery is connected to the powertrain.
    */
    virtual void setIsPowerConnected(bool IsPowerConnected) = 0;
    /**
    * Gets the value of the IsPowerConnected property.
    * @return Indicating if the power (positive terminator) of the traction battery is connected to the powertrain.
    */
    virtual bool getIsPowerConnected() const = 0;

    /**
    * Sets the value of the IsGroundConnected property.
    * @param IsGroundConnected Indicating if the ground (negative terminator) of the traction battery is connected to the powertrain.
    */
    virtual void setIsGroundConnected(bool IsGroundConnected) = 0;
    /**
    * Gets the value of the IsGroundConnected property.
    * @return Indicating if the ground (negative terminator) of the traction battery is connected to the powertrain.
    */
    virtual bool getIsGroundConnected() const = 0;

    /**
    * Sets the value of the GrossCapacity property.
    * @param GrossCapacity Gross capacity of the battery.
    */
    virtual void setGrossCapacity(int32_t GrossCapacity) = 0;
    /**
    * Gets the value of the GrossCapacity property.
    * @return Gross capacity of the battery.
    */
    virtual int32_t getGrossCapacity() const = 0;

    /**
    * Sets the value of the NetCapacity property.
    * @param NetCapacity Total net capacity of the battery considering aging.
    */
    virtual void setNetCapacity(int32_t NetCapacity) = 0;
    /**
    * Gets the value of the NetCapacity property.
    * @return Total net capacity of the battery considering aging.
    */
    virtual int32_t getNetCapacity() const = 0;

    /**
    * Sets the value of the StateOfHealth property.
    * @param StateOfHealth Calculated battery state of health at standard conditions.
    */
    virtual void setStateOfHealth(float StateOfHealth) = 0;
    /**
    * Gets the value of the StateOfHealth property.
    * @return Calculated battery state of health at standard conditions.
    */
    virtual float getStateOfHealth() const = 0;

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
    * Sets the value of the MaxVoltage property.
    * @param MaxVoltage Max allowed voltage of the battery, e.g. during charging.
    */
    virtual void setMaxVoltage(int32_t MaxVoltage) = 0;
    /**
    * Gets the value of the MaxVoltage property.
    * @return Max allowed voltage of the battery, e.g. during charging.
    */
    virtual int32_t getMaxVoltage() const = 0;

    /**
    * Sets the value of the CurrentVoltage property.
    * @param CurrentVoltage Current Voltage of the battery.
    */
    virtual void setCurrentVoltage(float CurrentVoltage) = 0;
    /**
    * Gets the value of the CurrentVoltage property.
    * @return Current Voltage of the battery.
    */
    virtual float getCurrentVoltage() const = 0;

    /**
    * Sets the value of the CurrentCurrent property.
    * @param CurrentCurrent Current current flowing in/out of battery. Positive = Current flowing in to battery, e.g. during charging. Negative = Current flowing out of battery, e.g. during driving.
    */
    virtual void setCurrentCurrent(float CurrentCurrent) = 0;
    /**
    * Gets the value of the CurrentCurrent property.
    * @return Current current flowing in/out of battery. Positive = Current flowing in to battery, e.g. during charging. Negative = Current flowing out of battery, e.g. during driving.
    */
    virtual float getCurrentCurrent() const = 0;

    /**
    * Sets the value of the CurrentPower property.
    * @param CurrentPower Current electrical energy flowing in/out of battery. Positive = Energy flowing in to battery, e.g. during charging. Negative = Energy flowing out of battery, e.g. during driving.
    */
    virtual void setCurrentPower(float CurrentPower) = 0;
    /**
    * Gets the value of the CurrentPower property.
    * @return Current electrical energy flowing in/out of battery. Positive = Energy flowing in to battery, e.g. during charging. Negative = Energy flowing out of battery, e.g. during driving.
    */
    virtual float getCurrentPower() const = 0;

    /**
    * Sets the value of the AccumulatedChargedEnergy property.
    * @param AccumulatedChargedEnergy The accumulated energy delivered to the battery during charging over lifetime of the battery.
    */
    virtual void setAccumulatedChargedEnergy(float AccumulatedChargedEnergy) = 0;
    /**
    * Gets the value of the AccumulatedChargedEnergy property.
    * @return The accumulated energy delivered to the battery during charging over lifetime of the battery.
    */
    virtual float getAccumulatedChargedEnergy() const = 0;

    /**
    * Sets the value of the AccumulatedConsumedEnergy property.
    * @param AccumulatedConsumedEnergy The accumulated energy leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    */
    virtual void setAccumulatedConsumedEnergy(float AccumulatedConsumedEnergy) = 0;
    /**
    * Gets the value of the AccumulatedConsumedEnergy property.
    * @return The accumulated energy leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    */
    virtual float getAccumulatedConsumedEnergy() const = 0;

    /**
    * Sets the value of the AccumulatedChargedThroughput property.
    * @param AccumulatedChargedThroughput The accumulated charge throughput delivered to the battery during charging over lifetime of the battery.
    */
    virtual void setAccumulatedChargedThroughput(float AccumulatedChargedThroughput) = 0;
    /**
    * Gets the value of the AccumulatedChargedThroughput property.
    * @return The accumulated charge throughput delivered to the battery during charging over lifetime of the battery.
    */
    virtual float getAccumulatedChargedThroughput() const = 0;

    /**
    * Sets the value of the AccumulatedConsumedThroughput property.
    * @param AccumulatedConsumedThroughput The accumulated charge throughput leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    */
    virtual void setAccumulatedConsumedThroughput(float AccumulatedConsumedThroughput) = 0;
    /**
    * Gets the value of the AccumulatedConsumedThroughput property.
    * @return The accumulated charge throughput leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    */
    virtual float getAccumulatedConsumedThroughput() const = 0;

    /**
    * Sets the value of the PowerLoss property.
    * @param PowerLoss Electrical energy lost by power dissipation to heat inside the battery.
    */
    virtual void setPowerLoss(float PowerLoss) = 0;
    /**
    * Gets the value of the PowerLoss property.
    * @return Electrical energy lost by power dissipation to heat inside the battery.
    */
    virtual float getPowerLoss() const = 0;

    /**
    * Sets the value of the Range property.
    * @param Range Remaining range in meters using only battery.
    */
    virtual void setRange(int32_t Range) = 0;
    /**
    * Gets the value of the Range property.
    * @return Remaining range in meters using only battery.
    */
    virtual int32_t getRange() const = 0;

    /**
    * Sets the value of the TimeRemaining property.
    * @param TimeRemaining Time remaining in seconds before the battery is empty.
    */
    virtual void setTimeRemaining(int32_t TimeRemaining) = 0;
    /**
    * Gets the value of the TimeRemaining property.
    * @return Time remaining in seconds before the battery is empty.
    */
    virtual int32_t getTimeRemaining() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery.
    */
    virtual IVehiclePowertrainTractionBatteryPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatterySubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBattery implementation.
 * The implementation for IVehiclePowertrainTractionBattery should provide mechanism for subscription of the IVehiclePowertrainTractionBatterySubscriber clients.
 * See IVehiclePowertrainTractionBatteryPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBattery should call the IVehiclePowertrainTractionBatterySubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatterySubscriber class to implement clients of the IVehiclePowertrainTractionBattery or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatterySubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatterySubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when Id value has changed if subscribed for the Id change.
    * @param Id Battery Identification Number as assigned by OEM.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIdChanged(const std::string& Id) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when ProductionDate value has changed if subscribed for the ProductionDate change.
    * @param ProductionDate Production date of battery in ISO8601 format, e.g. YYYY-MM-DD.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onProductionDateChanged(const std::string& ProductionDate) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when ErrorCodes value has changed if subscribed for the ErrorCodes change.
    * @param ErrorCodes Current error codes related to the battery, if any.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onErrorCodesChanged(const std::list<std::string>& ErrorCodes) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when IsPowerConnected value has changed if subscribed for the IsPowerConnected change.
    * @param IsPowerConnected Indicating if the power (positive terminator) of the traction battery is connected to the powertrain.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsPowerConnectedChanged(bool IsPowerConnected) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when IsGroundConnected value has changed if subscribed for the IsGroundConnected change.
    * @param IsGroundConnected Indicating if the ground (negative terminator) of the traction battery is connected to the powertrain.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsGroundConnectedChanged(bool IsGroundConnected) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when GrossCapacity value has changed if subscribed for the GrossCapacity change.
    * @param GrossCapacity Gross capacity of the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onGrossCapacityChanged(int32_t GrossCapacity) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when NetCapacity value has changed if subscribed for the NetCapacity change.
    * @param NetCapacity Total net capacity of the battery considering aging.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onNetCapacityChanged(int32_t NetCapacity) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when StateOfHealth value has changed if subscribed for the StateOfHealth change.
    * @param StateOfHealth Calculated battery state of health at standard conditions.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onStateOfHealthChanged(float StateOfHealth) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when NominalVoltage value has changed if subscribed for the NominalVoltage change.
    * @param NominalVoltage Nominal Voltage of the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onNominalVoltageChanged(int32_t NominalVoltage) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when MaxVoltage value has changed if subscribed for the MaxVoltage change.
    * @param MaxVoltage Max allowed voltage of the battery, e.g. during charging.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMaxVoltageChanged(int32_t MaxVoltage) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when CurrentVoltage value has changed if subscribed for the CurrentVoltage change.
    * @param CurrentVoltage Current Voltage of the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCurrentVoltageChanged(float CurrentVoltage) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when CurrentCurrent value has changed if subscribed for the CurrentCurrent change.
    * @param CurrentCurrent Current current flowing in/out of battery. Positive = Current flowing in to battery, e.g. during charging. Negative = Current flowing out of battery, e.g. during driving.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCurrentCurrentChanged(float CurrentCurrent) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when CurrentPower value has changed if subscribed for the CurrentPower change.
    * @param CurrentPower Current electrical energy flowing in/out of battery. Positive = Energy flowing in to battery, e.g. during charging. Negative = Energy flowing out of battery, e.g. during driving.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCurrentPowerChanged(float CurrentPower) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when AccumulatedChargedEnergy value has changed if subscribed for the AccumulatedChargedEnergy change.
    * @param AccumulatedChargedEnergy The accumulated energy delivered to the battery during charging over lifetime of the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAccumulatedChargedEnergyChanged(float AccumulatedChargedEnergy) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when AccumulatedConsumedEnergy value has changed if subscribed for the AccumulatedConsumedEnergy change.
    * @param AccumulatedConsumedEnergy The accumulated energy leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAccumulatedConsumedEnergyChanged(float AccumulatedConsumedEnergy) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when AccumulatedChargedThroughput value has changed if subscribed for the AccumulatedChargedThroughput change.
    * @param AccumulatedChargedThroughput The accumulated charge throughput delivered to the battery during charging over lifetime of the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAccumulatedChargedThroughputChanged(float AccumulatedChargedThroughput) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when AccumulatedConsumedThroughput value has changed if subscribed for the AccumulatedConsumedThroughput change.
    * @param AccumulatedConsumedThroughput The accumulated charge throughput leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAccumulatedConsumedThroughputChanged(float AccumulatedConsumedThroughput) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when PowerLoss value has changed if subscribed for the PowerLoss change.
    * @param PowerLoss Electrical energy lost by power dissipation to heat inside the battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPowerLossChanged(float PowerLoss) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when Range value has changed if subscribed for the Range change.
    * @param Range Remaining range in meters using only battery.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onRangeChanged(int32_t Range) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryPublisher when TimeRemaining value has changed if subscribed for the TimeRemaining change.
    * @param TimeRemaining Time remaining in seconds before the battery is empty.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTimeRemainingChanged(int32_t TimeRemaining) = 0;
};

/** Callback for changes of Id */
using VehiclePowertrainTractionBatteryIdPropertyCb = std::function<void(const std::string& Id)>;
/** Callback for changes of ProductionDate */
using VehiclePowertrainTractionBatteryProductionDatePropertyCb = std::function<void(const std::string& ProductionDate)>;
/** Callback for changes of ErrorCodes */
using VehiclePowertrainTractionBatteryErrorCodesPropertyCb = std::function<void(const std::list<std::string>& ErrorCodes)>;
/** Callback for changes of IsPowerConnected */
using VehiclePowertrainTractionBatteryIsPowerConnectedPropertyCb = std::function<void(bool IsPowerConnected)>;
/** Callback for changes of IsGroundConnected */
using VehiclePowertrainTractionBatteryIsGroundConnectedPropertyCb = std::function<void(bool IsGroundConnected)>;
/** Callback for changes of GrossCapacity */
using VehiclePowertrainTractionBatteryGrossCapacityPropertyCb = std::function<void(int32_t GrossCapacity)>;
/** Callback for changes of NetCapacity */
using VehiclePowertrainTractionBatteryNetCapacityPropertyCb = std::function<void(int32_t NetCapacity)>;
/** Callback for changes of StateOfHealth */
using VehiclePowertrainTractionBatteryStateOfHealthPropertyCb = std::function<void(float StateOfHealth)>;
/** Callback for changes of NominalVoltage */
using VehiclePowertrainTractionBatteryNominalVoltagePropertyCb = std::function<void(int32_t NominalVoltage)>;
/** Callback for changes of MaxVoltage */
using VehiclePowertrainTractionBatteryMaxVoltagePropertyCb = std::function<void(int32_t MaxVoltage)>;
/** Callback for changes of CurrentVoltage */
using VehiclePowertrainTractionBatteryCurrentVoltagePropertyCb = std::function<void(float CurrentVoltage)>;
/** Callback for changes of CurrentCurrent */
using VehiclePowertrainTractionBatteryCurrentCurrentPropertyCb = std::function<void(float CurrentCurrent)>;
/** Callback for changes of CurrentPower */
using VehiclePowertrainTractionBatteryCurrentPowerPropertyCb = std::function<void(float CurrentPower)>;
/** Callback for changes of AccumulatedChargedEnergy */
using VehiclePowertrainTractionBatteryAccumulatedChargedEnergyPropertyCb = std::function<void(float AccumulatedChargedEnergy)>;
/** Callback for changes of AccumulatedConsumedEnergy */
using VehiclePowertrainTractionBatteryAccumulatedConsumedEnergyPropertyCb = std::function<void(float AccumulatedConsumedEnergy)>;
/** Callback for changes of AccumulatedChargedThroughput */
using VehiclePowertrainTractionBatteryAccumulatedChargedThroughputPropertyCb = std::function<void(float AccumulatedChargedThroughput)>;
/** Callback for changes of AccumulatedConsumedThroughput */
using VehiclePowertrainTractionBatteryAccumulatedConsumedThroughputPropertyCb = std::function<void(float AccumulatedConsumedThroughput)>;
/** Callback for changes of PowerLoss */
using VehiclePowertrainTractionBatteryPowerLossPropertyCb = std::function<void(float PowerLoss)>;
/** Callback for changes of Range */
using VehiclePowertrainTractionBatteryRangePropertyCb = std::function<void(int32_t Range)>;
/** Callback for changes of TimeRemaining */
using VehiclePowertrainTractionBatteryTimeRemainingPropertyCb = std::function<void(int32_t TimeRemaining)>;


/**
 * The IVehiclePowertrainTractionBatteryPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBattery implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatterySubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBattery on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryPublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatterySubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatterySubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatterySubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatterySubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Id value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryIdPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIdChanged(VehiclePowertrainTractionBatteryIdPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Id property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIdChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for ProductionDate value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryProductionDatePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToProductionDateChanged(VehiclePowertrainTractionBatteryProductionDatePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from ProductionDate property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromProductionDateChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for ErrorCodes value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryErrorCodesPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToErrorCodesChanged(VehiclePowertrainTractionBatteryErrorCodesPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from ErrorCodes property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromErrorCodesChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsPowerConnected value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryIsPowerConnectedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsPowerConnectedChanged(VehiclePowertrainTractionBatteryIsPowerConnectedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsPowerConnected property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsPowerConnectedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsGroundConnected value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryIsGroundConnectedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsGroundConnectedChanged(VehiclePowertrainTractionBatteryIsGroundConnectedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsGroundConnected property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsGroundConnectedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for GrossCapacity value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryGrossCapacityPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToGrossCapacityChanged(VehiclePowertrainTractionBatteryGrossCapacityPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from GrossCapacity property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromGrossCapacityChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for NetCapacity value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryNetCapacityPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToNetCapacityChanged(VehiclePowertrainTractionBatteryNetCapacityPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from NetCapacity property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromNetCapacityChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for StateOfHealth value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryStateOfHealthPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToStateOfHealthChanged(VehiclePowertrainTractionBatteryStateOfHealthPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from StateOfHealth property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromStateOfHealthChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for NominalVoltage value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryNominalVoltagePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToNominalVoltageChanged(VehiclePowertrainTractionBatteryNominalVoltagePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from NominalVoltage property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromNominalVoltageChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for MaxVoltage value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryMaxVoltagePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMaxVoltageChanged(VehiclePowertrainTractionBatteryMaxVoltagePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from MaxVoltage property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMaxVoltageChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CurrentVoltage value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryCurrentVoltagePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCurrentVoltageChanged(VehiclePowertrainTractionBatteryCurrentVoltagePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CurrentVoltage property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCurrentVoltageChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CurrentCurrent value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryCurrentCurrentPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCurrentCurrentChanged(VehiclePowertrainTractionBatteryCurrentCurrentPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CurrentCurrent property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCurrentCurrentChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CurrentPower value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryCurrentPowerPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCurrentPowerChanged(VehiclePowertrainTractionBatteryCurrentPowerPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CurrentPower property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCurrentPowerChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AccumulatedChargedEnergy value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryAccumulatedChargedEnergyPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAccumulatedChargedEnergyChanged(VehiclePowertrainTractionBatteryAccumulatedChargedEnergyPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AccumulatedChargedEnergy property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAccumulatedChargedEnergyChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AccumulatedConsumedEnergy value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryAccumulatedConsumedEnergyPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAccumulatedConsumedEnergyChanged(VehiclePowertrainTractionBatteryAccumulatedConsumedEnergyPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AccumulatedConsumedEnergy property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAccumulatedConsumedEnergyChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AccumulatedChargedThroughput value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryAccumulatedChargedThroughputPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAccumulatedChargedThroughputChanged(VehiclePowertrainTractionBatteryAccumulatedChargedThroughputPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AccumulatedChargedThroughput property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAccumulatedChargedThroughputChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AccumulatedConsumedThroughput value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryAccumulatedConsumedThroughputPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAccumulatedConsumedThroughputChanged(VehiclePowertrainTractionBatteryAccumulatedConsumedThroughputPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AccumulatedConsumedThroughput property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAccumulatedConsumedThroughputChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for PowerLoss value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryPowerLossPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryPowerLossPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from PowerLoss property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPowerLossChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Range value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryRangePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToRangeChanged(VehiclePowertrainTractionBatteryRangePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Range property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromRangeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TimeRemaining value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryTimeRemainingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTimeRemainingChanged(VehiclePowertrainTractionBatteryTimeRemainingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TimeRemaining property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatterySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTimeRemainingChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property Id changes.
    * @param The new value of Id.
    * Battery Identification Number as assigned by OEM.
    */
    virtual void publishIdChanged(const std::string& Id) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property ProductionDate changes.
    * @param The new value of ProductionDate.
    * Production date of battery in ISO8601 format, e.g. YYYY-MM-DD.
    */
    virtual void publishProductionDateChanged(const std::string& ProductionDate) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property ErrorCodes changes.
    * @param The new value of ErrorCodes.
    * Current error codes related to the battery, if any.
    */
    virtual void publishErrorCodesChanged(const std::list<std::string>& ErrorCodes) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property IsPowerConnected changes.
    * @param The new value of IsPowerConnected.
    * Indicating if the power (positive terminator) of the traction battery is connected to the powertrain.
    */
    virtual void publishIsPowerConnectedChanged(bool IsPowerConnected) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property IsGroundConnected changes.
    * @param The new value of IsGroundConnected.
    * Indicating if the ground (negative terminator) of the traction battery is connected to the powertrain.
    */
    virtual void publishIsGroundConnectedChanged(bool IsGroundConnected) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property GrossCapacity changes.
    * @param The new value of GrossCapacity.
    * Gross capacity of the battery.
    */
    virtual void publishGrossCapacityChanged(int32_t GrossCapacity) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property NetCapacity changes.
    * @param The new value of NetCapacity.
    * Total net capacity of the battery considering aging.
    */
    virtual void publishNetCapacityChanged(int32_t NetCapacity) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property StateOfHealth changes.
    * @param The new value of StateOfHealth.
    * Calculated battery state of health at standard conditions.
    */
    virtual void publishStateOfHealthChanged(float StateOfHealth) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property NominalVoltage changes.
    * @param The new value of NominalVoltage.
    * Nominal Voltage of the battery.
    */
    virtual void publishNominalVoltageChanged(int32_t NominalVoltage) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property MaxVoltage changes.
    * @param The new value of MaxVoltage.
    * Max allowed voltage of the battery, e.g. during charging.
    */
    virtual void publishMaxVoltageChanged(int32_t MaxVoltage) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property CurrentVoltage changes.
    * @param The new value of CurrentVoltage.
    * Current Voltage of the battery.
    */
    virtual void publishCurrentVoltageChanged(float CurrentVoltage) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property CurrentCurrent changes.
    * @param The new value of CurrentCurrent.
    * Current current flowing in/out of battery. Positive = Current flowing in to battery, e.g. during charging. Negative = Current flowing out of battery, e.g. during driving.
    */
    virtual void publishCurrentCurrentChanged(float CurrentCurrent) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property CurrentPower changes.
    * @param The new value of CurrentPower.
    * Current electrical energy flowing in/out of battery. Positive = Energy flowing in to battery, e.g. during charging. Negative = Energy flowing out of battery, e.g. during driving.
    */
    virtual void publishCurrentPowerChanged(float CurrentPower) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property AccumulatedChargedEnergy changes.
    * @param The new value of AccumulatedChargedEnergy.
    * The accumulated energy delivered to the battery during charging over lifetime of the battery.
    */
    virtual void publishAccumulatedChargedEnergyChanged(float AccumulatedChargedEnergy) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property AccumulatedConsumedEnergy changes.
    * @param The new value of AccumulatedConsumedEnergy.
    * The accumulated energy leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    */
    virtual void publishAccumulatedConsumedEnergyChanged(float AccumulatedConsumedEnergy) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property AccumulatedChargedThroughput changes.
    * @param The new value of AccumulatedChargedThroughput.
    * The accumulated charge throughput delivered to the battery during charging over lifetime of the battery.
    */
    virtual void publishAccumulatedChargedThroughputChanged(float AccumulatedChargedThroughput) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property AccumulatedConsumedThroughput changes.
    * @param The new value of AccumulatedConsumedThroughput.
    * The accumulated charge throughput leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    */
    virtual void publishAccumulatedConsumedThroughputChanged(float AccumulatedConsumedThroughput) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property PowerLoss changes.
    * @param The new value of PowerLoss.
    * Electrical energy lost by power dissipation to heat inside the battery.
    */
    virtual void publishPowerLossChanged(float PowerLoss) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property Range changes.
    * @param The new value of Range.
    * Remaining range in meters using only battery.
    */
    virtual void publishRangeChanged(int32_t Range) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery implementation when property TimeRemaining changes.
    * @param The new value of TimeRemaining.
    * Time remaining in seconds before the battery is empty.
    */
    virtual void publishTimeRemainingChanged(int32_t TimeRemaining) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
