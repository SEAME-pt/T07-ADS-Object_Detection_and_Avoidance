#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryCellVoltageSubscriber;
class IVehiclePowertrainTractionBatteryCellVoltagePublisher;

/**
*
* IVehiclePowertrainTractionBatteryCellVoltage provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_CellVoltage 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryCellVoltage also provides an interface to access a publisher IVehiclePowertrainTractionBatteryCellVoltagePublisher, a class used by IVehiclePowertrainTractionBatteryCellVoltageSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryCellVoltagePublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryCellVoltageSubscriber, IVehiclePowertrainTractionBatteryCellVoltagePublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_CellVoltage  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryCellVoltage
{
public:
    virtual ~IVehiclePowertrainTractionBatteryCellVoltage() = default;

    /**
    * Sets the value of the Min property.
    * @param Min Current voltage of the battery cell with lowest voltage.
    */
    virtual void setMin(float Min) = 0;
    /**
    * Gets the value of the Min property.
    * @return Current voltage of the battery cell with lowest voltage.
    */
    virtual float getMin() const = 0;

    /**
    * Sets the value of the IdMin property.
    * @param IdMin Identifier of the battery cell with lowest voltage.
    */
    virtual void setIdMin(int32_t IdMin) = 0;
    /**
    * Gets the value of the IdMin property.
    * @return Identifier of the battery cell with lowest voltage.
    */
    virtual int32_t getIdMin() const = 0;

    /**
    * Sets the value of the Max property.
    * @param Max Current voltage of the battery cell with highest voltage.
    */
    virtual void setMax(float Max) = 0;
    /**
    * Gets the value of the Max property.
    * @return Current voltage of the battery cell with highest voltage.
    */
    virtual float getMax() const = 0;

    /**
    * Sets the value of the IdMax property.
    * @param IdMax Identifier of the battery cell with highest voltage.
    */
    virtual void setIdMax(int32_t IdMax) = 0;
    /**
    * Gets the value of the IdMax property.
    * @return Identifier of the battery cell with highest voltage.
    */
    virtual int32_t getIdMax() const = 0;

    /**
    * Sets the value of the CellVoltages property.
    * @param CellVoltages Array of cell voltages. Length or array shall correspond to number of cells in vehicle.
    */
    virtual void setCellVoltages(const std::list<float>& CellVoltages) = 0;
    /**
    * Gets the value of the CellVoltages property.
    * @return Array of cell voltages. Length or array shall correspond to number of cells in vehicle.
    */
    virtual const std::list<float>& getCellVoltages() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_CellVoltage changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_CellVoltage.
    */
    virtual IVehiclePowertrainTractionBatteryCellVoltagePublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryCellVoltageSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryCellVoltage implementation.
 * The implementation for IVehiclePowertrainTractionBatteryCellVoltage should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryCellVoltageSubscriber clients.
 * See IVehiclePowertrainTractionBatteryCellVoltagePublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryCellVoltage should call the IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryCellVoltageSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryCellVoltage or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryCellVoltageSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryCellVoltageSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryCellVoltagePublisher when Min value has changed if subscribed for the Min change.
    * @param Min Current voltage of the battery cell with lowest voltage.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMinChanged(float Min) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryCellVoltagePublisher when IdMin value has changed if subscribed for the IdMin change.
    * @param IdMin Identifier of the battery cell with lowest voltage.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIdMinChanged(int32_t IdMin) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryCellVoltagePublisher when Max value has changed if subscribed for the Max change.
    * @param Max Current voltage of the battery cell with highest voltage.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMaxChanged(float Max) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryCellVoltagePublisher when IdMax value has changed if subscribed for the IdMax change.
    * @param IdMax Identifier of the battery cell with highest voltage.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIdMaxChanged(int32_t IdMax) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryCellVoltagePublisher when CellVoltages value has changed if subscribed for the CellVoltages change.
    * @param CellVoltages Array of cell voltages. Length or array shall correspond to number of cells in vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCellVoltagesChanged(const std::list<float>& CellVoltages) = 0;
};

/** Callback for changes of Min */
using VehiclePowertrainTractionBatteryCellVoltageMinPropertyCb = std::function<void(float Min)>;
/** Callback for changes of IdMin */
using VehiclePowertrainTractionBatteryCellVoltageIdMinPropertyCb = std::function<void(int32_t IdMin)>;
/** Callback for changes of Max */
using VehiclePowertrainTractionBatteryCellVoltageMaxPropertyCb = std::function<void(float Max)>;
/** Callback for changes of IdMax */
using VehiclePowertrainTractionBatteryCellVoltageIdMaxPropertyCb = std::function<void(int32_t IdMax)>;
/** Callback for changes of CellVoltages */
using VehiclePowertrainTractionBatteryCellVoltageCellVoltagesPropertyCb = std::function<void(const std::list<float>& CellVoltages)>;


/**
 * The IVehiclePowertrainTractionBatteryCellVoltagePublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryCellVoltage implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryCellVoltageSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryCellVoltage on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryCellVoltagePublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryCellVoltagePublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_CellVoltage.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryCellVoltageSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_CellVoltage.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryCellVoltageSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_CellVoltage.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryCellVoltageSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_CellVoltage is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryCellVoltageSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Min value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryCellVoltageMinPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMinChanged(VehiclePowertrainTractionBatteryCellVoltageMinPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Min property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMinChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IdMin value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryCellVoltageIdMinPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIdMinChanged(VehiclePowertrainTractionBatteryCellVoltageIdMinPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IdMin property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIdMinChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Max value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryCellVoltageMaxPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMaxChanged(VehiclePowertrainTractionBatteryCellVoltageMaxPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Max property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMaxChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IdMax value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryCellVoltageIdMaxPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIdMaxChanged(VehiclePowertrainTractionBatteryCellVoltageIdMaxPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IdMax property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIdMaxChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CellVoltages value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryCellVoltageCellVoltagesPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCellVoltagesChanged(VehiclePowertrainTractionBatteryCellVoltageCellVoltagesPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CellVoltages property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryCellVoltageSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCellVoltagesChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_CellVoltage implementation when property Min changes.
    * @param The new value of Min.
    * Current voltage of the battery cell with lowest voltage.
    */
    virtual void publishMinChanged(float Min) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_CellVoltage implementation when property IdMin changes.
    * @param The new value of IdMin.
    * Identifier of the battery cell with lowest voltage.
    */
    virtual void publishIdMinChanged(int32_t IdMin) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_CellVoltage implementation when property Max changes.
    * @param The new value of Max.
    * Current voltage of the battery cell with highest voltage.
    */
    virtual void publishMaxChanged(float Max) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_CellVoltage implementation when property IdMax changes.
    * @param The new value of IdMax.
    * Identifier of the battery cell with highest voltage.
    */
    virtual void publishIdMaxChanged(int32_t IdMax) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_CellVoltage implementation when property CellVoltages changes.
    * @param The new value of CellVoltages.
    * Array of cell voltages. Length or array shall correspond to number of cells in vehicle.
    */
    virtual void publishCellVoltagesChanged(const std::list<float>& CellVoltages) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
