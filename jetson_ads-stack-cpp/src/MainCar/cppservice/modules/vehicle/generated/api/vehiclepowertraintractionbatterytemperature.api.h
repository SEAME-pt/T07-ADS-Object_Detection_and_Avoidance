#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTractionBatteryTemperatureSubscriber;
class IVehiclePowertrainTractionBatteryTemperaturePublisher;

/**
*
* IVehiclePowertrainTractionBatteryTemperature provides an interface for
 *  - methods defined for your Vehicle_Powertrain_TractionBattery_Temperature 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTractionBatteryTemperature also provides an interface to access a publisher IVehiclePowertrainTractionBatteryTemperaturePublisher, a class used by IVehiclePowertrainTractionBatteryTemperatureSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTractionBatteryTemperaturePublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTractionBatteryTemperatureSubscriber, IVehiclePowertrainTractionBatteryTemperaturePublisher
 * and the example implementation Vehicle_Powertrain_TractionBattery_Temperature  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryTemperature
{
public:
    virtual ~IVehiclePowertrainTractionBatteryTemperature() = default;

    /**
    * Sets the value of the Average property.
    * @param Average Current average temperature of the battery cells.
    */
    virtual void setAverage(float Average) = 0;
    /**
    * Gets the value of the Average property.
    * @return Current average temperature of the battery cells.
    */
    virtual float getAverage() const = 0;

    /**
    * Sets the value of the Min property.
    * @param Min Current minimum temperature of the battery cells, i.e. temperature of the coldest cell.
    */
    virtual void setMin(float Min) = 0;
    /**
    * Gets the value of the Min property.
    * @return Current minimum temperature of the battery cells, i.e. temperature of the coldest cell.
    */
    virtual float getMin() const = 0;

    /**
    * Sets the value of the Max property.
    * @param Max Current maximum temperature of the battery cells, i.e. temperature of the hottest cell.
    */
    virtual void setMax(float Max) = 0;
    /**
    * Gets the value of the Max property.
    * @return Current maximum temperature of the battery cells, i.e. temperature of the hottest cell.
    */
    virtual float getMax() const = 0;

    /**
    * Sets the value of the CellTemperature property.
    * @param CellTemperature Array of cell temperatures. Length or array shall correspond to number of cells in vehicle.
    */
    virtual void setCellTemperature(const std::list<float>& CellTemperature) = 0;
    /**
    * Gets the value of the CellTemperature property.
    * @return Array of cell temperatures. Length or array shall correspond to number of cells in vehicle.
    */
    virtual const std::list<float>& getCellTemperature() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Temperature changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Temperature.
    */
    virtual IVehiclePowertrainTractionBatteryTemperaturePublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTractionBatteryTemperatureSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTractionBatteryTemperature implementation.
 * The implementation for IVehiclePowertrainTractionBatteryTemperature should provide mechanism for subscription of the IVehiclePowertrainTractionBatteryTemperatureSubscriber clients.
 * See IVehiclePowertrainTractionBatteryTemperaturePublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTractionBatteryTemperature should call the IVehiclePowertrainTractionBatteryTemperatureSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTractionBatteryTemperatureSubscriber class to implement clients of the IVehiclePowertrainTractionBatteryTemperature or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryTemperatureSubscriber
{
public:
    virtual ~IVehiclePowertrainTractionBatteryTemperatureSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTractionBatteryTemperaturePublisher when Average value has changed if subscribed for the Average change.
    * @param Average Current average temperature of the battery cells.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAverageChanged(float Average) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryTemperaturePublisher when Min value has changed if subscribed for the Min change.
    * @param Min Current minimum temperature of the battery cells, i.e. temperature of the coldest cell.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMinChanged(float Min) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryTemperaturePublisher when Max value has changed if subscribed for the Max change.
    * @param Max Current maximum temperature of the battery cells, i.e. temperature of the hottest cell.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMaxChanged(float Max) = 0;
    /**
    * Called by the IVehiclePowertrainTractionBatteryTemperaturePublisher when CellTemperature value has changed if subscribed for the CellTemperature change.
    * @param CellTemperature Array of cell temperatures. Length or array shall correspond to number of cells in vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCellTemperatureChanged(const std::list<float>& CellTemperature) = 0;
};

/** Callback for changes of Average */
using VehiclePowertrainTractionBatteryTemperatureAveragePropertyCb = std::function<void(float Average)>;
/** Callback for changes of Min */
using VehiclePowertrainTractionBatteryTemperatureMinPropertyCb = std::function<void(float Min)>;
/** Callback for changes of Max */
using VehiclePowertrainTractionBatteryTemperatureMaxPropertyCb = std::function<void(float Max)>;
/** Callback for changes of CellTemperature */
using VehiclePowertrainTractionBatteryTemperatureCellTemperaturePropertyCb = std::function<void(const std::list<float>& CellTemperature)>;


/**
 * The IVehiclePowertrainTractionBatteryTemperaturePublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTractionBatteryTemperature implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTractionBatteryTemperatureSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTractionBatteryTemperature on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTractionBatteryTemperaturePublisher
{
public:
    virtual ~IVehiclePowertrainTractionBatteryTemperaturePublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_TractionBattery_Temperature.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTractionBatteryTemperatureSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_TractionBattery_Temperature.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTractionBatteryTemperatureSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_TractionBattery_Temperature.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTractionBatteryTemperatureSubscriber which subscription for any change of the Vehicle_Powertrain_TractionBattery_Temperature is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryTemperatureSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Average value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryTemperatureSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryTemperatureAveragePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAverageChanged(VehiclePowertrainTractionBatteryTemperatureAveragePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Average property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryTemperatureSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAverageChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Min value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryTemperatureSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryTemperatureMinPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMinChanged(VehiclePowertrainTractionBatteryTemperatureMinPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Min property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryTemperatureSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMinChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Max value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryTemperatureSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryTemperatureMaxPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMaxChanged(VehiclePowertrainTractionBatteryTemperatureMaxPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Max property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryTemperatureSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMaxChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CellTemperature value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryTemperatureSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTractionBatteryTemperatureCellTemperaturePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCellTemperatureChanged(VehiclePowertrainTractionBatteryTemperatureCellTemperaturePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CellTemperature property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTractionBatteryTemperatureSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCellTemperatureChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Temperature implementation when property Average changes.
    * @param The new value of Average.
    * Current average temperature of the battery cells.
    */
    virtual void publishAverageChanged(float Average) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Temperature implementation when property Min changes.
    * @param The new value of Min.
    * Current minimum temperature of the battery cells, i.e. temperature of the coldest cell.
    */
    virtual void publishMinChanged(float Min) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Temperature implementation when property Max changes.
    * @param The new value of Max.
    * Current maximum temperature of the battery cells, i.e. temperature of the hottest cell.
    */
    virtual void publishMaxChanged(float Max) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_TractionBattery_Temperature implementation when property CellTemperature changes.
    * @param The new value of CellTemperature.
    * Array of cell temperatures. Length or array shall correspond to number of cells in vehicle.
    */
    virtual void publishCellTemperatureChanged(const std::list<float>& CellTemperature) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
