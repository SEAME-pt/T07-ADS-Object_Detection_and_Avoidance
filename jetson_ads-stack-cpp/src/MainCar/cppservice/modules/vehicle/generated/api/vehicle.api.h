#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleSubscriber;
class IVehiclePublisher;

/**
*
* IVehicle provides an interface for
 *  - methods defined for your Vehicle 
 *  - property setters and getters for defined properties
 * The IVehicle also provides an interface to access a publisher IVehiclePublisher, a class used by IVehicleSubscriber clients.
 * The implementation should notify the publisher IVehiclePublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleSubscriber, IVehiclePublisher
 * and the example implementation Vehicle  or the
 */
class CPP_VEHICLE_EXPORT IVehicle
{
public:
    virtual ~IVehicle() = default;

    /**
    * Sets the value of the LowVoltageSystemState property.
    * @param LowVoltageSystemState State of the supply voltage of the control units (usually 12V).
    */
    virtual void setLowVoltageSystemState(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState) = 0;
    /**
    * Gets the value of the LowVoltageSystemState property.
    * @return State of the supply voltage of the control units (usually 12V).
    */
    virtual Vehicle_LowVoltageSystemState_ValueEnum getLowVoltageSystemState() const = 0;

    /**
    * Sets the value of the Speed property.
    * @param Speed Vehicle speed.
    */
    virtual void setSpeed(float Speed) = 0;
    /**
    * Gets the value of the Speed property.
    * @return Vehicle speed.
    */
    virtual float getSpeed() const = 0;

    /**
    * Sets the value of the TraveledDistance property.
    * @param TraveledDistance Odometer reading, total distance traveled during the lifetime of the vehicle.
    */
    virtual void setTraveledDistance(float TraveledDistance) = 0;
    /**
    * Gets the value of the TraveledDistance property.
    * @return Odometer reading, total distance traveled during the lifetime of the vehicle.
    */
    virtual float getTraveledDistance() const = 0;

    /**
    * Sets the value of the TraveledDistanceSinceStart property.
    * @param TraveledDistanceSinceStart Distance traveled since start of current trip.
    */
    virtual void setTraveledDistanceSinceStart(float TraveledDistanceSinceStart) = 0;
    /**
    * Gets the value of the TraveledDistanceSinceStart property.
    * @return Distance traveled since start of current trip.
    */
    virtual float getTraveledDistanceSinceStart() const = 0;

    /**
    * Sets the value of the StartTime property.
    * @param StartTime Start time of current or latest trip, formatted according to ISO 8601 with UTC time zone.
    */
    virtual void setStartTime(const std::string& StartTime) = 0;
    /**
    * Gets the value of the StartTime property.
    * @return Start time of current or latest trip, formatted according to ISO 8601 with UTC time zone.
    */
    virtual const std::string& getStartTime() const = 0;

    /**
    * Sets the value of the TripDuration property.
    * @param TripDuration Duration of latest trip.
    */
    virtual void setTripDuration(float TripDuration) = 0;
    /**
    * Gets the value of the TripDuration property.
    * @return Duration of latest trip.
    */
    virtual float getTripDuration() const = 0;

    /**
    * Sets the value of the TripMeterReading property.
    * @param TripMeterReading Trip meter reading.
    */
    virtual void setTripMeterReading(float TripMeterReading) = 0;
    /**
    * Gets the value of the TripMeterReading property.
    * @return Trip meter reading.
    */
    virtual float getTripMeterReading() const = 0;

    /**
    * Sets the value of the IsBrokenDown property.
    * @param IsBrokenDown Vehicle breakdown or any similar event causing vehicle to stop on the road, that might pose a risk to other road users. True = Vehicle broken down on the road, due to e.g. engine problems, flat tire, out of gas, brake problems. False = Vehicle not broken down.
    */
    virtual void setIsBrokenDown(bool IsBrokenDown) = 0;
    /**
    * Gets the value of the IsBrokenDown property.
    * @return Vehicle breakdown or any similar event causing vehicle to stop on the road, that might pose a risk to other road users. True = Vehicle broken down on the road, due to e.g. engine problems, flat tire, out of gas, brake problems. False = Vehicle not broken down.
    */
    virtual bool getIsBrokenDown() const = 0;

    /**
    * Sets the value of the IsMoving property.
    * @param IsMoving Indicates whether the vehicle is stationary or moving.
    */
    virtual void setIsMoving(bool IsMoving) = 0;
    /**
    * Gets the value of the IsMoving property.
    * @return Indicates whether the vehicle is stationary or moving.
    */
    virtual bool getIsMoving() const = 0;

    /**
    * Sets the value of the AverageSpeed property.
    * @param AverageSpeed Average speed for the current trip.
    */
    virtual void setAverageSpeed(float AverageSpeed) = 0;
    /**
    * Gets the value of the AverageSpeed property.
    * @return Average speed for the current trip.
    */
    virtual float getAverageSpeed() const = 0;

    /**
    * Sets the value of the PowerOptimizeLevel property.
    * @param PowerOptimizeLevel Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    virtual void setPowerOptimizeLevel(int32_t PowerOptimizeLevel) = 0;
    /**
    * Gets the value of the PowerOptimizeLevel property.
    * @return Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    virtual int32_t getPowerOptimizeLevel() const = 0;

    /**
    * Sets the value of the IsAutoPowerOptimize property.
    * @param IsAutoPowerOptimize Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    virtual void setIsAutoPowerOptimize(bool IsAutoPowerOptimize) = 0;
    /**
    * Gets the value of the IsAutoPowerOptimize property.
    * @return Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    virtual bool getIsAutoPowerOptimize() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle.
    */
    virtual IVehiclePublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleSubscriber contains functions to allow informing about signals or property changes of the IVehicle implementation.
 * The implementation for IVehicle should provide mechanism for subscription of the IVehicleSubscriber clients.
 * See IVehiclePublisher, which provides facilitation for this purpose.
 * The implementation for IVehicle should call the IVehicleSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleSubscriber class to implement clients of the IVehicle or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleSubscriber
{
public:
    virtual ~IVehicleSubscriber() = default;
    /**
    * Called by the IVehiclePublisher when LowVoltageSystemState value has changed if subscribed for the LowVoltageSystemState change.
    * @param LowVoltageSystemState State of the supply voltage of the control units (usually 12V).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLowVoltageSystemStateChanged(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState) = 0;
    /**
    * Called by the IVehiclePublisher when Speed value has changed if subscribed for the Speed change.
    * @param Speed Vehicle speed.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onSpeedChanged(float Speed) = 0;
    /**
    * Called by the IVehiclePublisher when TraveledDistance value has changed if subscribed for the TraveledDistance change.
    * @param TraveledDistance Odometer reading, total distance traveled during the lifetime of the vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTraveledDistanceChanged(float TraveledDistance) = 0;
    /**
    * Called by the IVehiclePublisher when TraveledDistanceSinceStart value has changed if subscribed for the TraveledDistanceSinceStart change.
    * @param TraveledDistanceSinceStart Distance traveled since start of current trip.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTraveledDistanceSinceStartChanged(float TraveledDistanceSinceStart) = 0;
    /**
    * Called by the IVehiclePublisher when StartTime value has changed if subscribed for the StartTime change.
    * @param StartTime Start time of current or latest trip, formatted according to ISO 8601 with UTC time zone.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onStartTimeChanged(const std::string& StartTime) = 0;
    /**
    * Called by the IVehiclePublisher when TripDuration value has changed if subscribed for the TripDuration change.
    * @param TripDuration Duration of latest trip.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTripDurationChanged(float TripDuration) = 0;
    /**
    * Called by the IVehiclePublisher when TripMeterReading value has changed if subscribed for the TripMeterReading change.
    * @param TripMeterReading Trip meter reading.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTripMeterReadingChanged(float TripMeterReading) = 0;
    /**
    * Called by the IVehiclePublisher when IsBrokenDown value has changed if subscribed for the IsBrokenDown change.
    * @param IsBrokenDown Vehicle breakdown or any similar event causing vehicle to stop on the road, that might pose a risk to other road users. True = Vehicle broken down on the road, due to e.g. engine problems, flat tire, out of gas, brake problems. False = Vehicle not broken down.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsBrokenDownChanged(bool IsBrokenDown) = 0;
    /**
    * Called by the IVehiclePublisher when IsMoving value has changed if subscribed for the IsMoving change.
    * @param IsMoving Indicates whether the vehicle is stationary or moving.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsMovingChanged(bool IsMoving) = 0;
    /**
    * Called by the IVehiclePublisher when AverageSpeed value has changed if subscribed for the AverageSpeed change.
    * @param AverageSpeed Average speed for the current trip.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAverageSpeedChanged(float AverageSpeed) = 0;
    /**
    * Called by the IVehiclePublisher when PowerOptimizeLevel value has changed if subscribed for the PowerOptimizeLevel change.
    * @param PowerOptimizeLevel Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) = 0;
    /**
    * Called by the IVehiclePublisher when IsAutoPowerOptimize value has changed if subscribed for the IsAutoPowerOptimize change.
    * @param IsAutoPowerOptimize Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) = 0;
};

/** Callback for changes of LowVoltageSystemState */
using VehicleLowVoltageSystemStatePropertyCb = std::function<void(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState)>;
/** Callback for changes of Speed */
using VehicleSpeedPropertyCb = std::function<void(float Speed)>;
/** Callback for changes of TraveledDistance */
using VehicleTraveledDistancePropertyCb = std::function<void(float TraveledDistance)>;
/** Callback for changes of TraveledDistanceSinceStart */
using VehicleTraveledDistanceSinceStartPropertyCb = std::function<void(float TraveledDistanceSinceStart)>;
/** Callback for changes of StartTime */
using VehicleStartTimePropertyCb = std::function<void(const std::string& StartTime)>;
/** Callback for changes of TripDuration */
using VehicleTripDurationPropertyCb = std::function<void(float TripDuration)>;
/** Callback for changes of TripMeterReading */
using VehicleTripMeterReadingPropertyCb = std::function<void(float TripMeterReading)>;
/** Callback for changes of IsBrokenDown */
using VehicleIsBrokenDownPropertyCb = std::function<void(bool IsBrokenDown)>;
/** Callback for changes of IsMoving */
using VehicleIsMovingPropertyCb = std::function<void(bool IsMoving)>;
/** Callback for changes of AverageSpeed */
using VehicleAverageSpeedPropertyCb = std::function<void(float AverageSpeed)>;
/** Callback for changes of PowerOptimizeLevel */
using VehiclePowerOptimizeLevelPropertyCb = std::function<void(int32_t PowerOptimizeLevel)>;
/** Callback for changes of IsAutoPowerOptimize */
using VehicleIsAutoPowerOptimizePropertyCb = std::function<void(bool IsAutoPowerOptimize)>;


/**
 * The IVehiclePublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicle implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicle on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePublisher
{
public:
    virtual ~IVehiclePublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleSubscriber which is subscribed in this function to any change of the Vehicle.
    */
    virtual void subscribeToAllChanges(IVehicleSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleSubscriber which subscription for any change of the Vehicle is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for LowVoltageSystemState value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleLowVoltageSystemStatePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLowVoltageSystemStateChanged(VehicleLowVoltageSystemStatePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from LowVoltageSystemState property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLowVoltageSystemStateChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Speed value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleSpeedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToSpeedChanged(VehicleSpeedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Speed property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromSpeedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TraveledDistance value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleTraveledDistancePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTraveledDistanceChanged(VehicleTraveledDistancePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TraveledDistance property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTraveledDistanceChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TraveledDistanceSinceStart value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleTraveledDistanceSinceStartPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTraveledDistanceSinceStartChanged(VehicleTraveledDistanceSinceStartPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TraveledDistanceSinceStart property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTraveledDistanceSinceStartChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for StartTime value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleStartTimePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToStartTimeChanged(VehicleStartTimePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from StartTime property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromStartTimeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TripDuration value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleTripDurationPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTripDurationChanged(VehicleTripDurationPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TripDuration property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTripDurationChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TripMeterReading value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleTripMeterReadingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTripMeterReadingChanged(VehicleTripMeterReadingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TripMeterReading property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTripMeterReadingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsBrokenDown value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleIsBrokenDownPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsBrokenDownChanged(VehicleIsBrokenDownPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsBrokenDown property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsBrokenDownChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsMoving value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleIsMovingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsMovingChanged(VehicleIsMovingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsMoving property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsMovingChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AverageSpeed value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleAverageSpeedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAverageSpeedChanged(VehicleAverageSpeedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AverageSpeed property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAverageSpeedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for PowerOptimizeLevel value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowerOptimizeLevelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPowerOptimizeLevelChanged(VehiclePowerOptimizeLevelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from PowerOptimizeLevel property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPowerOptimizeLevelChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsAutoPowerOptimize value changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleIsAutoPowerOptimizePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsAutoPowerOptimizeChanged(VehicleIsAutoPowerOptimizePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsAutoPowerOptimize property changes.
    * If your subscriber uses subscription with IVehicleSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsAutoPowerOptimizeChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property LowVoltageSystemState changes.
    * @param The new value of LowVoltageSystemState.
    * State of the supply voltage of the control units (usually 12V).
    */
    virtual void publishLowVoltageSystemStateChanged(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property Speed changes.
    * @param The new value of Speed.
    * Vehicle speed.
    */
    virtual void publishSpeedChanged(float Speed) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property TraveledDistance changes.
    * @param The new value of TraveledDistance.
    * Odometer reading, total distance traveled during the lifetime of the vehicle.
    */
    virtual void publishTraveledDistanceChanged(float TraveledDistance) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property TraveledDistanceSinceStart changes.
    * @param The new value of TraveledDistanceSinceStart.
    * Distance traveled since start of current trip.
    */
    virtual void publishTraveledDistanceSinceStartChanged(float TraveledDistanceSinceStart) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property StartTime changes.
    * @param The new value of StartTime.
    * Start time of current or latest trip, formatted according to ISO 8601 with UTC time zone.
    */
    virtual void publishStartTimeChanged(const std::string& StartTime) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property TripDuration changes.
    * @param The new value of TripDuration.
    * Duration of latest trip.
    */
    virtual void publishTripDurationChanged(float TripDuration) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property TripMeterReading changes.
    * @param The new value of TripMeterReading.
    * Trip meter reading.
    */
    virtual void publishTripMeterReadingChanged(float TripMeterReading) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property IsBrokenDown changes.
    * @param The new value of IsBrokenDown.
    * Vehicle breakdown or any similar event causing vehicle to stop on the road, that might pose a risk to other road users. True = Vehicle broken down on the road, due to e.g. engine problems, flat tire, out of gas, brake problems. False = Vehicle not broken down.
    */
    virtual void publishIsBrokenDownChanged(bool IsBrokenDown) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property IsMoving changes.
    * @param The new value of IsMoving.
    * Indicates whether the vehicle is stationary or moving.
    */
    virtual void publishIsMovingChanged(bool IsMoving) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property AverageSpeed changes.
    * @param The new value of AverageSpeed.
    * Average speed for the current trip.
    */
    virtual void publishAverageSpeedChanged(float AverageSpeed) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property PowerOptimizeLevel changes.
    * @param The new value of PowerOptimizeLevel.
    * Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    virtual void publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle implementation when property IsAutoPowerOptimize changes.
    * @param The new value of IsAutoPowerOptimize.
    * Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    virtual void publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
