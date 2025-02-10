#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleDriverSubscriber;
class IVehicleDriverPublisher;

/**
*
* IVehicleDriver provides an interface for
 *  - methods defined for your Vehicle_Driver 
 *  - property setters and getters for defined properties
 * The IVehicleDriver also provides an interface to access a publisher IVehicleDriverPublisher, a class used by IVehicleDriverSubscriber clients.
 * The implementation should notify the publisher IVehicleDriverPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleDriverSubscriber, IVehicleDriverPublisher
 * and the example implementation Vehicle_Driver  or the
 */
class CPP_VEHICLE_EXPORT IVehicleDriver
{
public:
    virtual ~IVehicleDriver() = default;

    /**
    * Sets the value of the DistractionLevel property.
    * @param DistractionLevel Distraction level of the driver, which can be evaluated by multiple factors e.g. driving situation, acoustical or optical signals inside the cockpit, ongoing phone calls.
    */
    virtual void setDistractionLevel(float DistractionLevel) = 0;
    /**
    * Gets the value of the DistractionLevel property.
    * @return Distraction level of the driver, which can be evaluated by multiple factors e.g. driving situation, acoustical or optical signals inside the cockpit, ongoing phone calls.
    */
    virtual float getDistractionLevel() const = 0;

    /**
    * Sets the value of the IsEyesOnRoad property.
    * @param IsEyesOnRoad Has driver the eyes on road or not?
    */
    virtual void setIsEyesOnRoad(bool IsEyesOnRoad) = 0;
    /**
    * Gets the value of the IsEyesOnRoad property.
    * @return Has driver the eyes on road or not?
    */
    virtual bool getIsEyesOnRoad() const = 0;

    /**
    * Sets the value of the IsHandsOnWheel property.
    * @param IsHandsOnWheel Are the driver's hands on the steering wheel or not?
    */
    virtual void setIsHandsOnWheel(bool IsHandsOnWheel) = 0;
    /**
    * Gets the value of the IsHandsOnWheel property.
    * @return Are the driver's hands on the steering wheel or not?
    */
    virtual bool getIsHandsOnWheel() const = 0;

    /**
    * Sets the value of the AttentiveProbability property.
    * @param AttentiveProbability Probability of attentiveness of the driver.
    */
    virtual void setAttentiveProbability(float AttentiveProbability) = 0;
    /**
    * Gets the value of the AttentiveProbability property.
    * @return Probability of attentiveness of the driver.
    */
    virtual float getAttentiveProbability() const = 0;

    /**
    * Sets the value of the FatigueLevel property.
    * @param FatigueLevel Fatigue level of the driver, which can be evaluated by multiple factors e.g. trip time, behaviour of steering, eye status.
    */
    virtual void setFatigueLevel(float FatigueLevel) = 0;
    /**
    * Gets the value of the FatigueLevel property.
    * @return Fatigue level of the driver, which can be evaluated by multiple factors e.g. trip time, behaviour of steering, eye status.
    */
    virtual float getFatigueLevel() const = 0;

    /**
    * Sets the value of the HeartRate property.
    * @param HeartRate Heart rate of the driver.
    */
    virtual void setHeartRate(int32_t HeartRate) = 0;
    /**
    * Gets the value of the HeartRate property.
    * @return Heart rate of the driver.
    */
    virtual int32_t getHeartRate() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Driver changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Driver.
    */
    virtual IVehicleDriverPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleDriverSubscriber contains functions to allow informing about signals or property changes of the IVehicleDriver implementation.
 * The implementation for IVehicleDriver should provide mechanism for subscription of the IVehicleDriverSubscriber clients.
 * See IVehicleDriverPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleDriver should call the IVehicleDriverSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleDriverSubscriber class to implement clients of the IVehicleDriver or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleDriverSubscriber
{
public:
    virtual ~IVehicleDriverSubscriber() = default;
    /**
    * Called by the IVehicleDriverPublisher when DistractionLevel value has changed if subscribed for the DistractionLevel change.
    * @param DistractionLevel Distraction level of the driver, which can be evaluated by multiple factors e.g. driving situation, acoustical or optical signals inside the cockpit, ongoing phone calls.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDistractionLevelChanged(float DistractionLevel) = 0;
    /**
    * Called by the IVehicleDriverPublisher when IsEyesOnRoad value has changed if subscribed for the IsEyesOnRoad change.
    * @param IsEyesOnRoad Has driver the eyes on road or not?
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsEyesOnRoadChanged(bool IsEyesOnRoad) = 0;
    /**
    * Called by the IVehicleDriverPublisher when IsHandsOnWheel value has changed if subscribed for the IsHandsOnWheel change.
    * @param IsHandsOnWheel Are the driver's hands on the steering wheel or not?
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsHandsOnWheelChanged(bool IsHandsOnWheel) = 0;
    /**
    * Called by the IVehicleDriverPublisher when AttentiveProbability value has changed if subscribed for the AttentiveProbability change.
    * @param AttentiveProbability Probability of attentiveness of the driver.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAttentiveProbabilityChanged(float AttentiveProbability) = 0;
    /**
    * Called by the IVehicleDriverPublisher when FatigueLevel value has changed if subscribed for the FatigueLevel change.
    * @param FatigueLevel Fatigue level of the driver, which can be evaluated by multiple factors e.g. trip time, behaviour of steering, eye status.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onFatigueLevelChanged(float FatigueLevel) = 0;
    /**
    * Called by the IVehicleDriverPublisher when HeartRate value has changed if subscribed for the HeartRate change.
    * @param HeartRate Heart rate of the driver.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onHeartRateChanged(int32_t HeartRate) = 0;
};

/** Callback for changes of DistractionLevel */
using VehicleDriverDistractionLevelPropertyCb = std::function<void(float DistractionLevel)>;
/** Callback for changes of IsEyesOnRoad */
using VehicleDriverIsEyesOnRoadPropertyCb = std::function<void(bool IsEyesOnRoad)>;
/** Callback for changes of IsHandsOnWheel */
using VehicleDriverIsHandsOnWheelPropertyCb = std::function<void(bool IsHandsOnWheel)>;
/** Callback for changes of AttentiveProbability */
using VehicleDriverAttentiveProbabilityPropertyCb = std::function<void(float AttentiveProbability)>;
/** Callback for changes of FatigueLevel */
using VehicleDriverFatigueLevelPropertyCb = std::function<void(float FatigueLevel)>;
/** Callback for changes of HeartRate */
using VehicleDriverHeartRatePropertyCb = std::function<void(int32_t HeartRate)>;


/**
 * The IVehicleDriverPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleDriver implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleDriverSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleDriver on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleDriverPublisher
{
public:
    virtual ~IVehicleDriverPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Driver.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleDriverSubscriber which is subscribed in this function to any change of the Vehicle_Driver.
    */
    virtual void subscribeToAllChanges(IVehicleDriverSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Driver.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleDriverSubscriber which subscription for any change of the Vehicle_Driver is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleDriverSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for DistractionLevel value changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDriverDistractionLevelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDistractionLevelChanged(VehicleDriverDistractionLevelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DistractionLevel property changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDistractionLevelChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsEyesOnRoad value changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDriverIsEyesOnRoadPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsEyesOnRoadChanged(VehicleDriverIsEyesOnRoadPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsEyesOnRoad property changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsEyesOnRoadChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsHandsOnWheel value changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDriverIsHandsOnWheelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsHandsOnWheelChanged(VehicleDriverIsHandsOnWheelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsHandsOnWheel property changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsHandsOnWheelChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AttentiveProbability value changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDriverAttentiveProbabilityPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAttentiveProbabilityChanged(VehicleDriverAttentiveProbabilityPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AttentiveProbability property changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAttentiveProbabilityChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for FatigueLevel value changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDriverFatigueLevelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToFatigueLevelChanged(VehicleDriverFatigueLevelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from FatigueLevel property changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromFatigueLevelChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for HeartRate value changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDriverHeartRatePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToHeartRateChanged(VehicleDriverHeartRatePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from HeartRate property changes.
    * If your subscriber uses subscription with IVehicleDriverSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromHeartRateChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Driver implementation when property DistractionLevel changes.
    * @param The new value of DistractionLevel.
    * Distraction level of the driver, which can be evaluated by multiple factors e.g. driving situation, acoustical or optical signals inside the cockpit, ongoing phone calls.
    */
    virtual void publishDistractionLevelChanged(float DistractionLevel) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Driver implementation when property IsEyesOnRoad changes.
    * @param The new value of IsEyesOnRoad.
    * Has driver the eyes on road or not?
    */
    virtual void publishIsEyesOnRoadChanged(bool IsEyesOnRoad) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Driver implementation when property IsHandsOnWheel changes.
    * @param The new value of IsHandsOnWheel.
    * Are the driver's hands on the steering wheel or not?
    */
    virtual void publishIsHandsOnWheelChanged(bool IsHandsOnWheel) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Driver implementation when property AttentiveProbability changes.
    * @param The new value of AttentiveProbability.
    * Probability of attentiveness of the driver.
    */
    virtual void publishAttentiveProbabilityChanged(float AttentiveProbability) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Driver implementation when property FatigueLevel changes.
    * @param The new value of FatigueLevel.
    * Fatigue level of the driver, which can be evaluated by multiple factors e.g. trip time, behaviour of steering, eye status.
    */
    virtual void publishFatigueLevelChanged(float FatigueLevel) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Driver implementation when property HeartRate changes.
    * @param The new value of HeartRate.
    * Heart rate of the driver.
    */
    virtual void publishHeartRateChanged(int32_t HeartRate) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
