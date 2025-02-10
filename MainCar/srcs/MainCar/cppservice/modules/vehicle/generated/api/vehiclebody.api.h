#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodySubscriber;
class IVehicleBodyPublisher;

/**
*
* IVehicleBody provides an interface for
 *  - methods defined for your Vehicle_Body 
 *  - property setters and getters for defined properties
 * The IVehicleBody also provides an interface to access a publisher IVehicleBodyPublisher, a class used by IVehicleBodySubscriber clients.
 * The implementation should notify the publisher IVehicleBodyPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodySubscriber, IVehicleBodyPublisher
 * and the example implementation Vehicle_Body  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBody
{
public:
    virtual ~IVehicleBody() = default;

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
    * Access to a publisher, use it to subscribe for Vehicle_Body changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body.
    */
    virtual IVehicleBodyPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodySubscriber contains functions to allow informing about signals or property changes of the IVehicleBody implementation.
 * The implementation for IVehicleBody should provide mechanism for subscription of the IVehicleBodySubscriber clients.
 * See IVehicleBodyPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBody should call the IVehicleBodySubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodySubscriber class to implement clients of the IVehicleBody or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodySubscriber
{
public:
    virtual ~IVehicleBodySubscriber() = default;
    /**
    * Called by the IVehicleBodyPublisher when PowerOptimizeLevel value has changed if subscribed for the PowerOptimizeLevel change.
    * @param PowerOptimizeLevel Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) = 0;
    /**
    * Called by the IVehicleBodyPublisher when IsAutoPowerOptimize value has changed if subscribed for the IsAutoPowerOptimize change.
    * @param IsAutoPowerOptimize Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) = 0;
};

/** Callback for changes of PowerOptimizeLevel */
using VehicleBodyPowerOptimizeLevelPropertyCb = std::function<void(int32_t PowerOptimizeLevel)>;
/** Callback for changes of IsAutoPowerOptimize */
using VehicleBodyIsAutoPowerOptimizePropertyCb = std::function<void(bool IsAutoPowerOptimize)>;


/**
 * The IVehicleBodyPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBody implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodySubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBody on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyPublisher
{
public:
    virtual ~IVehicleBodyPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodySubscriber which is subscribed in this function to any change of the Vehicle_Body.
    */
    virtual void subscribeToAllChanges(IVehicleBodySubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodySubscriber which subscription for any change of the Vehicle_Body is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodySubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for PowerOptimizeLevel value changes.
    * If your subscriber uses subscription with IVehicleBodySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyPowerOptimizeLevelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPowerOptimizeLevelChanged(VehicleBodyPowerOptimizeLevelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from PowerOptimizeLevel property changes.
    * If your subscriber uses subscription with IVehicleBodySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPowerOptimizeLevelChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsAutoPowerOptimize value changes.
    * If your subscriber uses subscription with IVehicleBodySubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyIsAutoPowerOptimizePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsAutoPowerOptimizeChanged(VehicleBodyIsAutoPowerOptimizePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsAutoPowerOptimize property changes.
    * If your subscriber uses subscription with IVehicleBodySubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsAutoPowerOptimizeChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body implementation when property PowerOptimizeLevel changes.
    * @param The new value of PowerOptimizeLevel.
    * Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    virtual void publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body implementation when property IsAutoPowerOptimize changes.
    * @param The new value of IsAutoPowerOptimize.
    * Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    virtual void publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
