#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleBodyLightsBackupSubscriber;
class IVehicleBodyLightsBackupPublisher;

/**
*
* IVehicleBodyLightsBackup provides an interface for
 *  - methods defined for your Vehicle_Body_Lights_Backup 
 *  - property setters and getters for defined properties
 * The IVehicleBodyLightsBackup also provides an interface to access a publisher IVehicleBodyLightsBackupPublisher, a class used by IVehicleBodyLightsBackupSubscriber clients.
 * The implementation should notify the publisher IVehicleBodyLightsBackupPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleBodyLightsBackupSubscriber, IVehicleBodyLightsBackupPublisher
 * and the example implementation Vehicle_Body_Lights_Backup  or the
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBackup
{
public:
    virtual ~IVehicleBodyLightsBackup() = default;

    /**
    * Sets the value of the IsOn property.
    * @param IsOn Indicates if light is on or off. True = On. False = Off.
    */
    virtual void setIsOn(bool IsOn) = 0;
    /**
    * Gets the value of the IsOn property.
    * @return Indicates if light is on or off. True = On. False = Off.
    */
    virtual bool getIsOn() const = 0;

    /**
    * Sets the value of the IsDefect property.
    * @param IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual void setIsDefect(bool IsDefect) = 0;
    /**
    * Gets the value of the IsDefect property.
    * @return Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual bool getIsDefect() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Backup changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Body_Lights_Backup.
    */
    virtual IVehicleBodyLightsBackupPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleBodyLightsBackupSubscriber contains functions to allow informing about signals or property changes of the IVehicleBodyLightsBackup implementation.
 * The implementation for IVehicleBodyLightsBackup should provide mechanism for subscription of the IVehicleBodyLightsBackupSubscriber clients.
 * See IVehicleBodyLightsBackupPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleBodyLightsBackup should call the IVehicleBodyLightsBackupSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleBodyLightsBackupSubscriber class to implement clients of the IVehicleBodyLightsBackup or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBackupSubscriber
{
public:
    virtual ~IVehicleBodyLightsBackupSubscriber() = default;
    /**
    * Called by the IVehicleBodyLightsBackupPublisher when IsOn value has changed if subscribed for the IsOn change.
    * @param IsOn Indicates if light is on or off. True = On. False = Off.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsOnChanged(bool IsOn) = 0;
    /**
    * Called by the IVehicleBodyLightsBackupPublisher when IsDefect value has changed if subscribed for the IsDefect change.
    * @param IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsDefectChanged(bool IsDefect) = 0;
};

/** Callback for changes of IsOn */
using VehicleBodyLightsBackupIsOnPropertyCb = std::function<void(bool IsOn)>;
/** Callback for changes of IsDefect */
using VehicleBodyLightsBackupIsDefectPropertyCb = std::function<void(bool IsDefect)>;


/**
 * The IVehicleBodyLightsBackupPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleBodyLightsBackup implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleBodyLightsBackupSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleBodyLightsBackup on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleBodyLightsBackupPublisher
{
public:
    virtual ~IVehicleBodyLightsBackupPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Body_Lights_Backup.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleBodyLightsBackupSubscriber which is subscribed in this function to any change of the Vehicle_Body_Lights_Backup.
    */
    virtual void subscribeToAllChanges(IVehicleBodyLightsBackupSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Body_Lights_Backup.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleBodyLightsBackupSubscriber which subscription for any change of the Vehicle_Body_Lights_Backup is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleBodyLightsBackupSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for IsOn value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBackupSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsBackupIsOnPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsOnChanged(VehicleBodyLightsBackupIsOnPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsOn property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBackupSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsOnChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsDefect value changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBackupSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleBodyLightsBackupIsDefectPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsDefectChanged(VehicleBodyLightsBackupIsDefectPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsDefect property changes.
    * If your subscriber uses subscription with IVehicleBodyLightsBackupSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsDefectChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Backup implementation when property IsOn changes.
    * @param The new value of IsOn.
    * Indicates if light is on or off. True = On. False = Off.
    */
    virtual void publishIsOnChanged(bool IsOn) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Body_Lights_Backup implementation when property IsDefect changes.
    * @param The new value of IsDefect.
    * Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    virtual void publishIsDefectChanged(bool IsDefect) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
