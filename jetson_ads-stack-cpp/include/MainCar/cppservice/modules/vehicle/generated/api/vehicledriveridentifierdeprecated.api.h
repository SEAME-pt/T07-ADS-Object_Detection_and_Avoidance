#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleDriverIdentifierDeprecatedSubscriber;
class IVehicleDriverIdentifierDeprecatedPublisher;

/**
*
* IVehicleDriverIdentifierDeprecated provides an interface for
 *  - methods defined for your Vehicle_Driver_Identifier_Deprecated 
 *  - property setters and getters for defined properties
 * The IVehicleDriverIdentifierDeprecated also provides an interface to access a publisher IVehicleDriverIdentifierDeprecatedPublisher, a class used by IVehicleDriverIdentifierDeprecatedSubscriber clients.
 * The implementation should notify the publisher IVehicleDriverIdentifierDeprecatedPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleDriverIdentifierDeprecatedSubscriber, IVehicleDriverIdentifierDeprecatedPublisher
 * and the example implementation Vehicle_Driver_Identifier_Deprecated  or the
 */
class CPP_VEHICLE_EXPORT IVehicleDriverIdentifierDeprecated
{
public:
    virtual ~IVehicleDriverIdentifierDeprecated() = default;

    /**
    * Sets the value of the Subject_Deprecated property.
    * @param Subject_Deprecated Subject for the authentication of the occupant e.g. UserID 7331677.
    */
    virtual void setSubjectDeprecated(const std::string& Subject_Deprecated) = 0;
    /**
    * Gets the value of the Subject_Deprecated property.
    * @return Subject for the authentication of the occupant e.g. UserID 7331677.
    */
    virtual const std::string& getSubjectDeprecated() const = 0;

    /**
    * Sets the value of the Issuer_Deprecated property.
    * @param Issuer_Deprecated Unique Issuer for the authentication of the occupant e.g. https://accounts.funcorp.com.
    */
    virtual void setIssuerDeprecated(const std::string& Issuer_Deprecated) = 0;
    /**
    * Gets the value of the Issuer_Deprecated property.
    * @return Unique Issuer for the authentication of the occupant e.g. https://accounts.funcorp.com.
    */
    virtual const std::string& getIssuerDeprecated() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Driver_Identifier_Deprecated changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Driver_Identifier_Deprecated.
    */
    virtual IVehicleDriverIdentifierDeprecatedPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleDriverIdentifierDeprecatedSubscriber contains functions to allow informing about signals or property changes of the IVehicleDriverIdentifierDeprecated implementation.
 * The implementation for IVehicleDriverIdentifierDeprecated should provide mechanism for subscription of the IVehicleDriverIdentifierDeprecatedSubscriber clients.
 * See IVehicleDriverIdentifierDeprecatedPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleDriverIdentifierDeprecated should call the IVehicleDriverIdentifierDeprecatedSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleDriverIdentifierDeprecatedSubscriber class to implement clients of the IVehicleDriverIdentifierDeprecated or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleDriverIdentifierDeprecatedSubscriber
{
public:
    virtual ~IVehicleDriverIdentifierDeprecatedSubscriber() = default;
    /**
    * Called by the IVehicleDriverIdentifierDeprecatedPublisher when Subject_Deprecated value has changed if subscribed for the Subject_Deprecated change.
    * @param Subject_Deprecated Subject for the authentication of the occupant e.g. UserID 7331677.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onSubjectDeprecatedChanged(const std::string& Subject_Deprecated) = 0;
    /**
    * Called by the IVehicleDriverIdentifierDeprecatedPublisher when Issuer_Deprecated value has changed if subscribed for the Issuer_Deprecated change.
    * @param Issuer_Deprecated Unique Issuer for the authentication of the occupant e.g. https://accounts.funcorp.com.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIssuerDeprecatedChanged(const std::string& Issuer_Deprecated) = 0;
};

/** Callback for changes of Subject_Deprecated */
using VehicleDriverIdentifierDeprecatedSubjectDeprecatedPropertyCb = std::function<void(const std::string& Subject_Deprecated)>;
/** Callback for changes of Issuer_Deprecated */
using VehicleDriverIdentifierDeprecatedIssuerDeprecatedPropertyCb = std::function<void(const std::string& Issuer_Deprecated)>;


/**
 * The IVehicleDriverIdentifierDeprecatedPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleDriverIdentifierDeprecated implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleDriverIdentifierDeprecatedSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleDriverIdentifierDeprecated on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleDriverIdentifierDeprecatedPublisher
{
public:
    virtual ~IVehicleDriverIdentifierDeprecatedPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Driver_Identifier_Deprecated.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleDriverIdentifierDeprecatedSubscriber which is subscribed in this function to any change of the Vehicle_Driver_Identifier_Deprecated.
    */
    virtual void subscribeToAllChanges(IVehicleDriverIdentifierDeprecatedSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Driver_Identifier_Deprecated.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleDriverIdentifierDeprecatedSubscriber which subscription for any change of the Vehicle_Driver_Identifier_Deprecated is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleDriverIdentifierDeprecatedSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Subject_Deprecated value changes.
    * If your subscriber uses subscription with IVehicleDriverIdentifierDeprecatedSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDriverIdentifierDeprecatedSubjectDeprecatedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToSubjectDeprecatedChanged(VehicleDriverIdentifierDeprecatedSubjectDeprecatedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Subject_Deprecated property changes.
    * If your subscriber uses subscription with IVehicleDriverIdentifierDeprecatedSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromSubjectDeprecatedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Issuer_Deprecated value changes.
    * If your subscriber uses subscription with IVehicleDriverIdentifierDeprecatedSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDriverIdentifierDeprecatedIssuerDeprecatedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIssuerDeprecatedChanged(VehicleDriverIdentifierDeprecatedIssuerDeprecatedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Issuer_Deprecated property changes.
    * If your subscriber uses subscription with IVehicleDriverIdentifierDeprecatedSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIssuerDeprecatedChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Driver_Identifier_Deprecated implementation when property Subject_Deprecated changes.
    * @param The new value of Subject_Deprecated.
    * Subject for the authentication of the occupant e.g. UserID 7331677.
    */
    virtual void publishSubjectDeprecatedChanged(const std::string& Subject_Deprecated) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Driver_Identifier_Deprecated implementation when property Issuer_Deprecated changes.
    * @param The new value of Issuer_Deprecated.
    * Unique Issuer for the authentication of the occupant e.g. https://accounts.funcorp.com.
    */
    virtual void publishIssuerDeprecatedChanged(const std::string& Issuer_Deprecated) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
