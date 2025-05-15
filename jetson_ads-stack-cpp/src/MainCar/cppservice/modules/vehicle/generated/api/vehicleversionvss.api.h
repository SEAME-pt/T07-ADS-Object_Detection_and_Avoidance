#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleVersionVssSubscriber;
class IVehicleVersionVssPublisher;

/**
*
* IVehicleVersionVss provides an interface for
 *  - methods defined for your Vehicle_VersionVSS 
 *  - property setters and getters for defined properties
 * The IVehicleVersionVss also provides an interface to access a publisher IVehicleVersionVssPublisher, a class used by IVehicleVersionVssSubscriber clients.
 * The implementation should notify the publisher IVehicleVersionVssPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleVersionVssSubscriber, IVehicleVersionVssPublisher
 * and the example implementation Vehicle_VersionVSS  or the
 */
class CPP_VEHICLE_EXPORT IVehicleVersionVss
{
public:
    virtual ~IVehicleVersionVss() = default;

    /**
    * Sets the value of the Major property.
    * @param Major Supported Version of VSS - Major version.
    */
    virtual void setMajor(int32_t Major) = 0;
    /**
    * Gets the value of the Major property.
    * @return Supported Version of VSS - Major version.
    */
    virtual int32_t getMajor() const = 0;

    /**
    * Sets the value of the Minor property.
    * @param Minor Supported Version of VSS - Minor version.
    */
    virtual void setMinor(int32_t Minor) = 0;
    /**
    * Gets the value of the Minor property.
    * @return Supported Version of VSS - Minor version.
    */
    virtual int32_t getMinor() const = 0;

    /**
    * Sets the value of the Patch property.
    * @param Patch Supported Version of VSS - Patch version.
    */
    virtual void setPatch(int32_t Patch) = 0;
    /**
    * Gets the value of the Patch property.
    * @return Supported Version of VSS - Patch version.
    */
    virtual int32_t getPatch() const = 0;

    /**
    * Sets the value of the Label property.
    * @param Label Label to further describe the version.
    */
    virtual void setLabel(const std::string& Label) = 0;
    /**
    * Gets the value of the Label property.
    * @return Label to further describe the version.
    */
    virtual const std::string& getLabel() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_VersionVSS changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_VersionVSS.
    */
    virtual IVehicleVersionVssPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleVersionVssSubscriber contains functions to allow informing about signals or property changes of the IVehicleVersionVss implementation.
 * The implementation for IVehicleVersionVss should provide mechanism for subscription of the IVehicleVersionVssSubscriber clients.
 * See IVehicleVersionVssPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleVersionVss should call the IVehicleVersionVssSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleVersionVssSubscriber class to implement clients of the IVehicleVersionVss or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleVersionVssSubscriber
{
public:
    virtual ~IVehicleVersionVssSubscriber() = default;
    /**
    * Called by the IVehicleVersionVssPublisher when Major value has changed if subscribed for the Major change.
    * @param Major Supported Version of VSS - Major version.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMajorChanged(int32_t Major) = 0;
    /**
    * Called by the IVehicleVersionVssPublisher when Minor value has changed if subscribed for the Minor change.
    * @param Minor Supported Version of VSS - Minor version.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMinorChanged(int32_t Minor) = 0;
    /**
    * Called by the IVehicleVersionVssPublisher when Patch value has changed if subscribed for the Patch change.
    * @param Patch Supported Version of VSS - Patch version.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPatchChanged(int32_t Patch) = 0;
    /**
    * Called by the IVehicleVersionVssPublisher when Label value has changed if subscribed for the Label change.
    * @param Label Label to further describe the version.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLabelChanged(const std::string& Label) = 0;
};

/** Callback for changes of Major */
using VehicleVersionVssMajorPropertyCb = std::function<void(int32_t Major)>;
/** Callback for changes of Minor */
using VehicleVersionVssMinorPropertyCb = std::function<void(int32_t Minor)>;
/** Callback for changes of Patch */
using VehicleVersionVssPatchPropertyCb = std::function<void(int32_t Patch)>;
/** Callback for changes of Label */
using VehicleVersionVssLabelPropertyCb = std::function<void(const std::string& Label)>;


/**
 * The IVehicleVersionVssPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleVersionVss implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleVersionVssSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleVersionVss on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleVersionVssPublisher
{
public:
    virtual ~IVehicleVersionVssPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_VersionVSS.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleVersionVssSubscriber which is subscribed in this function to any change of the Vehicle_VersionVSS.
    */
    virtual void subscribeToAllChanges(IVehicleVersionVssSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_VersionVSS.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleVersionVssSubscriber which subscription for any change of the Vehicle_VersionVSS is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleVersionVssSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Major value changes.
    * If your subscriber uses subscription with IVehicleVersionVssSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVersionVssMajorPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMajorChanged(VehicleVersionVssMajorPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Major property changes.
    * If your subscriber uses subscription with IVehicleVersionVssSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMajorChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Minor value changes.
    * If your subscriber uses subscription with IVehicleVersionVssSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVersionVssMinorPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMinorChanged(VehicleVersionVssMinorPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Minor property changes.
    * If your subscriber uses subscription with IVehicleVersionVssSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMinorChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Patch value changes.
    * If your subscriber uses subscription with IVehicleVersionVssSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVersionVssPatchPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPatchChanged(VehicleVersionVssPatchPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Patch property changes.
    * If your subscriber uses subscription with IVehicleVersionVssSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPatchChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Label value changes.
    * If your subscriber uses subscription with IVehicleVersionVssSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVersionVssLabelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLabelChanged(VehicleVersionVssLabelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Label property changes.
    * If your subscriber uses subscription with IVehicleVersionVssSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLabelChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VersionVSS implementation when property Major changes.
    * @param The new value of Major.
    * Supported Version of VSS - Major version.
    */
    virtual void publishMajorChanged(int32_t Major) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VersionVSS implementation when property Minor changes.
    * @param The new value of Minor.
    * Supported Version of VSS - Minor version.
    */
    virtual void publishMinorChanged(int32_t Minor) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VersionVSS implementation when property Patch changes.
    * @param The new value of Patch.
    * Supported Version of VSS - Patch version.
    */
    virtual void publishPatchChanged(int32_t Patch) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VersionVSS implementation when property Label changes.
    * @param The new value of Label.
    * Label to further describe the version.
    */
    virtual void publishLabelChanged(const std::string& Label) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
