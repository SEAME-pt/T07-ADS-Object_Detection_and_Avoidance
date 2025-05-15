#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleDiagnosticsSubscriber;
class IVehicleDiagnosticsPublisher;

/**
*
* IVehicleDiagnostics provides an interface for
 *  - methods defined for your Vehicle_Diagnostics 
 *  - property setters and getters for defined properties
 * The IVehicleDiagnostics also provides an interface to access a publisher IVehicleDiagnosticsPublisher, a class used by IVehicleDiagnosticsSubscriber clients.
 * The implementation should notify the publisher IVehicleDiagnosticsPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleDiagnosticsSubscriber, IVehicleDiagnosticsPublisher
 * and the example implementation Vehicle_Diagnostics  or the
 */
class CPP_VEHICLE_EXPORT IVehicleDiagnostics
{
public:
    virtual ~IVehicleDiagnostics() = default;

    /**
    * Sets the value of the DTCCount property.
    * @param DTCCount Number of Diagnostic Trouble Codes (DTC)
    */
    virtual void setDtcCount(int32_t DTCCount) = 0;
    /**
    * Gets the value of the DTCCount property.
    * @return Number of Diagnostic Trouble Codes (DTC)
    */
    virtual int32_t getDtcCount() const = 0;

    /**
    * Sets the value of the DTCList property.
    * @param DTCList List of currently active DTCs formatted according OBD II (SAE-J2012DA_201812) standard ([P|C|B|U]XXXXX )
    */
    virtual void setDtcList(const std::list<std::string>& DTCList) = 0;
    /**
    * Gets the value of the DTCList property.
    * @return List of currently active DTCs formatted according OBD II (SAE-J2012DA_201812) standard ([P|C|B|U]XXXXX )
    */
    virtual const std::list<std::string>& getDtcList() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Diagnostics changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Diagnostics.
    */
    virtual IVehicleDiagnosticsPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleDiagnosticsSubscriber contains functions to allow informing about signals or property changes of the IVehicleDiagnostics implementation.
 * The implementation for IVehicleDiagnostics should provide mechanism for subscription of the IVehicleDiagnosticsSubscriber clients.
 * See IVehicleDiagnosticsPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleDiagnostics should call the IVehicleDiagnosticsSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleDiagnosticsSubscriber class to implement clients of the IVehicleDiagnostics or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleDiagnosticsSubscriber
{
public:
    virtual ~IVehicleDiagnosticsSubscriber() = default;
    /**
    * Called by the IVehicleDiagnosticsPublisher when DTCCount value has changed if subscribed for the DTCCount change.
    * @param DTCCount Number of Diagnostic Trouble Codes (DTC)
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDtcCountChanged(int32_t DTCCount) = 0;
    /**
    * Called by the IVehicleDiagnosticsPublisher when DTCList value has changed if subscribed for the DTCList change.
    * @param DTCList List of currently active DTCs formatted according OBD II (SAE-J2012DA_201812) standard ([P|C|B|U]XXXXX )
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDtcListChanged(const std::list<std::string>& DTCList) = 0;
};

/** Callback for changes of DTCCount */
using VehicleDiagnosticsDtcCountPropertyCb = std::function<void(int32_t DTCCount)>;
/** Callback for changes of DTCList */
using VehicleDiagnosticsDtcListPropertyCb = std::function<void(const std::list<std::string>& DTCList)>;


/**
 * The IVehicleDiagnosticsPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleDiagnostics implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleDiagnosticsSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleDiagnostics on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleDiagnosticsPublisher
{
public:
    virtual ~IVehicleDiagnosticsPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Diagnostics.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleDiagnosticsSubscriber which is subscribed in this function to any change of the Vehicle_Diagnostics.
    */
    virtual void subscribeToAllChanges(IVehicleDiagnosticsSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Diagnostics.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleDiagnosticsSubscriber which subscription for any change of the Vehicle_Diagnostics is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleDiagnosticsSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for DTCCount value changes.
    * If your subscriber uses subscription with IVehicleDiagnosticsSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDiagnosticsDtcCountPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDtcCountChanged(VehicleDiagnosticsDtcCountPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DTCCount property changes.
    * If your subscriber uses subscription with IVehicleDiagnosticsSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDtcCountChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for DTCList value changes.
    * If your subscriber uses subscription with IVehicleDiagnosticsSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleDiagnosticsDtcListPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDtcListChanged(VehicleDiagnosticsDtcListPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DTCList property changes.
    * If your subscriber uses subscription with IVehicleDiagnosticsSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDtcListChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Diagnostics implementation when property DTCCount changes.
    * @param The new value of DTCCount.
    * Number of Diagnostic Trouble Codes (DTC)
    */
    virtual void publishDtcCountChanged(int32_t DTCCount) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Diagnostics implementation when property DTCList changes.
    * @param The new value of DTCList.
    * List of currently active DTCs formatted according OBD II (SAE-J2012DA_201812) standard ([P|C|B|U]XXXXX )
    */
    virtual void publishDtcListChanged(const std::list<std::string>& DTCList) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
