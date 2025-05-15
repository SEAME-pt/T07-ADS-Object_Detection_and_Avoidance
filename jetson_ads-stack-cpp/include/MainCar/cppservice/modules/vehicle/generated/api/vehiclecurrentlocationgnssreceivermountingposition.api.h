#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber;
class IVehicleCurrentLocationGnssReceiverMountingPositionPublisher;

/**
*
* IVehicleCurrentLocationGnssReceiverMountingPosition provides an interface for
 *  - methods defined for your Vehicle_CurrentLocation_GNSSReceiver_MountingPosition 
 *  - property setters and getters for defined properties
 * The IVehicleCurrentLocationGnssReceiverMountingPosition also provides an interface to access a publisher IVehicleCurrentLocationGnssReceiverMountingPositionPublisher, a class used by IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber clients.
 * The implementation should notify the publisher IVehicleCurrentLocationGnssReceiverMountingPositionPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber, IVehicleCurrentLocationGnssReceiverMountingPositionPublisher
 * and the example implementation Vehicle_CurrentLocation_GNSSReceiver_MountingPosition  or the
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocationGnssReceiverMountingPosition
{
public:
    virtual ~IVehicleCurrentLocationGnssReceiverMountingPosition() = default;

    /**
    * Sets the value of the X property.
    * @param X Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = forward of rear axle. Negative values = backward of rear axle.
    */
    virtual void setX(int32_t X) = 0;
    /**
    * Gets the value of the X property.
    * @return Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = forward of rear axle. Negative values = backward of rear axle.
    */
    virtual int32_t getX() const = 0;

    /**
    * Sets the value of the Y property.
    * @param Y Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = left of origin. Negative values = right of origin. Left/Right is as seen from driver perspective, i.e. by a person looking forward.
    */
    virtual void setY(int32_t Y) = 0;
    /**
    * Gets the value of the Y property.
    * @return Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = left of origin. Negative values = right of origin. Left/Right is as seen from driver perspective, i.e. by a person looking forward.
    */
    virtual int32_t getY() const = 0;

    /**
    * Sets the value of the Z property.
    * @param Z Mounting position of GNSS receiver on Z-axis. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = above center of rear axle. Negative values = below center of rear axle.
    */
    virtual void setZ(int32_t Z) = 0;
    /**
    * Gets the value of the Z property.
    * @return Mounting position of GNSS receiver on Z-axis. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = above center of rear axle. Negative values = below center of rear axle.
    */
    virtual int32_t getZ() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation_GNSSReceiver_MountingPosition changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_CurrentLocation_GNSSReceiver_MountingPosition.
    */
    virtual IVehicleCurrentLocationGnssReceiverMountingPositionPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber contains functions to allow informing about signals or property changes of the IVehicleCurrentLocationGnssReceiverMountingPosition implementation.
 * The implementation for IVehicleCurrentLocationGnssReceiverMountingPosition should provide mechanism for subscription of the IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber clients.
 * See IVehicleCurrentLocationGnssReceiverMountingPositionPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleCurrentLocationGnssReceiverMountingPosition should call the IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber class to implement clients of the IVehicleCurrentLocationGnssReceiverMountingPosition or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber
{
public:
    virtual ~IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber() = default;
    /**
    * Called by the IVehicleCurrentLocationGnssReceiverMountingPositionPublisher when X value has changed if subscribed for the X change.
    * @param X Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = forward of rear axle. Negative values = backward of rear axle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onXChanged(int32_t X) = 0;
    /**
    * Called by the IVehicleCurrentLocationGnssReceiverMountingPositionPublisher when Y value has changed if subscribed for the Y change.
    * @param Y Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = left of origin. Negative values = right of origin. Left/Right is as seen from driver perspective, i.e. by a person looking forward.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onYChanged(int32_t Y) = 0;
    /**
    * Called by the IVehicleCurrentLocationGnssReceiverMountingPositionPublisher when Z value has changed if subscribed for the Z change.
    * @param Z Mounting position of GNSS receiver on Z-axis. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = above center of rear axle. Negative values = below center of rear axle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onZChanged(int32_t Z) = 0;
};

/** Callback for changes of X */
using VehicleCurrentLocationGnssReceiverMountingPositionXPropertyCb = std::function<void(int32_t X)>;
/** Callback for changes of Y */
using VehicleCurrentLocationGnssReceiverMountingPositionYPropertyCb = std::function<void(int32_t Y)>;
/** Callback for changes of Z */
using VehicleCurrentLocationGnssReceiverMountingPositionZPropertyCb = std::function<void(int32_t Z)>;


/**
 * The IVehicleCurrentLocationGnssReceiverMountingPositionPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleCurrentLocationGnssReceiverMountingPosition implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleCurrentLocationGnssReceiverMountingPosition on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleCurrentLocationGnssReceiverMountingPositionPublisher
{
public:
    virtual ~IVehicleCurrentLocationGnssReceiverMountingPositionPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber which is subscribed in this function to any change of the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition.
    */
    virtual void subscribeToAllChanges(IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber which subscription for any change of the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for X value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationGnssReceiverMountingPositionXPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToXChanged(VehicleCurrentLocationGnssReceiverMountingPositionXPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from X property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromXChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Y value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationGnssReceiverMountingPositionYPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToYChanged(VehicleCurrentLocationGnssReceiverMountingPositionYPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Y property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromYChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Z value changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleCurrentLocationGnssReceiverMountingPositionZPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToZChanged(VehicleCurrentLocationGnssReceiverMountingPositionZPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Z property changes.
    * If your subscriber uses subscription with IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromZChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation when property X changes.
    * @param The new value of X.
    * Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = forward of rear axle. Negative values = backward of rear axle.
    */
    virtual void publishXChanged(int32_t X) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation when property Y changes.
    * @param The new value of Y.
    * Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = left of origin. Negative values = right of origin. Left/Right is as seen from driver perspective, i.e. by a person looking forward.
    */
    virtual void publishYChanged(int32_t Y) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation when property Z changes.
    * @param The new value of Z.
    * Mounting position of GNSS receiver on Z-axis. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = above center of rear axle. Negative values = below center of rear axle.
    */
    virtual void publishZChanged(int32_t Z) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
