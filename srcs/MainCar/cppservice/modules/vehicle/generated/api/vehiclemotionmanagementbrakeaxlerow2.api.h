#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleMotionManagementBrakeAxleRow2Subscriber;
class IVehicleMotionManagementBrakeAxleRow2Publisher;

/**
*
* IVehicleMotionManagementBrakeAxleRow2 provides an interface for
 *  - methods defined for your Vehicle_MotionManagement_Brake_Axle_Row2 
 *  - property setters and getters for defined properties
 * The IVehicleMotionManagementBrakeAxleRow2 also provides an interface to access a publisher IVehicleMotionManagementBrakeAxleRow2Publisher, a class used by IVehicleMotionManagementBrakeAxleRow2Subscriber clients.
 * The implementation should notify the publisher IVehicleMotionManagementBrakeAxleRow2Publisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleMotionManagementBrakeAxleRow2Subscriber, IVehicleMotionManagementBrakeAxleRow2Publisher
 * and the example implementation Vehicle_MotionManagement_Brake_Axle_Row2  or the
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementBrakeAxleRow2
{
public:
    virtual ~IVehicleMotionManagementBrakeAxleRow2() = default;

    /**
    * Sets the value of the TorqueElectricMinimum property.
    * @param TorqueElectricMinimum Limit for regenerative brake torque at given axle. Brake Torque < 0Nm.
    */
    virtual void setTorqueElectricMinimum(int32_t TorqueElectricMinimum) = 0;
    /**
    * Gets the value of the TorqueElectricMinimum property.
    * @return Limit for regenerative brake torque at given axle. Brake Torque < 0Nm.
    */
    virtual int32_t getTorqueElectricMinimum() const = 0;

    /**
    * Sets the value of the TorqueFrictionDifferenceMaximum property.
    * @param TorqueFrictionDifferenceMaximum Maximum absolute wheel torque difference between left and right wheel for friction brake.
    */
    virtual void setTorqueFrictionDifferenceMaximum(int32_t TorqueFrictionDifferenceMaximum) = 0;
    /**
    * Gets the value of the TorqueFrictionDifferenceMaximum property.
    * @return Maximum absolute wheel torque difference between left and right wheel for friction brake.
    */
    virtual int32_t getTorqueFrictionDifferenceMaximum() const = 0;

    /**
    * Sets the value of the TorqueDistributionFrictionRightMaximum property.
    * @param TorqueDistributionFrictionRightMaximum Maximum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque shall be shifted to the left wheel. 50% = At most 50% friction torque may be shifted to the right wheel. 100% = Complete friction torque may be shifted to the right wheel.
    */
    virtual void setTorqueDistributionFrictionRightMaximum(int32_t TorqueDistributionFrictionRightMaximum) = 0;
    /**
    * Gets the value of the TorqueDistributionFrictionRightMaximum property.
    * @return Maximum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque shall be shifted to the left wheel. 50% = At most 50% friction torque may be shifted to the right wheel. 100% = Complete friction torque may be shifted to the right wheel.
    */
    virtual int32_t getTorqueDistributionFrictionRightMaximum() const = 0;

    /**
    * Sets the value of the TorqueDistributionFrictionRightMinimum property.
    * @param TorqueDistributionFrictionRightMinimum Minimum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque may be shifted to the left wheel. 50% = At least 50% friction torque shall be shifted to the right wheel. 100% = Complete friction torque shall be shifted to the right wheel.
    */
    virtual void setTorqueDistributionFrictionRightMinimum(int32_t TorqueDistributionFrictionRightMinimum) = 0;
    /**
    * Gets the value of the TorqueDistributionFrictionRightMinimum property.
    * @return Minimum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque may be shifted to the left wheel. 50% = At least 50% friction torque shall be shifted to the right wheel. 100% = Complete friction torque shall be shifted to the right wheel.
    */
    virtual int32_t getTorqueDistributionFrictionRightMinimum() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake_Axle_Row2 changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_MotionManagement_Brake_Axle_Row2.
    */
    virtual IVehicleMotionManagementBrakeAxleRow2Publisher& _getPublisher() const = 0;
};


/**
 * The IVehicleMotionManagementBrakeAxleRow2Subscriber contains functions to allow informing about signals or property changes of the IVehicleMotionManagementBrakeAxleRow2 implementation.
 * The implementation for IVehicleMotionManagementBrakeAxleRow2 should provide mechanism for subscription of the IVehicleMotionManagementBrakeAxleRow2Subscriber clients.
 * See IVehicleMotionManagementBrakeAxleRow2Publisher, which provides facilitation for this purpose.
 * The implementation for IVehicleMotionManagementBrakeAxleRow2 should call the IVehicleMotionManagementBrakeAxleRow2Subscriber interface functions on either signal emit or property change.
 * You can use IVehicleMotionManagementBrakeAxleRow2Subscriber class to implement clients of the IVehicleMotionManagementBrakeAxleRow2 or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementBrakeAxleRow2Subscriber
{
public:
    virtual ~IVehicleMotionManagementBrakeAxleRow2Subscriber() = default;
    /**
    * Called by the IVehicleMotionManagementBrakeAxleRow2Publisher when TorqueElectricMinimum value has changed if subscribed for the TorqueElectricMinimum change.
    * @param TorqueElectricMinimum Limit for regenerative brake torque at given axle. Brake Torque < 0Nm.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueElectricMinimumChanged(int32_t TorqueElectricMinimum) = 0;
    /**
    * Called by the IVehicleMotionManagementBrakeAxleRow2Publisher when TorqueFrictionDifferenceMaximum value has changed if subscribed for the TorqueFrictionDifferenceMaximum change.
    * @param TorqueFrictionDifferenceMaximum Maximum absolute wheel torque difference between left and right wheel for friction brake.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueFrictionDifferenceMaximumChanged(int32_t TorqueFrictionDifferenceMaximum) = 0;
    /**
    * Called by the IVehicleMotionManagementBrakeAxleRow2Publisher when TorqueDistributionFrictionRightMaximum value has changed if subscribed for the TorqueDistributionFrictionRightMaximum change.
    * @param TorqueDistributionFrictionRightMaximum Maximum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque shall be shifted to the left wheel. 50% = At most 50% friction torque may be shifted to the right wheel. 100% = Complete friction torque may be shifted to the right wheel.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueDistributionFrictionRightMaximumChanged(int32_t TorqueDistributionFrictionRightMaximum) = 0;
    /**
    * Called by the IVehicleMotionManagementBrakeAxleRow2Publisher when TorqueDistributionFrictionRightMinimum value has changed if subscribed for the TorqueDistributionFrictionRightMinimum change.
    * @param TorqueDistributionFrictionRightMinimum Minimum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque may be shifted to the left wheel. 50% = At least 50% friction torque shall be shifted to the right wheel. 100% = Complete friction torque shall be shifted to the right wheel.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueDistributionFrictionRightMinimumChanged(int32_t TorqueDistributionFrictionRightMinimum) = 0;
};

/** Callback for changes of TorqueElectricMinimum */
using VehicleMotionManagementBrakeAxleRow2TorqueElectricMinimumPropertyCb = std::function<void(int32_t TorqueElectricMinimum)>;
/** Callback for changes of TorqueFrictionDifferenceMaximum */
using VehicleMotionManagementBrakeAxleRow2TorqueFrictionDifferenceMaximumPropertyCb = std::function<void(int32_t TorqueFrictionDifferenceMaximum)>;
/** Callback for changes of TorqueDistributionFrictionRightMaximum */
using VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMaximumPropertyCb = std::function<void(int32_t TorqueDistributionFrictionRightMaximum)>;
/** Callback for changes of TorqueDistributionFrictionRightMinimum */
using VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMinimumPropertyCb = std::function<void(int32_t TorqueDistributionFrictionRightMinimum)>;


/**
 * The IVehicleMotionManagementBrakeAxleRow2Publisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleMotionManagementBrakeAxleRow2 implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleMotionManagementBrakeAxleRow2Subscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleMotionManagementBrakeAxleRow2 on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleMotionManagementBrakeAxleRow2Publisher
{
public:
    virtual ~IVehicleMotionManagementBrakeAxleRow2Publisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_MotionManagement_Brake_Axle_Row2.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleMotionManagementBrakeAxleRow2Subscriber which is subscribed in this function to any change of the Vehicle_MotionManagement_Brake_Axle_Row2.
    */
    virtual void subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow2Subscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_MotionManagement_Brake_Axle_Row2.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleMotionManagementBrakeAxleRow2Subscriber which subscription for any change of the Vehicle_MotionManagement_Brake_Axle_Row2 is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow2Subscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for TorqueElectricMinimum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeAxleRow2Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeAxleRow2TorqueElectricMinimumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueElectricMinimumChanged(VehicleMotionManagementBrakeAxleRow2TorqueElectricMinimumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueElectricMinimum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeAxleRow2Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueElectricMinimumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueFrictionDifferenceMaximum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeAxleRow2Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeAxleRow2TorqueFrictionDifferenceMaximumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueFrictionDifferenceMaximumChanged(VehicleMotionManagementBrakeAxleRow2TorqueFrictionDifferenceMaximumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueFrictionDifferenceMaximum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeAxleRow2Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueFrictionDifferenceMaximumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueDistributionFrictionRightMaximum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeAxleRow2Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMaximumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueDistributionFrictionRightMaximumChanged(VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMaximumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueDistributionFrictionRightMaximum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeAxleRow2Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueDistributionFrictionRightMaximumChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueDistributionFrictionRightMinimum value changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeAxleRow2Subscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMinimumPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueDistributionFrictionRightMinimumChanged(VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMinimumPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueDistributionFrictionRightMinimum property changes.
    * If your subscriber uses subscription with IVehicleMotionManagementBrakeAxleRow2Subscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueDistributionFrictionRightMinimumChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake_Axle_Row2 implementation when property TorqueElectricMinimum changes.
    * @param The new value of TorqueElectricMinimum.
    * Limit for regenerative brake torque at given axle. Brake Torque < 0Nm.
    */
    virtual void publishTorqueElectricMinimumChanged(int32_t TorqueElectricMinimum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake_Axle_Row2 implementation when property TorqueFrictionDifferenceMaximum changes.
    * @param The new value of TorqueFrictionDifferenceMaximum.
    * Maximum absolute wheel torque difference between left and right wheel for friction brake.
    */
    virtual void publishTorqueFrictionDifferenceMaximumChanged(int32_t TorqueFrictionDifferenceMaximum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake_Axle_Row2 implementation when property TorqueDistributionFrictionRightMaximum changes.
    * @param The new value of TorqueDistributionFrictionRightMaximum.
    * Maximum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque shall be shifted to the left wheel. 50% = At most 50% friction torque may be shifted to the right wheel. 100% = Complete friction torque may be shifted to the right wheel.
    */
    virtual void publishTorqueDistributionFrictionRightMaximumChanged(int32_t TorqueDistributionFrictionRightMaximum) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_MotionManagement_Brake_Axle_Row2 implementation when property TorqueDistributionFrictionRightMinimum changes.
    * @param The new value of TorqueDistributionFrictionRightMinimum.
    * Minimum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque may be shifted to the left wheel. 50% = At least 50% friction torque shall be shifted to the right wheel. 100% = Complete friction torque shall be shifted to the right wheel.
    */
    virtual void publishTorqueDistributionFrictionRightMinimumChanged(int32_t TorqueDistributionFrictionRightMinimum) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
