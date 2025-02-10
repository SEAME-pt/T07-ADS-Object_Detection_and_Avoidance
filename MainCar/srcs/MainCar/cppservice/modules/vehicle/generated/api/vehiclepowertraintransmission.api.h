#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainTransmissionSubscriber;
class IVehiclePowertrainTransmissionPublisher;

/**
*
* IVehiclePowertrainTransmission provides an interface for
 *  - methods defined for your Vehicle_Powertrain_Transmission 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainTransmission also provides an interface to access a publisher IVehiclePowertrainTransmissionPublisher, a class used by IVehiclePowertrainTransmissionSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainTransmissionPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainTransmissionSubscriber, IVehiclePowertrainTransmissionPublisher
 * and the example implementation Vehicle_Powertrain_Transmission  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTransmission
{
public:
    virtual ~IVehiclePowertrainTransmission() = default;

    /**
    * Sets the value of the Type property.
    * @param Type Transmission type.
    */
    virtual void setType(Vehicle_Powertrain_Transmission_Type_ValueEnum Type) = 0;
    /**
    * Gets the value of the Type property.
    * @return Transmission type.
    */
    virtual Vehicle_Powertrain_Transmission_Type_ValueEnum getType() const = 0;

    /**
    * Sets the value of the GearCount property.
    * @param GearCount Number of forward gears in the transmission. -1 = CVT.
    */
    virtual void setGearCount(int32_t GearCount) = 0;
    /**
    * Gets the value of the GearCount property.
    * @return Number of forward gears in the transmission. -1 = CVT.
    */
    virtual int32_t getGearCount() const = 0;

    /**
    * Sets the value of the DriveType property.
    * @param DriveType Drive type.
    */
    virtual void setDriveType(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType) = 0;
    /**
    * Gets the value of the DriveType property.
    * @return Drive type.
    */
    virtual Vehicle_Powertrain_Transmission_DriveType_ValueEnum getDriveType() const = 0;

    /**
    * Sets the value of the TravelledDistance property.
    * @param TravelledDistance Odometer reading, total distance travelled during the lifetime of the transmission.
    */
    virtual void setTravelledDistance(float TravelledDistance) = 0;
    /**
    * Gets the value of the TravelledDistance property.
    * @return Odometer reading, total distance travelled during the lifetime of the transmission.
    */
    virtual float getTravelledDistance() const = 0;

    /**
    * Sets the value of the CurrentGear property.
    * @param CurrentGear The current gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse.
    */
    virtual void setCurrentGear(int32_t CurrentGear) = 0;
    /**
    * Gets the value of the CurrentGear property.
    * @return The current gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse.
    */
    virtual int32_t getCurrentGear() const = 0;

    /**
    * Sets the value of the SelectedGear property.
    * @param SelectedGear The selected gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse, 126=Park, 127=Drive.
    */
    virtual void setSelectedGear(int32_t SelectedGear) = 0;
    /**
    * Gets the value of the SelectedGear property.
    * @return The selected gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse, 126=Park, 127=Drive.
    */
    virtual int32_t getSelectedGear() const = 0;

    /**
    * Sets the value of the IsParkLockEngaged property.
    * @param IsParkLockEngaged Is the transmission park lock engaged or not. False = Disengaged. True = Engaged.
    */
    virtual void setIsParkLockEngaged(bool IsParkLockEngaged) = 0;
    /**
    * Gets the value of the IsParkLockEngaged property.
    * @return Is the transmission park lock engaged or not. False = Disengaged. True = Engaged.
    */
    virtual bool getIsParkLockEngaged() const = 0;

    /**
    * Sets the value of the IsLowRangeEngaged property.
    * @param IsLowRangeEngaged Is gearbox in low range mode or not. False = Normal/High range engaged. True = Low range engaged.
    */
    virtual void setIsLowRangeEngaged(bool IsLowRangeEngaged) = 0;
    /**
    * Gets the value of the IsLowRangeEngaged property.
    * @return Is gearbox in low range mode or not. False = Normal/High range engaged. True = Low range engaged.
    */
    virtual bool getIsLowRangeEngaged() const = 0;

    /**
    * Sets the value of the IsElectricalPowertrainEngaged property.
    * @param IsElectricalPowertrainEngaged Is electrical powertrain mechanically connected/engaged to the drivetrain or not. False = Disconnected/Disengaged. True = Connected/Engaged.
    */
    virtual void setIsElectricalPowertrainEngaged(bool IsElectricalPowertrainEngaged) = 0;
    /**
    * Gets the value of the IsElectricalPowertrainEngaged property.
    * @return Is electrical powertrain mechanically connected/engaged to the drivetrain or not. False = Disconnected/Disengaged. True = Connected/Engaged.
    */
    virtual bool getIsElectricalPowertrainEngaged() const = 0;

    /**
    * Sets the value of the PerformanceMode property.
    * @param PerformanceMode Current gearbox performance mode.
    */
    virtual void setPerformanceMode(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode) = 0;
    /**
    * Gets the value of the PerformanceMode property.
    * @return Current gearbox performance mode.
    */
    virtual Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum getPerformanceMode() const = 0;

    /**
    * Sets the value of the GearChangeMode property.
    * @param GearChangeMode Is the gearbox in automatic or manual (paddle) mode.
    */
    virtual void setGearChangeMode(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode) = 0;
    /**
    * Gets the value of the GearChangeMode property.
    * @return Is the gearbox in automatic or manual (paddle) mode.
    */
    virtual Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum getGearChangeMode() const = 0;

    /**
    * Sets the value of the Temperature property.
    * @param Temperature The current gearbox temperature.
    */
    virtual void setTemperature(float Temperature) = 0;
    /**
    * Gets the value of the Temperature property.
    * @return The current gearbox temperature.
    */
    virtual float getTemperature() const = 0;

    /**
    * Sets the value of the ClutchEngagement property.
    * @param ClutchEngagement Clutch engagement. 0% = Clutch fully disengaged. 100% = Clutch fully engaged.
    */
    virtual void setClutchEngagement(float ClutchEngagement) = 0;
    /**
    * Gets the value of the ClutchEngagement property.
    * @return Clutch engagement. 0% = Clutch fully disengaged. 100% = Clutch fully engaged.
    */
    virtual float getClutchEngagement() const = 0;

    /**
    * Sets the value of the ClutchWear property.
    * @param ClutchWear Clutch wear as a percent. 0 = no wear. 100 = worn.
    */
    virtual void setClutchWear(int32_t ClutchWear) = 0;
    /**
    * Gets the value of the ClutchWear property.
    * @return Clutch wear as a percent. 0 = no wear. 100 = worn.
    */
    virtual int32_t getClutchWear() const = 0;

    /**
    * Sets the value of the DiffLockFrontEngagement property.
    * @param DiffLockFrontEngagement Front Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    */
    virtual void setDiffLockFrontEngagement(float DiffLockFrontEngagement) = 0;
    /**
    * Gets the value of the DiffLockFrontEngagement property.
    * @return Front Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    */
    virtual float getDiffLockFrontEngagement() const = 0;

    /**
    * Sets the value of the DiffLockRearEngagement property.
    * @param DiffLockRearEngagement Rear Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    */
    virtual void setDiffLockRearEngagement(float DiffLockRearEngagement) = 0;
    /**
    * Gets the value of the DiffLockRearEngagement property.
    * @return Rear Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    */
    virtual float getDiffLockRearEngagement() const = 0;

    /**
    * Sets the value of the TorqueDistribution property.
    * @param TorqueDistribution Torque distribution between front and rear axle in percent. -100% = Full torque to front axle, 0% = 50:50 Front/Rear, 100% = Full torque to rear axle.
    */
    virtual void setTorqueDistribution(float TorqueDistribution) = 0;
    /**
    * Gets the value of the TorqueDistribution property.
    * @return Torque distribution between front and rear axle in percent. -100% = Full torque to front axle, 0% = 50:50 Front/Rear, 100% = Full torque to rear axle.
    */
    virtual float getTorqueDistribution() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_Transmission changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_Transmission.
    */
    virtual IVehiclePowertrainTransmissionPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainTransmissionSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainTransmission implementation.
 * The implementation for IVehiclePowertrainTransmission should provide mechanism for subscription of the IVehiclePowertrainTransmissionSubscriber clients.
 * See IVehiclePowertrainTransmissionPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainTransmission should call the IVehiclePowertrainTransmissionSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainTransmissionSubscriber class to implement clients of the IVehiclePowertrainTransmission or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTransmissionSubscriber
{
public:
    virtual ~IVehiclePowertrainTransmissionSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when Type value has changed if subscribed for the Type change.
    * @param Type Transmission type.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTypeChanged(Vehicle_Powertrain_Transmission_Type_ValueEnum Type) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when GearCount value has changed if subscribed for the GearCount change.
    * @param GearCount Number of forward gears in the transmission. -1 = CVT.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onGearCountChanged(int32_t GearCount) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when DriveType value has changed if subscribed for the DriveType change.
    * @param DriveType Drive type.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDriveTypeChanged(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when TravelledDistance value has changed if subscribed for the TravelledDistance change.
    * @param TravelledDistance Odometer reading, total distance travelled during the lifetime of the transmission.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTravelledDistanceChanged(float TravelledDistance) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when CurrentGear value has changed if subscribed for the CurrentGear change.
    * @param CurrentGear The current gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCurrentGearChanged(int32_t CurrentGear) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when SelectedGear value has changed if subscribed for the SelectedGear change.
    * @param SelectedGear The selected gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse, 126=Park, 127=Drive.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onSelectedGearChanged(int32_t SelectedGear) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when IsParkLockEngaged value has changed if subscribed for the IsParkLockEngaged change.
    * @param IsParkLockEngaged Is the transmission park lock engaged or not. False = Disengaged. True = Engaged.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsParkLockEngagedChanged(bool IsParkLockEngaged) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when IsLowRangeEngaged value has changed if subscribed for the IsLowRangeEngaged change.
    * @param IsLowRangeEngaged Is gearbox in low range mode or not. False = Normal/High range engaged. True = Low range engaged.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsLowRangeEngagedChanged(bool IsLowRangeEngaged) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when IsElectricalPowertrainEngaged value has changed if subscribed for the IsElectricalPowertrainEngaged change.
    * @param IsElectricalPowertrainEngaged Is electrical powertrain mechanically connected/engaged to the drivetrain or not. False = Disconnected/Disengaged. True = Connected/Engaged.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onIsElectricalPowertrainEngagedChanged(bool IsElectricalPowertrainEngaged) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when PerformanceMode value has changed if subscribed for the PerformanceMode change.
    * @param PerformanceMode Current gearbox performance mode.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPerformanceModeChanged(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when GearChangeMode value has changed if subscribed for the GearChangeMode change.
    * @param GearChangeMode Is the gearbox in automatic or manual (paddle) mode.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onGearChangeModeChanged(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when Temperature value has changed if subscribed for the Temperature change.
    * @param Temperature The current gearbox temperature.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTemperatureChanged(float Temperature) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when ClutchEngagement value has changed if subscribed for the ClutchEngagement change.
    * @param ClutchEngagement Clutch engagement. 0% = Clutch fully disengaged. 100% = Clutch fully engaged.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onClutchEngagementChanged(float ClutchEngagement) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when ClutchWear value has changed if subscribed for the ClutchWear change.
    * @param ClutchWear Clutch wear as a percent. 0 = no wear. 100 = worn.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onClutchWearChanged(int32_t ClutchWear) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when DiffLockFrontEngagement value has changed if subscribed for the DiffLockFrontEngagement change.
    * @param DiffLockFrontEngagement Front Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDiffLockFrontEngagementChanged(float DiffLockFrontEngagement) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when DiffLockRearEngagement value has changed if subscribed for the DiffLockRearEngagement change.
    * @param DiffLockRearEngagement Rear Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDiffLockRearEngagementChanged(float DiffLockRearEngagement) = 0;
    /**
    * Called by the IVehiclePowertrainTransmissionPublisher when TorqueDistribution value has changed if subscribed for the TorqueDistribution change.
    * @param TorqueDistribution Torque distribution between front and rear axle in percent. -100% = Full torque to front axle, 0% = 50:50 Front/Rear, 100% = Full torque to rear axle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueDistributionChanged(float TorqueDistribution) = 0;
};

/** Callback for changes of Type */
using VehiclePowertrainTransmissionTypePropertyCb = std::function<void(Vehicle_Powertrain_Transmission_Type_ValueEnum Type)>;
/** Callback for changes of GearCount */
using VehiclePowertrainTransmissionGearCountPropertyCb = std::function<void(int32_t GearCount)>;
/** Callback for changes of DriveType */
using VehiclePowertrainTransmissionDriveTypePropertyCb = std::function<void(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType)>;
/** Callback for changes of TravelledDistance */
using VehiclePowertrainTransmissionTravelledDistancePropertyCb = std::function<void(float TravelledDistance)>;
/** Callback for changes of CurrentGear */
using VehiclePowertrainTransmissionCurrentGearPropertyCb = std::function<void(int32_t CurrentGear)>;
/** Callback for changes of SelectedGear */
using VehiclePowertrainTransmissionSelectedGearPropertyCb = std::function<void(int32_t SelectedGear)>;
/** Callback for changes of IsParkLockEngaged */
using VehiclePowertrainTransmissionIsParkLockEngagedPropertyCb = std::function<void(bool IsParkLockEngaged)>;
/** Callback for changes of IsLowRangeEngaged */
using VehiclePowertrainTransmissionIsLowRangeEngagedPropertyCb = std::function<void(bool IsLowRangeEngaged)>;
/** Callback for changes of IsElectricalPowertrainEngaged */
using VehiclePowertrainTransmissionIsElectricalPowertrainEngagedPropertyCb = std::function<void(bool IsElectricalPowertrainEngaged)>;
/** Callback for changes of PerformanceMode */
using VehiclePowertrainTransmissionPerformanceModePropertyCb = std::function<void(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode)>;
/** Callback for changes of GearChangeMode */
using VehiclePowertrainTransmissionGearChangeModePropertyCb = std::function<void(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode)>;
/** Callback for changes of Temperature */
using VehiclePowertrainTransmissionTemperaturePropertyCb = std::function<void(float Temperature)>;
/** Callback for changes of ClutchEngagement */
using VehiclePowertrainTransmissionClutchEngagementPropertyCb = std::function<void(float ClutchEngagement)>;
/** Callback for changes of ClutchWear */
using VehiclePowertrainTransmissionClutchWearPropertyCb = std::function<void(int32_t ClutchWear)>;
/** Callback for changes of DiffLockFrontEngagement */
using VehiclePowertrainTransmissionDiffLockFrontEngagementPropertyCb = std::function<void(float DiffLockFrontEngagement)>;
/** Callback for changes of DiffLockRearEngagement */
using VehiclePowertrainTransmissionDiffLockRearEngagementPropertyCb = std::function<void(float DiffLockRearEngagement)>;
/** Callback for changes of TorqueDistribution */
using VehiclePowertrainTransmissionTorqueDistributionPropertyCb = std::function<void(float TorqueDistribution)>;


/**
 * The IVehiclePowertrainTransmissionPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainTransmission implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainTransmissionSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainTransmission on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainTransmissionPublisher
{
public:
    virtual ~IVehiclePowertrainTransmissionPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_Transmission.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainTransmissionSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_Transmission.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainTransmissionSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_Transmission.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainTransmissionSubscriber which subscription for any change of the Vehicle_Powertrain_Transmission is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainTransmissionSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Type value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionTypePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTypeChanged(VehiclePowertrainTransmissionTypePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Type property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTypeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for GearCount value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionGearCountPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToGearCountChanged(VehiclePowertrainTransmissionGearCountPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from GearCount property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromGearCountChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for DriveType value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionDriveTypePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDriveTypeChanged(VehiclePowertrainTransmissionDriveTypePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DriveType property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDriveTypeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TravelledDistance value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionTravelledDistancePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTravelledDistanceChanged(VehiclePowertrainTransmissionTravelledDistancePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TravelledDistance property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTravelledDistanceChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CurrentGear value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionCurrentGearPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCurrentGearChanged(VehiclePowertrainTransmissionCurrentGearPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CurrentGear property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCurrentGearChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for SelectedGear value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionSelectedGearPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToSelectedGearChanged(VehiclePowertrainTransmissionSelectedGearPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from SelectedGear property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromSelectedGearChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsParkLockEngaged value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionIsParkLockEngagedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsParkLockEngagedChanged(VehiclePowertrainTransmissionIsParkLockEngagedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsParkLockEngaged property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsParkLockEngagedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsLowRangeEngaged value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionIsLowRangeEngagedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsLowRangeEngagedChanged(VehiclePowertrainTransmissionIsLowRangeEngagedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsLowRangeEngaged property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsLowRangeEngagedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for IsElectricalPowertrainEngaged value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionIsElectricalPowertrainEngagedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToIsElectricalPowertrainEngagedChanged(VehiclePowertrainTransmissionIsElectricalPowertrainEngagedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from IsElectricalPowertrainEngaged property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromIsElectricalPowertrainEngagedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for PerformanceMode value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionPerformanceModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPerformanceModeChanged(VehiclePowertrainTransmissionPerformanceModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from PerformanceMode property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPerformanceModeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for GearChangeMode value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionGearChangeModePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToGearChangeModeChanged(VehiclePowertrainTransmissionGearChangeModePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from GearChangeMode property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromGearChangeModeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Temperature value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionTemperaturePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTemperatureChanged(VehiclePowertrainTransmissionTemperaturePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Temperature property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTemperatureChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for ClutchEngagement value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionClutchEngagementPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToClutchEngagementChanged(VehiclePowertrainTransmissionClutchEngagementPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from ClutchEngagement property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromClutchEngagementChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for ClutchWear value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionClutchWearPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToClutchWearChanged(VehiclePowertrainTransmissionClutchWearPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from ClutchWear property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromClutchWearChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for DiffLockFrontEngagement value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionDiffLockFrontEngagementPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDiffLockFrontEngagementChanged(VehiclePowertrainTransmissionDiffLockFrontEngagementPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DiffLockFrontEngagement property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDiffLockFrontEngagementChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for DiffLockRearEngagement value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionDiffLockRearEngagementPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDiffLockRearEngagementChanged(VehiclePowertrainTransmissionDiffLockRearEngagementPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DiffLockRearEngagement property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDiffLockRearEngagementChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TorqueDistribution value changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainTransmissionTorqueDistributionPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueDistributionChanged(VehiclePowertrainTransmissionTorqueDistributionPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TorqueDistribution property changes.
    * If your subscriber uses subscription with IVehiclePowertrainTransmissionSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueDistributionChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property Type changes.
    * @param The new value of Type.
    * Transmission type.
    */
    virtual void publishTypeChanged(Vehicle_Powertrain_Transmission_Type_ValueEnum Type) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property GearCount changes.
    * @param The new value of GearCount.
    * Number of forward gears in the transmission. -1 = CVT.
    */
    virtual void publishGearCountChanged(int32_t GearCount) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property DriveType changes.
    * @param The new value of DriveType.
    * Drive type.
    */
    virtual void publishDriveTypeChanged(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property TravelledDistance changes.
    * @param The new value of TravelledDistance.
    * Odometer reading, total distance travelled during the lifetime of the transmission.
    */
    virtual void publishTravelledDistanceChanged(float TravelledDistance) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property CurrentGear changes.
    * @param The new value of CurrentGear.
    * The current gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse.
    */
    virtual void publishCurrentGearChanged(int32_t CurrentGear) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property SelectedGear changes.
    * @param The new value of SelectedGear.
    * The selected gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse, 126=Park, 127=Drive.
    */
    virtual void publishSelectedGearChanged(int32_t SelectedGear) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property IsParkLockEngaged changes.
    * @param The new value of IsParkLockEngaged.
    * Is the transmission park lock engaged or not. False = Disengaged. True = Engaged.
    */
    virtual void publishIsParkLockEngagedChanged(bool IsParkLockEngaged) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property IsLowRangeEngaged changes.
    * @param The new value of IsLowRangeEngaged.
    * Is gearbox in low range mode or not. False = Normal/High range engaged. True = Low range engaged.
    */
    virtual void publishIsLowRangeEngagedChanged(bool IsLowRangeEngaged) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property IsElectricalPowertrainEngaged changes.
    * @param The new value of IsElectricalPowertrainEngaged.
    * Is electrical powertrain mechanically connected/engaged to the drivetrain or not. False = Disconnected/Disengaged. True = Connected/Engaged.
    */
    virtual void publishIsElectricalPowertrainEngagedChanged(bool IsElectricalPowertrainEngaged) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property PerformanceMode changes.
    * @param The new value of PerformanceMode.
    * Current gearbox performance mode.
    */
    virtual void publishPerformanceModeChanged(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property GearChangeMode changes.
    * @param The new value of GearChangeMode.
    * Is the gearbox in automatic or manual (paddle) mode.
    */
    virtual void publishGearChangeModeChanged(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property Temperature changes.
    * @param The new value of Temperature.
    * The current gearbox temperature.
    */
    virtual void publishTemperatureChanged(float Temperature) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property ClutchEngagement changes.
    * @param The new value of ClutchEngagement.
    * Clutch engagement. 0% = Clutch fully disengaged. 100% = Clutch fully engaged.
    */
    virtual void publishClutchEngagementChanged(float ClutchEngagement) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property ClutchWear changes.
    * @param The new value of ClutchWear.
    * Clutch wear as a percent. 0 = no wear. 100 = worn.
    */
    virtual void publishClutchWearChanged(int32_t ClutchWear) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property DiffLockFrontEngagement changes.
    * @param The new value of DiffLockFrontEngagement.
    * Front Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    */
    virtual void publishDiffLockFrontEngagementChanged(float DiffLockFrontEngagement) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property DiffLockRearEngagement changes.
    * @param The new value of DiffLockRearEngagement.
    * Rear Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    */
    virtual void publishDiffLockRearEngagementChanged(float DiffLockRearEngagement) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_Transmission implementation when property TorqueDistribution changes.
    * @param The new value of TorqueDistribution.
    * Torque distribution between front and rear axle in percent. -100% = Full torque to front axle, 0% = 50:50 Front/Rear, 100% = Full torque to rear axle.
    */
    virtual void publishTorqueDistributionChanged(float TorqueDistribution) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
