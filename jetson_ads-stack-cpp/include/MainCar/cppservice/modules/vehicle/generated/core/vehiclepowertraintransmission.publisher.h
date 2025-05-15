#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintransmission.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTransmissionPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_Transmission and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTransmissionPublisher : public IVehiclePowertrainTransmissionPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTransmissionSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTransmissionSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToTypeChanged
    */
    long subscribeToTypeChanged(VehiclePowertrainTransmissionTypePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToTypeChanged
    */
    void unsubscribeFromTypeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToGearCountChanged
    */
    long subscribeToGearCountChanged(VehiclePowertrainTransmissionGearCountPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToGearCountChanged
    */
    void unsubscribeFromGearCountChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToDriveTypeChanged
    */
    long subscribeToDriveTypeChanged(VehiclePowertrainTransmissionDriveTypePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToDriveTypeChanged
    */
    void unsubscribeFromDriveTypeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToTravelledDistanceChanged
    */
    long subscribeToTravelledDistanceChanged(VehiclePowertrainTransmissionTravelledDistancePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToTravelledDistanceChanged
    */
    void unsubscribeFromTravelledDistanceChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToCurrentGearChanged
    */
    long subscribeToCurrentGearChanged(VehiclePowertrainTransmissionCurrentGearPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToCurrentGearChanged
    */
    void unsubscribeFromCurrentGearChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToSelectedGearChanged
    */
    long subscribeToSelectedGearChanged(VehiclePowertrainTransmissionSelectedGearPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToSelectedGearChanged
    */
    void unsubscribeFromSelectedGearChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToIsParkLockEngagedChanged
    */
    long subscribeToIsParkLockEngagedChanged(VehiclePowertrainTransmissionIsParkLockEngagedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToIsParkLockEngagedChanged
    */
    void unsubscribeFromIsParkLockEngagedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToIsLowRangeEngagedChanged
    */
    long subscribeToIsLowRangeEngagedChanged(VehiclePowertrainTransmissionIsLowRangeEngagedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToIsLowRangeEngagedChanged
    */
    void unsubscribeFromIsLowRangeEngagedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToIsElectricalPowertrainEngagedChanged
    */
    long subscribeToIsElectricalPowertrainEngagedChanged(VehiclePowertrainTransmissionIsElectricalPowertrainEngagedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToIsElectricalPowertrainEngagedChanged
    */
    void unsubscribeFromIsElectricalPowertrainEngagedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToPerformanceModeChanged
    */
    long subscribeToPerformanceModeChanged(VehiclePowertrainTransmissionPerformanceModePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToPerformanceModeChanged
    */
    void unsubscribeFromPerformanceModeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToGearChangeModeChanged
    */
    long subscribeToGearChangeModeChanged(VehiclePowertrainTransmissionGearChangeModePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToGearChangeModeChanged
    */
    void unsubscribeFromGearChangeModeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToTemperatureChanged
    */
    long subscribeToTemperatureChanged(VehiclePowertrainTransmissionTemperaturePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToTemperatureChanged
    */
    void unsubscribeFromTemperatureChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToClutchEngagementChanged
    */
    long subscribeToClutchEngagementChanged(VehiclePowertrainTransmissionClutchEngagementPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToClutchEngagementChanged
    */
    void unsubscribeFromClutchEngagementChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToClutchWearChanged
    */
    long subscribeToClutchWearChanged(VehiclePowertrainTransmissionClutchWearPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToClutchWearChanged
    */
    void unsubscribeFromClutchWearChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToDiffLockFrontEngagementChanged
    */
    long subscribeToDiffLockFrontEngagementChanged(VehiclePowertrainTransmissionDiffLockFrontEngagementPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToDiffLockFrontEngagementChanged
    */
    void unsubscribeFromDiffLockFrontEngagementChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToDiffLockRearEngagementChanged
    */
    long subscribeToDiffLockRearEngagementChanged(VehiclePowertrainTransmissionDiffLockRearEngagementPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToDiffLockRearEngagementChanged
    */
    void unsubscribeFromDiffLockRearEngagementChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToTorqueDistributionChanged
    */
    long subscribeToTorqueDistributionChanged(VehiclePowertrainTransmissionTorqueDistributionPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::subscribeToTorqueDistributionChanged
    */
    void unsubscribeFromTorqueDistributionChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishTypeChanged
    */
    void publishTypeChanged(Vehicle_Powertrain_Transmission_Type_ValueEnum Type) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishGearCountChanged
    */
    void publishGearCountChanged(int32_t GearCount) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishDriveTypeChanged
    */
    void publishDriveTypeChanged(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishTravelledDistanceChanged
    */
    void publishTravelledDistanceChanged(float TravelledDistance) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishCurrentGearChanged
    */
    void publishCurrentGearChanged(int32_t CurrentGear) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishSelectedGearChanged
    */
    void publishSelectedGearChanged(int32_t SelectedGear) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishIsParkLockEngagedChanged
    */
    void publishIsParkLockEngagedChanged(bool IsParkLockEngaged) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishIsLowRangeEngagedChanged
    */
    void publishIsLowRangeEngagedChanged(bool IsLowRangeEngaged) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishIsElectricalPowertrainEngagedChanged
    */
    void publishIsElectricalPowertrainEngagedChanged(bool IsElectricalPowertrainEngaged) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishPerformanceModeChanged
    */
    void publishPerformanceModeChanged(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishGearChangeModeChanged
    */
    void publishGearChangeModeChanged(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishTemperatureChanged
    */
    void publishTemperatureChanged(float Temperature) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishClutchEngagementChanged
    */
    void publishClutchEngagementChanged(float ClutchEngagement) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishClutchWearChanged
    */
    void publishClutchWearChanged(int32_t ClutchWear) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishDiffLockFrontEngagementChanged
    */
    void publishDiffLockFrontEngagementChanged(float DiffLockFrontEngagement) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishDiffLockRearEngagementChanged
    */
    void publishDiffLockRearEngagementChanged(float DiffLockRearEngagement) const override;
    /**
    * Implementation of IVehiclePowertrainTransmissionPublisher::publishTorqueDistributionChanged
    */
    void publishTorqueDistributionChanged(float TorqueDistribution) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_Transmission
    std::vector<std::reference_wrapper<IVehiclePowertrainTransmissionSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Type change.
    std::atomic<long> m_typeChangedCallbackNextId {0};
    // Subscribed callbacks for the Type change.
    std::map<long, VehiclePowertrainTransmissionTypePropertyCb> m_typeCallbacks;
    // Mutex for m_typeCallbacks
    mutable std::shared_timed_mutex m_typeCallbacksMutex;
    // Next free unique identifier to subscribe for the GearCount change.
    std::atomic<long> m_gearCountChangedCallbackNextId {0};
    // Subscribed callbacks for the GearCount change.
    std::map<long, VehiclePowertrainTransmissionGearCountPropertyCb> m_gearCountCallbacks;
    // Mutex for m_gearCountCallbacks
    mutable std::shared_timed_mutex m_gearCountCallbacksMutex;
    // Next free unique identifier to subscribe for the DriveType change.
    std::atomic<long> m_driveTypeChangedCallbackNextId {0};
    // Subscribed callbacks for the DriveType change.
    std::map<long, VehiclePowertrainTransmissionDriveTypePropertyCb> m_driveTypeCallbacks;
    // Mutex for m_driveTypeCallbacks
    mutable std::shared_timed_mutex m_driveTypeCallbacksMutex;
    // Next free unique identifier to subscribe for the TravelledDistance change.
    std::atomic<long> m_travelledDistanceChangedCallbackNextId {0};
    // Subscribed callbacks for the TravelledDistance change.
    std::map<long, VehiclePowertrainTransmissionTravelledDistancePropertyCb> m_travelledDistanceCallbacks;
    // Mutex for m_travelledDistanceCallbacks
    mutable std::shared_timed_mutex m_travelledDistanceCallbacksMutex;
    // Next free unique identifier to subscribe for the CurrentGear change.
    std::atomic<long> m_currentGearChangedCallbackNextId {0};
    // Subscribed callbacks for the CurrentGear change.
    std::map<long, VehiclePowertrainTransmissionCurrentGearPropertyCb> m_currentGearCallbacks;
    // Mutex for m_currentGearCallbacks
    mutable std::shared_timed_mutex m_currentGearCallbacksMutex;
    // Next free unique identifier to subscribe for the SelectedGear change.
    std::atomic<long> m_selectedGearChangedCallbackNextId {0};
    // Subscribed callbacks for the SelectedGear change.
    std::map<long, VehiclePowertrainTransmissionSelectedGearPropertyCb> m_selectedGearCallbacks;
    // Mutex for m_selectedGearCallbacks
    mutable std::shared_timed_mutex m_selectedGearCallbacksMutex;
    // Next free unique identifier to subscribe for the IsParkLockEngaged change.
    std::atomic<long> m_isParkLockEngagedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsParkLockEngaged change.
    std::map<long, VehiclePowertrainTransmissionIsParkLockEngagedPropertyCb> m_isParkLockEngagedCallbacks;
    // Mutex for m_isParkLockEngagedCallbacks
    mutable std::shared_timed_mutex m_isParkLockEngagedCallbacksMutex;
    // Next free unique identifier to subscribe for the IsLowRangeEngaged change.
    std::atomic<long> m_isLowRangeEngagedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsLowRangeEngaged change.
    std::map<long, VehiclePowertrainTransmissionIsLowRangeEngagedPropertyCb> m_isLowRangeEngagedCallbacks;
    // Mutex for m_isLowRangeEngagedCallbacks
    mutable std::shared_timed_mutex m_isLowRangeEngagedCallbacksMutex;
    // Next free unique identifier to subscribe for the IsElectricalPowertrainEngaged change.
    std::atomic<long> m_isElectricalPowertrainEngagedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsElectricalPowertrainEngaged change.
    std::map<long, VehiclePowertrainTransmissionIsElectricalPowertrainEngagedPropertyCb> m_isElectricalPowertrainEngagedCallbacks;
    // Mutex for m_isElectricalPowertrainEngagedCallbacks
    mutable std::shared_timed_mutex m_isElectricalPowertrainEngagedCallbacksMutex;
    // Next free unique identifier to subscribe for the PerformanceMode change.
    std::atomic<long> m_performanceModeChangedCallbackNextId {0};
    // Subscribed callbacks for the PerformanceMode change.
    std::map<long, VehiclePowertrainTransmissionPerformanceModePropertyCb> m_performanceModeCallbacks;
    // Mutex for m_performanceModeCallbacks
    mutable std::shared_timed_mutex m_performanceModeCallbacksMutex;
    // Next free unique identifier to subscribe for the GearChangeMode change.
    std::atomic<long> m_gearChangeModeChangedCallbackNextId {0};
    // Subscribed callbacks for the GearChangeMode change.
    std::map<long, VehiclePowertrainTransmissionGearChangeModePropertyCb> m_gearChangeModeCallbacks;
    // Mutex for m_gearChangeModeCallbacks
    mutable std::shared_timed_mutex m_gearChangeModeCallbacksMutex;
    // Next free unique identifier to subscribe for the Temperature change.
    std::atomic<long> m_temperatureChangedCallbackNextId {0};
    // Subscribed callbacks for the Temperature change.
    std::map<long, VehiclePowertrainTransmissionTemperaturePropertyCb> m_temperatureCallbacks;
    // Mutex for m_temperatureCallbacks
    mutable std::shared_timed_mutex m_temperatureCallbacksMutex;
    // Next free unique identifier to subscribe for the ClutchEngagement change.
    std::atomic<long> m_clutchEngagementChangedCallbackNextId {0};
    // Subscribed callbacks for the ClutchEngagement change.
    std::map<long, VehiclePowertrainTransmissionClutchEngagementPropertyCb> m_clutchEngagementCallbacks;
    // Mutex for m_clutchEngagementCallbacks
    mutable std::shared_timed_mutex m_clutchEngagementCallbacksMutex;
    // Next free unique identifier to subscribe for the ClutchWear change.
    std::atomic<long> m_clutchWearChangedCallbackNextId {0};
    // Subscribed callbacks for the ClutchWear change.
    std::map<long, VehiclePowertrainTransmissionClutchWearPropertyCb> m_clutchWearCallbacks;
    // Mutex for m_clutchWearCallbacks
    mutable std::shared_timed_mutex m_clutchWearCallbacksMutex;
    // Next free unique identifier to subscribe for the DiffLockFrontEngagement change.
    std::atomic<long> m_diffLockFrontEngagementChangedCallbackNextId {0};
    // Subscribed callbacks for the DiffLockFrontEngagement change.
    std::map<long, VehiclePowertrainTransmissionDiffLockFrontEngagementPropertyCb> m_diffLockFrontEngagementCallbacks;
    // Mutex for m_diffLockFrontEngagementCallbacks
    mutable std::shared_timed_mutex m_diffLockFrontEngagementCallbacksMutex;
    // Next free unique identifier to subscribe for the DiffLockRearEngagement change.
    std::atomic<long> m_diffLockRearEngagementChangedCallbackNextId {0};
    // Subscribed callbacks for the DiffLockRearEngagement change.
    std::map<long, VehiclePowertrainTransmissionDiffLockRearEngagementPropertyCb> m_diffLockRearEngagementCallbacks;
    // Mutex for m_diffLockRearEngagementCallbacks
    mutable std::shared_timed_mutex m_diffLockRearEngagementCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueDistribution change.
    std::atomic<long> m_torqueDistributionChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueDistribution change.
    std::map<long, VehiclePowertrainTransmissionTorqueDistributionPropertyCb> m_torqueDistributionCallbacks;
    // Mutex for m_torqueDistributionCallbacks
    mutable std::shared_timed_mutex m_torqueDistributionCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
