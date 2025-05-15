#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbattery.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryPublisher : public IVehiclePowertrainTractionBatteryPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatterySubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatterySubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToIdChanged
    */
    long subscribeToIdChanged(VehiclePowertrainTractionBatteryIdPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToIdChanged
    */
    void unsubscribeFromIdChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToProductionDateChanged
    */
    long subscribeToProductionDateChanged(VehiclePowertrainTractionBatteryProductionDatePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToProductionDateChanged
    */
    void unsubscribeFromProductionDateChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToErrorCodesChanged
    */
    long subscribeToErrorCodesChanged(VehiclePowertrainTractionBatteryErrorCodesPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToErrorCodesChanged
    */
    void unsubscribeFromErrorCodesChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToIsPowerConnectedChanged
    */
    long subscribeToIsPowerConnectedChanged(VehiclePowertrainTractionBatteryIsPowerConnectedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToIsPowerConnectedChanged
    */
    void unsubscribeFromIsPowerConnectedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToIsGroundConnectedChanged
    */
    long subscribeToIsGroundConnectedChanged(VehiclePowertrainTractionBatteryIsGroundConnectedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToIsGroundConnectedChanged
    */
    void unsubscribeFromIsGroundConnectedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToGrossCapacityChanged
    */
    long subscribeToGrossCapacityChanged(VehiclePowertrainTractionBatteryGrossCapacityPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToGrossCapacityChanged
    */
    void unsubscribeFromGrossCapacityChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToNetCapacityChanged
    */
    long subscribeToNetCapacityChanged(VehiclePowertrainTractionBatteryNetCapacityPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToNetCapacityChanged
    */
    void unsubscribeFromNetCapacityChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToStateOfHealthChanged
    */
    long subscribeToStateOfHealthChanged(VehiclePowertrainTractionBatteryStateOfHealthPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToStateOfHealthChanged
    */
    void unsubscribeFromStateOfHealthChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToNominalVoltageChanged
    */
    long subscribeToNominalVoltageChanged(VehiclePowertrainTractionBatteryNominalVoltagePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToNominalVoltageChanged
    */
    void unsubscribeFromNominalVoltageChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToMaxVoltageChanged
    */
    long subscribeToMaxVoltageChanged(VehiclePowertrainTractionBatteryMaxVoltagePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToMaxVoltageChanged
    */
    void unsubscribeFromMaxVoltageChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToCurrentVoltageChanged
    */
    long subscribeToCurrentVoltageChanged(VehiclePowertrainTractionBatteryCurrentVoltagePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToCurrentVoltageChanged
    */
    void unsubscribeFromCurrentVoltageChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToCurrentCurrentChanged
    */
    long subscribeToCurrentCurrentChanged(VehiclePowertrainTractionBatteryCurrentCurrentPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToCurrentCurrentChanged
    */
    void unsubscribeFromCurrentCurrentChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToCurrentPowerChanged
    */
    long subscribeToCurrentPowerChanged(VehiclePowertrainTractionBatteryCurrentPowerPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToCurrentPowerChanged
    */
    void unsubscribeFromCurrentPowerChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedChargedEnergyChanged
    */
    long subscribeToAccumulatedChargedEnergyChanged(VehiclePowertrainTractionBatteryAccumulatedChargedEnergyPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedChargedEnergyChanged
    */
    void unsubscribeFromAccumulatedChargedEnergyChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedConsumedEnergyChanged
    */
    long subscribeToAccumulatedConsumedEnergyChanged(VehiclePowertrainTractionBatteryAccumulatedConsumedEnergyPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedConsumedEnergyChanged
    */
    void unsubscribeFromAccumulatedConsumedEnergyChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedChargedThroughputChanged
    */
    long subscribeToAccumulatedChargedThroughputChanged(VehiclePowertrainTractionBatteryAccumulatedChargedThroughputPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedChargedThroughputChanged
    */
    void unsubscribeFromAccumulatedChargedThroughputChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedConsumedThroughputChanged
    */
    long subscribeToAccumulatedConsumedThroughputChanged(VehiclePowertrainTractionBatteryAccumulatedConsumedThroughputPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedConsumedThroughputChanged
    */
    void unsubscribeFromAccumulatedConsumedThroughputChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToPowerLossChanged
    */
    long subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryPowerLossPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToPowerLossChanged
    */
    void unsubscribeFromPowerLossChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToRangeChanged
    */
    long subscribeToRangeChanged(VehiclePowertrainTractionBatteryRangePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToRangeChanged
    */
    void unsubscribeFromRangeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToTimeRemainingChanged
    */
    long subscribeToTimeRemainingChanged(VehiclePowertrainTractionBatteryTimeRemainingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::subscribeToTimeRemainingChanged
    */
    void unsubscribeFromTimeRemainingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishIdChanged
    */
    void publishIdChanged(const std::string& Id) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishProductionDateChanged
    */
    void publishProductionDateChanged(const std::string& ProductionDate) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishErrorCodesChanged
    */
    void publishErrorCodesChanged(const std::list<std::string>& ErrorCodes) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishIsPowerConnectedChanged
    */
    void publishIsPowerConnectedChanged(bool IsPowerConnected) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishIsGroundConnectedChanged
    */
    void publishIsGroundConnectedChanged(bool IsGroundConnected) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishGrossCapacityChanged
    */
    void publishGrossCapacityChanged(int32_t GrossCapacity) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishNetCapacityChanged
    */
    void publishNetCapacityChanged(int32_t NetCapacity) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishStateOfHealthChanged
    */
    void publishStateOfHealthChanged(float StateOfHealth) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishNominalVoltageChanged
    */
    void publishNominalVoltageChanged(int32_t NominalVoltage) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishMaxVoltageChanged
    */
    void publishMaxVoltageChanged(int32_t MaxVoltage) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishCurrentVoltageChanged
    */
    void publishCurrentVoltageChanged(float CurrentVoltage) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishCurrentCurrentChanged
    */
    void publishCurrentCurrentChanged(float CurrentCurrent) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishCurrentPowerChanged
    */
    void publishCurrentPowerChanged(float CurrentPower) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishAccumulatedChargedEnergyChanged
    */
    void publishAccumulatedChargedEnergyChanged(float AccumulatedChargedEnergy) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishAccumulatedConsumedEnergyChanged
    */
    void publishAccumulatedConsumedEnergyChanged(float AccumulatedConsumedEnergy) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishAccumulatedChargedThroughputChanged
    */
    void publishAccumulatedChargedThroughputChanged(float AccumulatedChargedThroughput) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishAccumulatedConsumedThroughputChanged
    */
    void publishAccumulatedConsumedThroughputChanged(float AccumulatedConsumedThroughput) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishPowerLossChanged
    */
    void publishPowerLossChanged(float PowerLoss) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishRangeChanged
    */
    void publishRangeChanged(int32_t Range) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryPublisher::publishTimeRemainingChanged
    */
    void publishTimeRemainingChanged(int32_t TimeRemaining) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatterySubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Id change.
    std::atomic<long> m_idChangedCallbackNextId {0};
    // Subscribed callbacks for the Id change.
    std::map<long, VehiclePowertrainTractionBatteryIdPropertyCb> m_idCallbacks;
    // Mutex for m_idCallbacks
    mutable std::shared_timed_mutex m_idCallbacksMutex;
    // Next free unique identifier to subscribe for the ProductionDate change.
    std::atomic<long> m_productionDateChangedCallbackNextId {0};
    // Subscribed callbacks for the ProductionDate change.
    std::map<long, VehiclePowertrainTractionBatteryProductionDatePropertyCb> m_productionDateCallbacks;
    // Mutex for m_productionDateCallbacks
    mutable std::shared_timed_mutex m_productionDateCallbacksMutex;
    // Next free unique identifier to subscribe for the ErrorCodes change.
    std::atomic<long> m_errorCodesChangedCallbackNextId {0};
    // Subscribed callbacks for the ErrorCodes change.
    std::map<long, VehiclePowertrainTractionBatteryErrorCodesPropertyCb> m_errorCodesCallbacks;
    // Mutex for m_errorCodesCallbacks
    mutable std::shared_timed_mutex m_errorCodesCallbacksMutex;
    // Next free unique identifier to subscribe for the IsPowerConnected change.
    std::atomic<long> m_isPowerConnectedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsPowerConnected change.
    std::map<long, VehiclePowertrainTractionBatteryIsPowerConnectedPropertyCb> m_isPowerConnectedCallbacks;
    // Mutex for m_isPowerConnectedCallbacks
    mutable std::shared_timed_mutex m_isPowerConnectedCallbacksMutex;
    // Next free unique identifier to subscribe for the IsGroundConnected change.
    std::atomic<long> m_isGroundConnectedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsGroundConnected change.
    std::map<long, VehiclePowertrainTractionBatteryIsGroundConnectedPropertyCb> m_isGroundConnectedCallbacks;
    // Mutex for m_isGroundConnectedCallbacks
    mutable std::shared_timed_mutex m_isGroundConnectedCallbacksMutex;
    // Next free unique identifier to subscribe for the GrossCapacity change.
    std::atomic<long> m_grossCapacityChangedCallbackNextId {0};
    // Subscribed callbacks for the GrossCapacity change.
    std::map<long, VehiclePowertrainTractionBatteryGrossCapacityPropertyCb> m_grossCapacityCallbacks;
    // Mutex for m_grossCapacityCallbacks
    mutable std::shared_timed_mutex m_grossCapacityCallbacksMutex;
    // Next free unique identifier to subscribe for the NetCapacity change.
    std::atomic<long> m_netCapacityChangedCallbackNextId {0};
    // Subscribed callbacks for the NetCapacity change.
    std::map<long, VehiclePowertrainTractionBatteryNetCapacityPropertyCb> m_netCapacityCallbacks;
    // Mutex for m_netCapacityCallbacks
    mutable std::shared_timed_mutex m_netCapacityCallbacksMutex;
    // Next free unique identifier to subscribe for the StateOfHealth change.
    std::atomic<long> m_stateOfHealthChangedCallbackNextId {0};
    // Subscribed callbacks for the StateOfHealth change.
    std::map<long, VehiclePowertrainTractionBatteryStateOfHealthPropertyCb> m_stateOfHealthCallbacks;
    // Mutex for m_stateOfHealthCallbacks
    mutable std::shared_timed_mutex m_stateOfHealthCallbacksMutex;
    // Next free unique identifier to subscribe for the NominalVoltage change.
    std::atomic<long> m_nominalVoltageChangedCallbackNextId {0};
    // Subscribed callbacks for the NominalVoltage change.
    std::map<long, VehiclePowertrainTractionBatteryNominalVoltagePropertyCb> m_nominalVoltageCallbacks;
    // Mutex for m_nominalVoltageCallbacks
    mutable std::shared_timed_mutex m_nominalVoltageCallbacksMutex;
    // Next free unique identifier to subscribe for the MaxVoltage change.
    std::atomic<long> m_maxVoltageChangedCallbackNextId {0};
    // Subscribed callbacks for the MaxVoltage change.
    std::map<long, VehiclePowertrainTractionBatteryMaxVoltagePropertyCb> m_maxVoltageCallbacks;
    // Mutex for m_maxVoltageCallbacks
    mutable std::shared_timed_mutex m_maxVoltageCallbacksMutex;
    // Next free unique identifier to subscribe for the CurrentVoltage change.
    std::atomic<long> m_currentVoltageChangedCallbackNextId {0};
    // Subscribed callbacks for the CurrentVoltage change.
    std::map<long, VehiclePowertrainTractionBatteryCurrentVoltagePropertyCb> m_currentVoltageCallbacks;
    // Mutex for m_currentVoltageCallbacks
    mutable std::shared_timed_mutex m_currentVoltageCallbacksMutex;
    // Next free unique identifier to subscribe for the CurrentCurrent change.
    std::atomic<long> m_currentCurrentChangedCallbackNextId {0};
    // Subscribed callbacks for the CurrentCurrent change.
    std::map<long, VehiclePowertrainTractionBatteryCurrentCurrentPropertyCb> m_currentCurrentCallbacks;
    // Mutex for m_currentCurrentCallbacks
    mutable std::shared_timed_mutex m_currentCurrentCallbacksMutex;
    // Next free unique identifier to subscribe for the CurrentPower change.
    std::atomic<long> m_currentPowerChangedCallbackNextId {0};
    // Subscribed callbacks for the CurrentPower change.
    std::map<long, VehiclePowertrainTractionBatteryCurrentPowerPropertyCb> m_currentPowerCallbacks;
    // Mutex for m_currentPowerCallbacks
    mutable std::shared_timed_mutex m_currentPowerCallbacksMutex;
    // Next free unique identifier to subscribe for the AccumulatedChargedEnergy change.
    std::atomic<long> m_accumulatedChargedEnergyChangedCallbackNextId {0};
    // Subscribed callbacks for the AccumulatedChargedEnergy change.
    std::map<long, VehiclePowertrainTractionBatteryAccumulatedChargedEnergyPropertyCb> m_accumulatedChargedEnergyCallbacks;
    // Mutex for m_accumulatedChargedEnergyCallbacks
    mutable std::shared_timed_mutex m_accumulatedChargedEnergyCallbacksMutex;
    // Next free unique identifier to subscribe for the AccumulatedConsumedEnergy change.
    std::atomic<long> m_accumulatedConsumedEnergyChangedCallbackNextId {0};
    // Subscribed callbacks for the AccumulatedConsumedEnergy change.
    std::map<long, VehiclePowertrainTractionBatteryAccumulatedConsumedEnergyPropertyCb> m_accumulatedConsumedEnergyCallbacks;
    // Mutex for m_accumulatedConsumedEnergyCallbacks
    mutable std::shared_timed_mutex m_accumulatedConsumedEnergyCallbacksMutex;
    // Next free unique identifier to subscribe for the AccumulatedChargedThroughput change.
    std::atomic<long> m_accumulatedChargedThroughputChangedCallbackNextId {0};
    // Subscribed callbacks for the AccumulatedChargedThroughput change.
    std::map<long, VehiclePowertrainTractionBatteryAccumulatedChargedThroughputPropertyCb> m_accumulatedChargedThroughputCallbacks;
    // Mutex for m_accumulatedChargedThroughputCallbacks
    mutable std::shared_timed_mutex m_accumulatedChargedThroughputCallbacksMutex;
    // Next free unique identifier to subscribe for the AccumulatedConsumedThroughput change.
    std::atomic<long> m_accumulatedConsumedThroughputChangedCallbackNextId {0};
    // Subscribed callbacks for the AccumulatedConsumedThroughput change.
    std::map<long, VehiclePowertrainTractionBatteryAccumulatedConsumedThroughputPropertyCb> m_accumulatedConsumedThroughputCallbacks;
    // Mutex for m_accumulatedConsumedThroughputCallbacks
    mutable std::shared_timed_mutex m_accumulatedConsumedThroughputCallbacksMutex;
    // Next free unique identifier to subscribe for the PowerLoss change.
    std::atomic<long> m_powerLossChangedCallbackNextId {0};
    // Subscribed callbacks for the PowerLoss change.
    std::map<long, VehiclePowertrainTractionBatteryPowerLossPropertyCb> m_powerLossCallbacks;
    // Mutex for m_powerLossCallbacks
    mutable std::shared_timed_mutex m_powerLossCallbacksMutex;
    // Next free unique identifier to subscribe for the Range change.
    std::atomic<long> m_rangeChangedCallbackNextId {0};
    // Subscribed callbacks for the Range change.
    std::map<long, VehiclePowertrainTractionBatteryRangePropertyCb> m_rangeCallbacks;
    // Mutex for m_rangeCallbacks
    mutable std::shared_timed_mutex m_rangeCallbacksMutex;
    // Next free unique identifier to subscribe for the TimeRemaining change.
    std::atomic<long> m_timeRemainingChangedCallbackNextId {0};
    // Subscribed callbacks for the TimeRemaining change.
    std::map<long, VehiclePowertrainTractionBatteryTimeRemainingPropertyCb> m_timeRemainingCallbacks;
    // Mutex for m_timeRemainingCallbacks
    mutable std::shared_timed_mutex m_timeRemainingCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
