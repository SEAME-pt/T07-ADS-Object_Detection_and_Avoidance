#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertrain.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainPublisher.
 * Use this class to store clients of the Vehicle_Powertrain and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainPublisher : public IVehiclePowertrainPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToAccumulatedBrakingEnergyChanged
    */
    long subscribeToAccumulatedBrakingEnergyChanged(VehiclePowertrainAccumulatedBrakingEnergyPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToAccumulatedBrakingEnergyChanged
    */
    void unsubscribeFromAccumulatedBrakingEnergyChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToRangeChanged
    */
    long subscribeToRangeChanged(VehiclePowertrainRangePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToRangeChanged
    */
    void unsubscribeFromRangeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToTimeRemainingChanged
    */
    long subscribeToTimeRemainingChanged(VehiclePowertrainTimeRemainingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToTimeRemainingChanged
    */
    void unsubscribeFromTimeRemainingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToTypeChanged
    */
    long subscribeToTypeChanged(VehiclePowertrainTypePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToTypeChanged
    */
    void unsubscribeFromTypeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToPowerOptimizeLevelChanged
    */
    long subscribeToPowerOptimizeLevelChanged(VehiclePowertrainPowerOptimizeLevelPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToPowerOptimizeLevelChanged
    */
    void unsubscribeFromPowerOptimizeLevelChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToIsAutoPowerOptimizeChanged
    */
    long subscribeToIsAutoPowerOptimizeChanged(VehiclePowertrainIsAutoPowerOptimizePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainPublisher::subscribeToIsAutoPowerOptimizeChanged
    */
    void unsubscribeFromIsAutoPowerOptimizeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainPublisher::publishAccumulatedBrakingEnergyChanged
    */
    void publishAccumulatedBrakingEnergyChanged(float AccumulatedBrakingEnergy) const override;
    /**
    * Implementation of IVehiclePowertrainPublisher::publishRangeChanged
    */
    void publishRangeChanged(int32_t Range) const override;
    /**
    * Implementation of IVehiclePowertrainPublisher::publishTimeRemainingChanged
    */
    void publishTimeRemainingChanged(int32_t TimeRemaining) const override;
    /**
    * Implementation of IVehiclePowertrainPublisher::publishTypeChanged
    */
    void publishTypeChanged(Vehicle_Powertrain_Type_ValueEnum Type) const override;
    /**
    * Implementation of IVehiclePowertrainPublisher::publishPowerOptimizeLevelChanged
    */
    void publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const override;
    /**
    * Implementation of IVehiclePowertrainPublisher::publishIsAutoPowerOptimizeChanged
    */
    void publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain
    std::vector<std::reference_wrapper<IVehiclePowertrainSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the AccumulatedBrakingEnergy change.
    std::atomic<long> m_accumulatedBrakingEnergyChangedCallbackNextId {0};
    // Subscribed callbacks for the AccumulatedBrakingEnergy change.
    std::map<long, VehiclePowertrainAccumulatedBrakingEnergyPropertyCb> m_accumulatedBrakingEnergyCallbacks;
    // Mutex for m_accumulatedBrakingEnergyCallbacks
    mutable std::shared_timed_mutex m_accumulatedBrakingEnergyCallbacksMutex;
    // Next free unique identifier to subscribe for the Range change.
    std::atomic<long> m_rangeChangedCallbackNextId {0};
    // Subscribed callbacks for the Range change.
    std::map<long, VehiclePowertrainRangePropertyCb> m_rangeCallbacks;
    // Mutex for m_rangeCallbacks
    mutable std::shared_timed_mutex m_rangeCallbacksMutex;
    // Next free unique identifier to subscribe for the TimeRemaining change.
    std::atomic<long> m_timeRemainingChangedCallbackNextId {0};
    // Subscribed callbacks for the TimeRemaining change.
    std::map<long, VehiclePowertrainTimeRemainingPropertyCb> m_timeRemainingCallbacks;
    // Mutex for m_timeRemainingCallbacks
    mutable std::shared_timed_mutex m_timeRemainingCallbacksMutex;
    // Next free unique identifier to subscribe for the Type change.
    std::atomic<long> m_typeChangedCallbackNextId {0};
    // Subscribed callbacks for the Type change.
    std::map<long, VehiclePowertrainTypePropertyCb> m_typeCallbacks;
    // Mutex for m_typeCallbacks
    mutable std::shared_timed_mutex m_typeCallbacksMutex;
    // Next free unique identifier to subscribe for the PowerOptimizeLevel change.
    std::atomic<long> m_powerOptimizeLevelChangedCallbackNextId {0};
    // Subscribed callbacks for the PowerOptimizeLevel change.
    std::map<long, VehiclePowertrainPowerOptimizeLevelPropertyCb> m_powerOptimizeLevelCallbacks;
    // Mutex for m_powerOptimizeLevelCallbacks
    mutable std::shared_timed_mutex m_powerOptimizeLevelCallbacksMutex;
    // Next free unique identifier to subscribe for the IsAutoPowerOptimize change.
    std::atomic<long> m_isAutoPowerOptimizeChangedCallbackNextId {0};
    // Subscribed callbacks for the IsAutoPowerOptimize change.
    std::map<long, VehiclePowertrainIsAutoPowerOptimizePropertyCb> m_isAutoPowerOptimizeCallbacks;
    // Mutex for m_isAutoPowerOptimizeCallbacks
    mutable std::shared_timed_mutex m_isAutoPowerOptimizeCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
