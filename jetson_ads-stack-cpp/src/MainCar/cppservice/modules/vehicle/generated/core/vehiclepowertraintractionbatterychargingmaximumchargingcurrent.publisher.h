#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher : public IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToDcChanged
    */
    long subscribeToDcChanged(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentDcPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToDcChanged
    */
    void unsubscribeFromDcChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase1Changed
    */
    long subscribeToPhase1Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase1PropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase1Changed
    */
    void unsubscribeFromPhase1Changed(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase2Changed
    */
    long subscribeToPhase2Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase2PropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase2Changed
    */
    void unsubscribeFromPhase2Changed(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase3Changed
    */
    long subscribeToPhase3Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase3PropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase3Changed
    */
    void unsubscribeFromPhase3Changed(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::publishDcChanged
    */
    void publishDcChanged(float DC) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::publishPhase1Changed
    */
    void publishPhase1Changed(float Phase1) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::publishPhase2Changed
    */
    void publishPhase2Changed(float Phase2) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::publishPhase3Changed
    */
    void publishPhase3Changed(float Phase3) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Dc change.
    std::atomic<long> m_dcChangedCallbackNextId {0};
    // Subscribed callbacks for the Dc change.
    std::map<long, VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentDcPropertyCb> m_dcCallbacks;
    // Mutex for m_dcCallbacks
    mutable std::shared_timed_mutex m_dcCallbacksMutex;
    // Next free unique identifier to subscribe for the Phase1 change.
    std::atomic<long> m_phase1ChangedCallbackNextId {0};
    // Subscribed callbacks for the Phase1 change.
    std::map<long, VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase1PropertyCb> m_phase1Callbacks;
    // Mutex for m_phase1Callbacks
    mutable std::shared_timed_mutex m_phase1CallbacksMutex;
    // Next free unique identifier to subscribe for the Phase2 change.
    std::atomic<long> m_phase2ChangedCallbackNextId {0};
    // Subscribed callbacks for the Phase2 change.
    std::map<long, VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase2PropertyCb> m_phase2Callbacks;
    // Mutex for m_phase2Callbacks
    mutable std::shared_timed_mutex m_phase2CallbacksMutex;
    // Next free unique identifier to subscribe for the Phase3 change.
    std::atomic<long> m_phase3ChangedCallbackNextId {0};
    // Subscribed callbacks for the Phase3 change.
    std::map<long, VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase3PropertyCb> m_phase3Callbacks;
    // Mutex for m_phase3Callbacks
    mutable std::shared_timed_mutex m_phase3CallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
