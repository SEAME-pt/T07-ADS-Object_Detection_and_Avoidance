#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterychargingchargecurrent.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingChargeCurrentPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargeCurrentPublisher : public IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToDcChanged
    */
    long subscribeToDcChanged(VehiclePowertrainTractionBatteryChargingChargeCurrentDcPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToDcChanged
    */
    void unsubscribeFromDcChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToPhase1Changed
    */
    long subscribeToPhase1Changed(VehiclePowertrainTractionBatteryChargingChargeCurrentPhase1PropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToPhase1Changed
    */
    void unsubscribeFromPhase1Changed(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToPhase2Changed
    */
    long subscribeToPhase2Changed(VehiclePowertrainTractionBatteryChargingChargeCurrentPhase2PropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToPhase2Changed
    */
    void unsubscribeFromPhase2Changed(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToPhase3Changed
    */
    long subscribeToPhase3Changed(VehiclePowertrainTractionBatteryChargingChargeCurrentPhase3PropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::subscribeToPhase3Changed
    */
    void unsubscribeFromPhase3Changed(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::publishDcChanged
    */
    void publishDcChanged(float DC) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::publishPhase1Changed
    */
    void publishPhase1Changed(float Phase1) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::publishPhase2Changed
    */
    void publishPhase2Changed(float Phase2) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher::publishPhase3Changed
    */
    void publishPhase3Changed(float Phase3) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingChargeCurrentSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Dc change.
    std::atomic<long> m_dcChangedCallbackNextId {0};
    // Subscribed callbacks for the Dc change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargeCurrentDcPropertyCb> m_dcCallbacks;
    // Mutex for m_dcCallbacks
    mutable std::shared_timed_mutex m_dcCallbacksMutex;
    // Next free unique identifier to subscribe for the Phase1 change.
    std::atomic<long> m_phase1ChangedCallbackNextId {0};
    // Subscribed callbacks for the Phase1 change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargeCurrentPhase1PropertyCb> m_phase1Callbacks;
    // Mutex for m_phase1Callbacks
    mutable std::shared_timed_mutex m_phase1CallbacksMutex;
    // Next free unique identifier to subscribe for the Phase2 change.
    std::atomic<long> m_phase2ChangedCallbackNextId {0};
    // Subscribed callbacks for the Phase2 change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargeCurrentPhase2PropertyCb> m_phase2Callbacks;
    // Mutex for m_phase2Callbacks
    mutable std::shared_timed_mutex m_phase2CallbacksMutex;
    // Next free unique identifier to subscribe for the Phase3 change.
    std::atomic<long> m_phase3ChangedCallbackNextId {0};
    // Subscribed callbacks for the Phase3 change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargeCurrentPhase3PropertyCb> m_phase3Callbacks;
    // Mutex for m_phase3Callbacks
    mutable std::shared_timed_mutex m_phase3CallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
