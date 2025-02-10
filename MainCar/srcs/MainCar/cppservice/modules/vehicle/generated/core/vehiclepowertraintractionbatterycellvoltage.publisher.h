#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterycellvoltage.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryCellVoltagePublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_CellVoltage and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryCellVoltagePublisher : public IVehiclePowertrainTractionBatteryCellVoltagePublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryCellVoltageSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryCellVoltageSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToMinChanged
    */
    long subscribeToMinChanged(VehiclePowertrainTractionBatteryCellVoltageMinPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToMinChanged
    */
    void unsubscribeFromMinChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToIdMinChanged
    */
    long subscribeToIdMinChanged(VehiclePowertrainTractionBatteryCellVoltageIdMinPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToIdMinChanged
    */
    void unsubscribeFromIdMinChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToMaxChanged
    */
    long subscribeToMaxChanged(VehiclePowertrainTractionBatteryCellVoltageMaxPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToMaxChanged
    */
    void unsubscribeFromMaxChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToIdMaxChanged
    */
    long subscribeToIdMaxChanged(VehiclePowertrainTractionBatteryCellVoltageIdMaxPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToIdMaxChanged
    */
    void unsubscribeFromIdMaxChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToCellVoltagesChanged
    */
    long subscribeToCellVoltagesChanged(VehiclePowertrainTractionBatteryCellVoltageCellVoltagesPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToCellVoltagesChanged
    */
    void unsubscribeFromCellVoltagesChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::publishMinChanged
    */
    void publishMinChanged(float Min) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::publishIdMinChanged
    */
    void publishIdMinChanged(int32_t IdMin) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::publishMaxChanged
    */
    void publishMaxChanged(float Max) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::publishIdMaxChanged
    */
    void publishIdMaxChanged(int32_t IdMax) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryCellVoltagePublisher::publishCellVoltagesChanged
    */
    void publishCellVoltagesChanged(const std::list<float>& CellVoltages) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_CellVoltage
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryCellVoltageSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Min change.
    std::atomic<long> m_minChangedCallbackNextId {0};
    // Subscribed callbacks for the Min change.
    std::map<long, VehiclePowertrainTractionBatteryCellVoltageMinPropertyCb> m_minCallbacks;
    // Mutex for m_minCallbacks
    mutable std::shared_timed_mutex m_minCallbacksMutex;
    // Next free unique identifier to subscribe for the IdMin change.
    std::atomic<long> m_idMinChangedCallbackNextId {0};
    // Subscribed callbacks for the IdMin change.
    std::map<long, VehiclePowertrainTractionBatteryCellVoltageIdMinPropertyCb> m_idMinCallbacks;
    // Mutex for m_idMinCallbacks
    mutable std::shared_timed_mutex m_idMinCallbacksMutex;
    // Next free unique identifier to subscribe for the Max change.
    std::atomic<long> m_maxChangedCallbackNextId {0};
    // Subscribed callbacks for the Max change.
    std::map<long, VehiclePowertrainTractionBatteryCellVoltageMaxPropertyCb> m_maxCallbacks;
    // Mutex for m_maxCallbacks
    mutable std::shared_timed_mutex m_maxCallbacksMutex;
    // Next free unique identifier to subscribe for the IdMax change.
    std::atomic<long> m_idMaxChangedCallbackNextId {0};
    // Subscribed callbacks for the IdMax change.
    std::map<long, VehiclePowertrainTractionBatteryCellVoltageIdMaxPropertyCb> m_idMaxCallbacks;
    // Mutex for m_idMaxCallbacks
    mutable std::shared_timed_mutex m_idMaxCallbacksMutex;
    // Next free unique identifier to subscribe for the CellVoltages change.
    std::atomic<long> m_cellVoltagesChangedCallbackNextId {0};
    // Subscribed callbacks for the CellVoltages change.
    std::map<long, VehiclePowertrainTractionBatteryCellVoltageCellVoltagesPropertyCb> m_cellVoltagesCallbacks;
    // Mutex for m_cellVoltagesCallbacks
    mutable std::shared_timed_mutex m_cellVoltagesCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
