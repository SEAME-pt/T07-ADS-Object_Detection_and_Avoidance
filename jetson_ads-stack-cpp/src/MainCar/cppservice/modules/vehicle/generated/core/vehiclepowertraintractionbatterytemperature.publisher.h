#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterytemperature.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryTemperaturePublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Temperature and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryTemperaturePublisher : public IVehiclePowertrainTractionBatteryTemperaturePublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryTemperatureSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryTemperatureSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToAverageChanged
    */
    long subscribeToAverageChanged(VehiclePowertrainTractionBatteryTemperatureAveragePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToAverageChanged
    */
    void unsubscribeFromAverageChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToMinChanged
    */
    long subscribeToMinChanged(VehiclePowertrainTractionBatteryTemperatureMinPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToMinChanged
    */
    void unsubscribeFromMinChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToMaxChanged
    */
    long subscribeToMaxChanged(VehiclePowertrainTractionBatteryTemperatureMaxPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToMaxChanged
    */
    void unsubscribeFromMaxChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToCellTemperatureChanged
    */
    long subscribeToCellTemperatureChanged(VehiclePowertrainTractionBatteryTemperatureCellTemperaturePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToCellTemperatureChanged
    */
    void unsubscribeFromCellTemperatureChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::publishAverageChanged
    */
    void publishAverageChanged(float Average) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::publishMinChanged
    */
    void publishMinChanged(float Min) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::publishMaxChanged
    */
    void publishMaxChanged(float Max) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryTemperaturePublisher::publishCellTemperatureChanged
    */
    void publishCellTemperatureChanged(const std::list<float>& CellTemperature) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Temperature
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryTemperatureSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Average change.
    std::atomic<long> m_averageChangedCallbackNextId {0};
    // Subscribed callbacks for the Average change.
    std::map<long, VehiclePowertrainTractionBatteryTemperatureAveragePropertyCb> m_averageCallbacks;
    // Mutex for m_averageCallbacks
    mutable std::shared_timed_mutex m_averageCallbacksMutex;
    // Next free unique identifier to subscribe for the Min change.
    std::atomic<long> m_minChangedCallbackNextId {0};
    // Subscribed callbacks for the Min change.
    std::map<long, VehiclePowertrainTractionBatteryTemperatureMinPropertyCb> m_minCallbacks;
    // Mutex for m_minCallbacks
    mutable std::shared_timed_mutex m_minCallbacksMutex;
    // Next free unique identifier to subscribe for the Max change.
    std::atomic<long> m_maxChangedCallbackNextId {0};
    // Subscribed callbacks for the Max change.
    std::map<long, VehiclePowertrainTractionBatteryTemperatureMaxPropertyCb> m_maxCallbacks;
    // Mutex for m_maxCallbacks
    mutable std::shared_timed_mutex m_maxCallbacksMutex;
    // Next free unique identifier to subscribe for the CellTemperature change.
    std::atomic<long> m_cellTemperatureChangedCallbackNextId {0};
    // Subscribed callbacks for the CellTemperature change.
    std::map<long, VehiclePowertrainTractionBatteryTemperatureCellTemperaturePropertyCb> m_cellTemperatureCallbacks;
    // Mutex for m_cellTemperatureCallbacks
    mutable std::shared_timed_mutex m_cellTemperatureCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
