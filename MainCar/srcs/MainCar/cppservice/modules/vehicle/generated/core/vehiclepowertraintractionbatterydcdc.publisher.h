#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterydcdc.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryDcdcPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_DCDC and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryDcdcPublisher : public IVehiclePowertrainTractionBatteryDcdcPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryDcdcPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryDcdcSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryDcdcPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryDcdcSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryDcdcPublisher::subscribeToPowerLossChanged
    */
    long subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryDcdcPowerLossPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryDcdcPublisher::subscribeToPowerLossChanged
    */
    void unsubscribeFromPowerLossChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryDcdcPublisher::subscribeToTemperatureChanged
    */
    long subscribeToTemperatureChanged(VehiclePowertrainTractionBatteryDcdcTemperaturePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryDcdcPublisher::subscribeToTemperatureChanged
    */
    void unsubscribeFromTemperatureChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryDcdcPublisher::publishPowerLossChanged
    */
    void publishPowerLossChanged(float PowerLoss) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryDcdcPublisher::publishTemperatureChanged
    */
    void publishTemperatureChanged(float Temperature) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_DCDC
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryDcdcSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the PowerLoss change.
    std::atomic<long> m_powerLossChangedCallbackNextId {0};
    // Subscribed callbacks for the PowerLoss change.
    std::map<long, VehiclePowertrainTractionBatteryDcdcPowerLossPropertyCb> m_powerLossCallbacks;
    // Mutex for m_powerLossCallbacks
    mutable std::shared_timed_mutex m_powerLossCallbacksMutex;
    // Next free unique identifier to subscribe for the Temperature change.
    std::atomic<long> m_temperatureChangedCallbackNextId {0};
    // Subscribed callbacks for the Temperature change.
    std::map<long, VehiclePowertrainTractionBatteryDcdcTemperaturePropertyCb> m_temperatureCallbacks;
    // Mutex for m_temperatureCallbacks
    mutable std::shared_timed_mutex m_temperatureCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
