#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterystateofcharge.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryStateOfChargePublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_StateOfCharge and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryStateOfChargePublisher : public IVehiclePowertrainTractionBatteryStateOfChargePublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryStateOfChargeSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryStateOfChargeSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToCurrentChanged
    */
    long subscribeToCurrentChanged(VehiclePowertrainTractionBatteryStateOfChargeCurrentPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToCurrentChanged
    */
    void unsubscribeFromCurrentChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToDisplayedChanged
    */
    long subscribeToDisplayedChanged(VehiclePowertrainTractionBatteryStateOfChargeDisplayedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToDisplayedChanged
    */
    void unsubscribeFromDisplayedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToCurrentEnergyChanged
    */
    long subscribeToCurrentEnergyChanged(VehiclePowertrainTractionBatteryStateOfChargeCurrentEnergyPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToCurrentEnergyChanged
    */
    void unsubscribeFromCurrentEnergyChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::publishCurrentChanged
    */
    void publishCurrentChanged(float Current) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::publishDisplayedChanged
    */
    void publishDisplayedChanged(float Displayed) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryStateOfChargePublisher::publishCurrentEnergyChanged
    */
    void publishCurrentEnergyChanged(float CurrentEnergy) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_StateOfCharge
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryStateOfChargeSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Current change.
    std::atomic<long> m_currentChangedCallbackNextId {0};
    // Subscribed callbacks for the Current change.
    std::map<long, VehiclePowertrainTractionBatteryStateOfChargeCurrentPropertyCb> m_currentCallbacks;
    // Mutex for m_currentCallbacks
    mutable std::shared_timed_mutex m_currentCallbacksMutex;
    // Next free unique identifier to subscribe for the Displayed change.
    std::atomic<long> m_displayedChangedCallbackNextId {0};
    // Subscribed callbacks for the Displayed change.
    std::map<long, VehiclePowertrainTractionBatteryStateOfChargeDisplayedPropertyCb> m_displayedCallbacks;
    // Mutex for m_displayedCallbacks
    mutable std::shared_timed_mutex m_displayedCallbacksMutex;
    // Next free unique identifier to subscribe for the CurrentEnergy change.
    std::atomic<long> m_currentEnergyChangedCallbackNextId {0};
    // Subscribed callbacks for the CurrentEnergy change.
    std::map<long, VehiclePowertrainTractionBatteryStateOfChargeCurrentEnergyPropertyCb> m_currentEnergyCallbacks;
    // Mutex for m_currentEnergyCallbacks
    mutable std::shared_timed_mutex m_currentEnergyCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
