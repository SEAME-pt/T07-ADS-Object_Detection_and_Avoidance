#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterychargingchargingportrearmiddle.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher : public IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddleSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddleSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToSupportedInletTypesChanged
    */
    long subscribeToSupportedInletTypesChanged(VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleSupportedInletTypesPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToSupportedInletTypesChanged
    */
    void unsubscribeFromSupportedInletTypesChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToIsFlapOpenChanged
    */
    long subscribeToIsFlapOpenChanged(VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleIsFlapOpenPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToIsFlapOpenChanged
    */
    void unsubscribeFromIsFlapOpenChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToIsChargingCableConnectedChanged
    */
    long subscribeToIsChargingCableConnectedChanged(VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleIsChargingCableConnectedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToIsChargingCableConnectedChanged
    */
    void unsubscribeFromIsChargingCableConnectedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToIsChargingCableLockedChanged
    */
    long subscribeToIsChargingCableLockedChanged(VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleIsChargingCableLockedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::subscribeToIsChargingCableLockedChanged
    */
    void unsubscribeFromIsChargingCableLockedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::publishSupportedInletTypesChanged
    */
    void publishSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum SupportedInletTypes) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::publishIsFlapOpenChanged
    */
    void publishIsFlapOpenChanged(bool IsFlapOpen) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::publishIsChargingCableConnectedChanged
    */
    void publishIsChargingCableConnectedChanged(bool IsChargingCableConnected) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher::publishIsChargingCableLockedChanged
    */
    void publishIsChargingCableLockedChanged(bool IsChargingCableLocked) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddleSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the SupportedInletTypes change.
    std::atomic<long> m_supportedInletTypesChangedCallbackNextId {0};
    // Subscribed callbacks for the SupportedInletTypes change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleSupportedInletTypesPropertyCb> m_supportedInletTypesCallbacks;
    // Mutex for m_supportedInletTypesCallbacks
    mutable std::shared_timed_mutex m_supportedInletTypesCallbacksMutex;
    // Next free unique identifier to subscribe for the IsFlapOpen change.
    std::atomic<long> m_isFlapOpenChangedCallbackNextId {0};
    // Subscribed callbacks for the IsFlapOpen change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleIsFlapOpenPropertyCb> m_isFlapOpenCallbacks;
    // Mutex for m_isFlapOpenCallbacks
    mutable std::shared_timed_mutex m_isFlapOpenCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableConnected change.
    std::atomic<long> m_isChargingCableConnectedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableConnected change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleIsChargingCableConnectedPropertyCb> m_isChargingCableConnectedCallbacks;
    // Mutex for m_isChargingCableConnectedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableConnectedCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableLocked change.
    std::atomic<long> m_isChargingCableLockedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableLocked change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleIsChargingCableLockedPropertyCb> m_isChargingCableLockedCallbacks;
    // Mutex for m_isChargingCableLockedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableLockedCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
