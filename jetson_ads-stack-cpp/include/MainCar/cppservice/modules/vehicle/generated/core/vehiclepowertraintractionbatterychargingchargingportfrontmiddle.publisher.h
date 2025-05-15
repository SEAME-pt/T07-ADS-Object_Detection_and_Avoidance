#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterychargingchargingportfrontmiddle.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher : public IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToSupportedInletTypesChanged
    */
    long subscribeToSupportedInletTypesChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleSupportedInletTypesPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToSupportedInletTypesChanged
    */
    void unsubscribeFromSupportedInletTypesChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToIsFlapOpenChanged
    */
    long subscribeToIsFlapOpenChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleIsFlapOpenPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToIsFlapOpenChanged
    */
    void unsubscribeFromIsFlapOpenChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToIsChargingCableConnectedChanged
    */
    long subscribeToIsChargingCableConnectedChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleIsChargingCableConnectedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToIsChargingCableConnectedChanged
    */
    void unsubscribeFromIsChargingCableConnectedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToIsChargingCableLockedChanged
    */
    long subscribeToIsChargingCableLockedChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleIsChargingCableLockedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::subscribeToIsChargingCableLockedChanged
    */
    void unsubscribeFromIsChargingCableLockedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::publishSupportedInletTypesChanged
    */
    void publishSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum SupportedInletTypes) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::publishIsFlapOpenChanged
    */
    void publishIsFlapOpenChanged(bool IsFlapOpen) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::publishIsChargingCableConnectedChanged
    */
    void publishIsChargingCableConnectedChanged(bool IsChargingCableConnected) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher::publishIsChargingCableLockedChanged
    */
    void publishIsChargingCableLockedChanged(bool IsChargingCableLocked) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the SupportedInletTypes change.
    std::atomic<long> m_supportedInletTypesChangedCallbackNextId {0};
    // Subscribed callbacks for the SupportedInletTypes change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleSupportedInletTypesPropertyCb> m_supportedInletTypesCallbacks;
    // Mutex for m_supportedInletTypesCallbacks
    mutable std::shared_timed_mutex m_supportedInletTypesCallbacksMutex;
    // Next free unique identifier to subscribe for the IsFlapOpen change.
    std::atomic<long> m_isFlapOpenChangedCallbackNextId {0};
    // Subscribed callbacks for the IsFlapOpen change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleIsFlapOpenPropertyCb> m_isFlapOpenCallbacks;
    // Mutex for m_isFlapOpenCallbacks
    mutable std::shared_timed_mutex m_isFlapOpenCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableConnected change.
    std::atomic<long> m_isChargingCableConnectedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableConnected change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleIsChargingCableConnectedPropertyCb> m_isChargingCableConnectedCallbacks;
    // Mutex for m_isChargingCableConnectedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableConnectedCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableLocked change.
    std::atomic<long> m_isChargingCableLockedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableLocked change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleIsChargingCableLockedPropertyCb> m_isChargingCableLockedCallbacks;
    // Mutex for m_isChargingCableLockedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableLockedCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
