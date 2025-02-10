#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterychargingchargingportanyposition.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher : public IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToSupportedInletTypesChanged
    */
    long subscribeToSupportedInletTypesChanged(VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionSupportedInletTypesPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToSupportedInletTypesChanged
    */
    void unsubscribeFromSupportedInletTypesChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToIsFlapOpenChanged
    */
    long subscribeToIsFlapOpenChanged(VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionIsFlapOpenPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToIsFlapOpenChanged
    */
    void unsubscribeFromIsFlapOpenChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToIsChargingCableConnectedChanged
    */
    long subscribeToIsChargingCableConnectedChanged(VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionIsChargingCableConnectedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToIsChargingCableConnectedChanged
    */
    void unsubscribeFromIsChargingCableConnectedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToIsChargingCableLockedChanged
    */
    long subscribeToIsChargingCableLockedChanged(VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionIsChargingCableLockedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::subscribeToIsChargingCableLockedChanged
    */
    void unsubscribeFromIsChargingCableLockedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::publishSupportedInletTypesChanged
    */
    void publishSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum SupportedInletTypes) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::publishIsFlapOpenChanged
    */
    void publishIsFlapOpenChanged(bool IsFlapOpen) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::publishIsChargingCableConnectedChanged
    */
    void publishIsChargingCableConnectedChanged(bool IsChargingCableConnected) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher::publishIsChargingCableLockedChanged
    */
    void publishIsChargingCableLockedChanged(bool IsChargingCableLocked) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the SupportedInletTypes change.
    std::atomic<long> m_supportedInletTypesChangedCallbackNextId {0};
    // Subscribed callbacks for the SupportedInletTypes change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionSupportedInletTypesPropertyCb> m_supportedInletTypesCallbacks;
    // Mutex for m_supportedInletTypesCallbacks
    mutable std::shared_timed_mutex m_supportedInletTypesCallbacksMutex;
    // Next free unique identifier to subscribe for the IsFlapOpen change.
    std::atomic<long> m_isFlapOpenChangedCallbackNextId {0};
    // Subscribed callbacks for the IsFlapOpen change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionIsFlapOpenPropertyCb> m_isFlapOpenCallbacks;
    // Mutex for m_isFlapOpenCallbacks
    mutable std::shared_timed_mutex m_isFlapOpenCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableConnected change.
    std::atomic<long> m_isChargingCableConnectedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableConnected change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionIsChargingCableConnectedPropertyCb> m_isChargingCableConnectedCallbacks;
    // Mutex for m_isChargingCableConnectedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableConnectedCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableLocked change.
    std::atomic<long> m_isChargingCableLockedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableLocked change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionIsChargingCableLockedPropertyCb> m_isChargingCableLockedCallbacks;
    // Mutex for m_isChargingCableLockedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableLockedCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
