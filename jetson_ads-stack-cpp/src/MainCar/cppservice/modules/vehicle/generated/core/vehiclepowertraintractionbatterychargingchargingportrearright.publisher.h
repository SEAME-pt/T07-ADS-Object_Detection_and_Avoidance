#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterychargingchargingportrearright.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher : public IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortRearRightSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortRearRightSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToSupportedInletTypesChanged
    */
    long subscribeToSupportedInletTypesChanged(VehiclePowertrainTractionBatteryChargingChargingPortRearRightSupportedInletTypesPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToSupportedInletTypesChanged
    */
    void unsubscribeFromSupportedInletTypesChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToIsFlapOpenChanged
    */
    long subscribeToIsFlapOpenChanged(VehiclePowertrainTractionBatteryChargingChargingPortRearRightIsFlapOpenPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToIsFlapOpenChanged
    */
    void unsubscribeFromIsFlapOpenChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToIsChargingCableConnectedChanged
    */
    long subscribeToIsChargingCableConnectedChanged(VehiclePowertrainTractionBatteryChargingChargingPortRearRightIsChargingCableConnectedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToIsChargingCableConnectedChanged
    */
    void unsubscribeFromIsChargingCableConnectedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToIsChargingCableLockedChanged
    */
    long subscribeToIsChargingCableLockedChanged(VehiclePowertrainTractionBatteryChargingChargingPortRearRightIsChargingCableLockedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::subscribeToIsChargingCableLockedChanged
    */
    void unsubscribeFromIsChargingCableLockedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::publishSupportedInletTypesChanged
    */
    void publishSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum SupportedInletTypes) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::publishIsFlapOpenChanged
    */
    void publishIsFlapOpenChanged(bool IsFlapOpen) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::publishIsChargingCableConnectedChanged
    */
    void publishIsChargingCableConnectedChanged(bool IsChargingCableConnected) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher::publishIsChargingCableLockedChanged
    */
    void publishIsChargingCableLockedChanged(bool IsChargingCableLocked) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingChargingPortRearRightSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the SupportedInletTypes change.
    std::atomic<long> m_supportedInletTypesChangedCallbackNextId {0};
    // Subscribed callbacks for the SupportedInletTypes change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortRearRightSupportedInletTypesPropertyCb> m_supportedInletTypesCallbacks;
    // Mutex for m_supportedInletTypesCallbacks
    mutable std::shared_timed_mutex m_supportedInletTypesCallbacksMutex;
    // Next free unique identifier to subscribe for the IsFlapOpen change.
    std::atomic<long> m_isFlapOpenChangedCallbackNextId {0};
    // Subscribed callbacks for the IsFlapOpen change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortRearRightIsFlapOpenPropertyCb> m_isFlapOpenCallbacks;
    // Mutex for m_isFlapOpenCallbacks
    mutable std::shared_timed_mutex m_isFlapOpenCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableConnected change.
    std::atomic<long> m_isChargingCableConnectedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableConnected change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortRearRightIsChargingCableConnectedPropertyCb> m_isChargingCableConnectedCallbacks;
    // Mutex for m_isChargingCableConnectedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableConnectedCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableLocked change.
    std::atomic<long> m_isChargingCableLockedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableLocked change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortRearRightIsChargingCableLockedPropertyCb> m_isChargingCableLockedCallbacks;
    // Mutex for m_isChargingCableLockedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableLockedCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
