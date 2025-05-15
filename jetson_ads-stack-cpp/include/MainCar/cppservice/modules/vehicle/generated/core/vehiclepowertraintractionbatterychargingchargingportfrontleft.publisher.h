#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterychargingchargingportfrontleft.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher : public IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToSupportedInletTypesChanged
    */
    long subscribeToSupportedInletTypesChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSupportedInletTypesPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToSupportedInletTypesChanged
    */
    void unsubscribeFromSupportedInletTypesChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToIsFlapOpenChanged
    */
    long subscribeToIsFlapOpenChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsFlapOpenPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToIsFlapOpenChanged
    */
    void unsubscribeFromIsFlapOpenChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToIsChargingCableConnectedChanged
    */
    long subscribeToIsChargingCableConnectedChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableConnectedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToIsChargingCableConnectedChanged
    */
    void unsubscribeFromIsChargingCableConnectedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToIsChargingCableLockedChanged
    */
    long subscribeToIsChargingCableLockedChanged(VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableLockedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::subscribeToIsChargingCableLockedChanged
    */
    void unsubscribeFromIsChargingCableLockedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::publishSupportedInletTypesChanged
    */
    void publishSupportedInletTypesChanged(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum SupportedInletTypes) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::publishIsFlapOpenChanged
    */
    void publishIsFlapOpenChanged(bool IsFlapOpen) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::publishIsChargingCableConnectedChanged
    */
    void publishIsChargingCableConnectedChanged(bool IsChargingCableConnected) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftPublisher::publishIsChargingCableLockedChanged
    */
    void publishIsChargingCableLockedChanged(bool IsChargingCableLocked) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the SupportedInletTypes change.
    std::atomic<long> m_supportedInletTypesChangedCallbackNextId {0};
    // Subscribed callbacks for the SupportedInletTypes change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSupportedInletTypesPropertyCb> m_supportedInletTypesCallbacks;
    // Mutex for m_supportedInletTypesCallbacks
    mutable std::shared_timed_mutex m_supportedInletTypesCallbacksMutex;
    // Next free unique identifier to subscribe for the IsFlapOpen change.
    std::atomic<long> m_isFlapOpenChangedCallbackNextId {0};
    // Subscribed callbacks for the IsFlapOpen change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsFlapOpenPropertyCb> m_isFlapOpenCallbacks;
    // Mutex for m_isFlapOpenCallbacks
    mutable std::shared_timed_mutex m_isFlapOpenCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableConnected change.
    std::atomic<long> m_isChargingCableConnectedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableConnected change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableConnectedPropertyCb> m_isChargingCableConnectedCallbacks;
    // Mutex for m_isChargingCableConnectedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableConnectedCallbacksMutex;
    // Next free unique identifier to subscribe for the IsChargingCableLocked change.
    std::atomic<long> m_isChargingCableLockedChangedCallbackNextId {0};
    // Subscribed callbacks for the IsChargingCableLocked change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargingPortFrontLeftIsChargingCableLockedPropertyCb> m_isChargingCableLockedCallbacks;
    // Mutex for m_isChargingCableLockedCallbacks
    mutable std::shared_timed_mutex m_isChargingCableLockedCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
