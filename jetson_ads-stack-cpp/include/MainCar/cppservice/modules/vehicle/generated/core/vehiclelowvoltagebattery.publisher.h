#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclelowvoltagebattery.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleLowVoltageBatteryPublisher.
 * Use this class to store clients of the Vehicle_LowVoltageBattery and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleLowVoltageBatteryPublisher : public IVehicleLowVoltageBatteryPublisher
{
public:
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleLowVoltageBatterySubscriber& subscriber) override;
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleLowVoltageBatterySubscriber& subscriber) override;

    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToNominalVoltageChanged
    */
    long subscribeToNominalVoltageChanged(VehicleLowVoltageBatteryNominalVoltagePropertyCb callback) override;
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToNominalVoltageChanged
    */
    void unsubscribeFromNominalVoltageChanged(long handleId) override;

    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToNominalCapacityChanged
    */
    long subscribeToNominalCapacityChanged(VehicleLowVoltageBatteryNominalCapacityPropertyCb callback) override;
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToNominalCapacityChanged
    */
    void unsubscribeFromNominalCapacityChanged(long handleId) override;

    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToCurrentVoltageChanged
    */
    long subscribeToCurrentVoltageChanged(VehicleLowVoltageBatteryCurrentVoltagePropertyCb callback) override;
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToCurrentVoltageChanged
    */
    void unsubscribeFromCurrentVoltageChanged(long handleId) override;

    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToCurrentCurrentChanged
    */
    long subscribeToCurrentCurrentChanged(VehicleLowVoltageBatteryCurrentCurrentPropertyCb callback) override;
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::subscribeToCurrentCurrentChanged
    */
    void unsubscribeFromCurrentCurrentChanged(long handleId) override;

    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::publishNominalVoltageChanged
    */
    void publishNominalVoltageChanged(int32_t NominalVoltage) const override;
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::publishNominalCapacityChanged
    */
    void publishNominalCapacityChanged(int32_t NominalCapacity) const override;
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::publishCurrentVoltageChanged
    */
    void publishCurrentVoltageChanged(float CurrentVoltage) const override;
    /**
    * Implementation of IVehicleLowVoltageBatteryPublisher::publishCurrentCurrentChanged
    */
    void publishCurrentCurrentChanged(float CurrentCurrent) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_LowVoltageBattery
    std::vector<std::reference_wrapper<IVehicleLowVoltageBatterySubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the NominalVoltage change.
    std::atomic<long> m_nominalVoltageChangedCallbackNextId {0};
    // Subscribed callbacks for the NominalVoltage change.
    std::map<long, VehicleLowVoltageBatteryNominalVoltagePropertyCb> m_nominalVoltageCallbacks;
    // Mutex for m_nominalVoltageCallbacks
    mutable std::shared_timed_mutex m_nominalVoltageCallbacksMutex;
    // Next free unique identifier to subscribe for the NominalCapacity change.
    std::atomic<long> m_nominalCapacityChangedCallbackNextId {0};
    // Subscribed callbacks for the NominalCapacity change.
    std::map<long, VehicleLowVoltageBatteryNominalCapacityPropertyCb> m_nominalCapacityCallbacks;
    // Mutex for m_nominalCapacityCallbacks
    mutable std::shared_timed_mutex m_nominalCapacityCallbacksMutex;
    // Next free unique identifier to subscribe for the CurrentVoltage change.
    std::atomic<long> m_currentVoltageChangedCallbackNextId {0};
    // Subscribed callbacks for the CurrentVoltage change.
    std::map<long, VehicleLowVoltageBatteryCurrentVoltagePropertyCb> m_currentVoltageCallbacks;
    // Mutex for m_currentVoltageCallbacks
    mutable std::shared_timed_mutex m_currentVoltageCallbacksMutex;
    // Next free unique identifier to subscribe for the CurrentCurrent change.
    std::atomic<long> m_currentCurrentChangedCallbackNextId {0};
    // Subscribed callbacks for the CurrentCurrent change.
    std::map<long, VehicleLowVoltageBatteryCurrentCurrentPropertyCb> m_currentCurrentCallbacks;
    // Mutex for m_currentCurrentCallbacks
    mutable std::shared_timed_mutex m_currentCurrentCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
