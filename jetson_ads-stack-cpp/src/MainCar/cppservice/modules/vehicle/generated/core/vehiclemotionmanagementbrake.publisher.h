#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementbrake.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementBrakePublisher.
 * Use this class to store clients of the Vehicle_MotionManagement_Brake and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakePublisher : public IVehicleMotionManagementBrakePublisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementBrakeSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementBrakeSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceMaximumChanged
    */
    long subscribeToVehicleForceMaximumChanged(VehicleMotionManagementBrakeVehicleForceMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceMaximumChanged
    */
    void unsubscribeFromVehicleForceMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceElectricMinimumArbitratedChanged
    */
    long subscribeToVehicleForceElectricMinimumArbitratedChanged(VehicleMotionManagementBrakeVehicleForceElectricMinimumArbitratedPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceElectricMinimumArbitratedChanged
    */
    void unsubscribeFromVehicleForceElectricMinimumArbitratedChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceElectricChanged
    */
    long subscribeToVehicleForceElectricChanged(VehicleMotionManagementBrakeVehicleForceElectricPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceElectricChanged
    */
    void unsubscribeFromVehicleForceElectricChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceDistributionFrontMaximumChanged
    */
    long subscribeToVehicleForceDistributionFrontMaximumChanged(VehicleMotionManagementBrakeVehicleForceDistributionFrontMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceDistributionFrontMaximumChanged
    */
    void unsubscribeFromVehicleForceDistributionFrontMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceDistributionFrontMinimumChanged
    */
    long subscribeToVehicleForceDistributionFrontMinimumChanged(VehicleMotionManagementBrakeVehicleForceDistributionFrontMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::subscribeToVehicleForceDistributionFrontMinimumChanged
    */
    void unsubscribeFromVehicleForceDistributionFrontMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::publishVehicleForceMaximumChanged
    */
    void publishVehicleForceMaximumChanged(int32_t VehicleForceMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::publishVehicleForceElectricMinimumArbitratedChanged
    */
    void publishVehicleForceElectricMinimumArbitratedChanged(int32_t VehicleForceElectricMinimumArbitrated) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::publishVehicleForceElectricChanged
    */
    void publishVehicleForceElectricChanged(int32_t VehicleForceElectric) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::publishVehicleForceDistributionFrontMaximumChanged
    */
    void publishVehicleForceDistributionFrontMaximumChanged(int32_t VehicleForceDistributionFrontMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakePublisher::publishVehicleForceDistributionFrontMinimumChanged
    */
    void publishVehicleForceDistributionFrontMinimumChanged(int32_t VehicleForceDistributionFrontMinimum) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_Brake
    std::vector<std::reference_wrapper<IVehicleMotionManagementBrakeSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the VehicleForceMaximum change.
    std::atomic<long> m_vehicleForceMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleForceMaximum change.
    std::map<long, VehicleMotionManagementBrakeVehicleForceMaximumPropertyCb> m_vehicleForceMaximumCallbacks;
    // Mutex for m_vehicleForceMaximumCallbacks
    mutable std::shared_timed_mutex m_vehicleForceMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleForceElectricMinimumArbitrated change.
    std::atomic<long> m_vehicleForceElectricMinimumArbitratedChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleForceElectricMinimumArbitrated change.
    std::map<long, VehicleMotionManagementBrakeVehicleForceElectricMinimumArbitratedPropertyCb> m_vehicleForceElectricMinimumArbitratedCallbacks;
    // Mutex for m_vehicleForceElectricMinimumArbitratedCallbacks
    mutable std::shared_timed_mutex m_vehicleForceElectricMinimumArbitratedCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleForceElectric change.
    std::atomic<long> m_vehicleForceElectricChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleForceElectric change.
    std::map<long, VehicleMotionManagementBrakeVehicleForceElectricPropertyCb> m_vehicleForceElectricCallbacks;
    // Mutex for m_vehicleForceElectricCallbacks
    mutable std::shared_timed_mutex m_vehicleForceElectricCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleForceDistributionFrontMaximum change.
    std::atomic<long> m_vehicleForceDistributionFrontMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleForceDistributionFrontMaximum change.
    std::map<long, VehicleMotionManagementBrakeVehicleForceDistributionFrontMaximumPropertyCb> m_vehicleForceDistributionFrontMaximumCallbacks;
    // Mutex for m_vehicleForceDistributionFrontMaximumCallbacks
    mutable std::shared_timed_mutex m_vehicleForceDistributionFrontMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleForceDistributionFrontMinimum change.
    std::atomic<long> m_vehicleForceDistributionFrontMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleForceDistributionFrontMinimum change.
    std::map<long, VehicleMotionManagementBrakeVehicleForceDistributionFrontMinimumPropertyCb> m_vehicleForceDistributionFrontMinimumCallbacks;
    // Mutex for m_vehicleForceDistributionFrontMinimumCallbacks
    mutable std::shared_timed_mutex m_vehicleForceDistributionFrontMinimumCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
