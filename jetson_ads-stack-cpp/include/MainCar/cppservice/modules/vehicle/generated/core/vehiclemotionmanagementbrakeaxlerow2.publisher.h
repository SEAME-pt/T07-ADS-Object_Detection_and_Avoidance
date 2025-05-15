#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementbrakeaxlerow2.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementBrakeAxleRow2Publisher.
 * Use this class to store clients of the Vehicle_MotionManagement_Brake_Axle_Row2 and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow2Publisher : public IVehicleMotionManagementBrakeAxleRow2Publisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow2Subscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow2Subscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueElectricMinimumChanged
    */
    long subscribeToTorqueElectricMinimumChanged(VehicleMotionManagementBrakeAxleRow2TorqueElectricMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueElectricMinimumChanged
    */
    void unsubscribeFromTorqueElectricMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueFrictionDifferenceMaximumChanged
    */
    long subscribeToTorqueFrictionDifferenceMaximumChanged(VehicleMotionManagementBrakeAxleRow2TorqueFrictionDifferenceMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueFrictionDifferenceMaximumChanged
    */
    void unsubscribeFromTorqueFrictionDifferenceMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueDistributionFrictionRightMaximumChanged
    */
    long subscribeToTorqueDistributionFrictionRightMaximumChanged(VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueDistributionFrictionRightMaximumChanged
    */
    void unsubscribeFromTorqueDistributionFrictionRightMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueDistributionFrictionRightMinimumChanged
    */
    long subscribeToTorqueDistributionFrictionRightMinimumChanged(VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueDistributionFrictionRightMinimumChanged
    */
    void unsubscribeFromTorqueDistributionFrictionRightMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::publishTorqueElectricMinimumChanged
    */
    void publishTorqueElectricMinimumChanged(int32_t TorqueElectricMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::publishTorqueFrictionDifferenceMaximumChanged
    */
    void publishTorqueFrictionDifferenceMaximumChanged(int32_t TorqueFrictionDifferenceMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::publishTorqueDistributionFrictionRightMaximumChanged
    */
    void publishTorqueDistributionFrictionRightMaximumChanged(int32_t TorqueDistributionFrictionRightMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2Publisher::publishTorqueDistributionFrictionRightMinimumChanged
    */
    void publishTorqueDistributionFrictionRightMinimumChanged(int32_t TorqueDistributionFrictionRightMinimum) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_Brake_Axle_Row2
    std::vector<std::reference_wrapper<IVehicleMotionManagementBrakeAxleRow2Subscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the TorqueElectricMinimum change.
    std::atomic<long> m_torqueElectricMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueElectricMinimum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2TorqueElectricMinimumPropertyCb> m_torqueElectricMinimumCallbacks;
    // Mutex for m_torqueElectricMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueElectricMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueFrictionDifferenceMaximum change.
    std::atomic<long> m_torqueFrictionDifferenceMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueFrictionDifferenceMaximum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2TorqueFrictionDifferenceMaximumPropertyCb> m_torqueFrictionDifferenceMaximumCallbacks;
    // Mutex for m_torqueFrictionDifferenceMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueFrictionDifferenceMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueDistributionFrictionRightMaximum change.
    std::atomic<long> m_torqueDistributionFrictionRightMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueDistributionFrictionRightMaximum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMaximumPropertyCb> m_torqueDistributionFrictionRightMaximumCallbacks;
    // Mutex for m_torqueDistributionFrictionRightMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueDistributionFrictionRightMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueDistributionFrictionRightMinimum change.
    std::atomic<long> m_torqueDistributionFrictionRightMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueDistributionFrictionRightMinimum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMinimumPropertyCb> m_torqueDistributionFrictionRightMinimumCallbacks;
    // Mutex for m_torqueDistributionFrictionRightMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueDistributionFrictionRightMinimumCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
