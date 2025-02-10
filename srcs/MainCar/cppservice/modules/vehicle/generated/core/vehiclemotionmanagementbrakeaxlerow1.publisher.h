#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementbrakeaxlerow1.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementBrakeAxleRow1Publisher.
 * Use this class to store clients of the Vehicle_MotionManagement_Brake_Axle_Row1 and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow1Publisher : public IVehicleMotionManagementBrakeAxleRow1Publisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow1Subscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow1Subscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueElectricMinimumChanged
    */
    long subscribeToTorqueElectricMinimumChanged(VehicleMotionManagementBrakeAxleRow1TorqueElectricMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueElectricMinimumChanged
    */
    void unsubscribeFromTorqueElectricMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueFrictionDifferenceMaximumChanged
    */
    long subscribeToTorqueFrictionDifferenceMaximumChanged(VehicleMotionManagementBrakeAxleRow1TorqueFrictionDifferenceMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueFrictionDifferenceMaximumChanged
    */
    void unsubscribeFromTorqueFrictionDifferenceMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueDistributionFrictionRightMaximumChanged
    */
    long subscribeToTorqueDistributionFrictionRightMaximumChanged(VehicleMotionManagementBrakeAxleRow1TorqueDistributionFrictionRightMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueDistributionFrictionRightMaximumChanged
    */
    void unsubscribeFromTorqueDistributionFrictionRightMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueDistributionFrictionRightMinimumChanged
    */
    long subscribeToTorqueDistributionFrictionRightMinimumChanged(VehicleMotionManagementBrakeAxleRow1TorqueDistributionFrictionRightMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueDistributionFrictionRightMinimumChanged
    */
    void unsubscribeFromTorqueDistributionFrictionRightMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::publishTorqueElectricMinimumChanged
    */
    void publishTorqueElectricMinimumChanged(int32_t TorqueElectricMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::publishTorqueFrictionDifferenceMaximumChanged
    */
    void publishTorqueFrictionDifferenceMaximumChanged(int32_t TorqueFrictionDifferenceMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::publishTorqueDistributionFrictionRightMaximumChanged
    */
    void publishTorqueDistributionFrictionRightMaximumChanged(int32_t TorqueDistributionFrictionRightMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1Publisher::publishTorqueDistributionFrictionRightMinimumChanged
    */
    void publishTorqueDistributionFrictionRightMinimumChanged(int32_t TorqueDistributionFrictionRightMinimum) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_Brake_Axle_Row1
    std::vector<std::reference_wrapper<IVehicleMotionManagementBrakeAxleRow1Subscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the TorqueElectricMinimum change.
    std::atomic<long> m_torqueElectricMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueElectricMinimum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1TorqueElectricMinimumPropertyCb> m_torqueElectricMinimumCallbacks;
    // Mutex for m_torqueElectricMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueElectricMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueFrictionDifferenceMaximum change.
    std::atomic<long> m_torqueFrictionDifferenceMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueFrictionDifferenceMaximum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1TorqueFrictionDifferenceMaximumPropertyCb> m_torqueFrictionDifferenceMaximumCallbacks;
    // Mutex for m_torqueFrictionDifferenceMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueFrictionDifferenceMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueDistributionFrictionRightMaximum change.
    std::atomic<long> m_torqueDistributionFrictionRightMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueDistributionFrictionRightMaximum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1TorqueDistributionFrictionRightMaximumPropertyCb> m_torqueDistributionFrictionRightMaximumCallbacks;
    // Mutex for m_torqueDistributionFrictionRightMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueDistributionFrictionRightMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueDistributionFrictionRightMinimum change.
    std::atomic<long> m_torqueDistributionFrictionRightMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueDistributionFrictionRightMinimum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1TorqueDistributionFrictionRightMinimumPropertyCb> m_torqueDistributionFrictionRightMinimumCallbacks;
    // Mutex for m_torqueDistributionFrictionRightMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueDistributionFrictionRightMinimumCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
