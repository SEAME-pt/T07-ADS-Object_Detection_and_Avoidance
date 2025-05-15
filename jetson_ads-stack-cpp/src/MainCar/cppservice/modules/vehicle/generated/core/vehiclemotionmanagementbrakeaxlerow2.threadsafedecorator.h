
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_MotionManagement_Brake_Axle_Row2 interface.
* @see Vehicle_MotionManagement_Brake_Axle_Row2
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleMotionManagementBrakeAxleRow2> testVehicleMotionManagementBrakeAxleRow2 = std::make_unique<VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator>(std::make_shared<VehicleMotionManagementBrakeAxleRow2>());

// Thread safe access
auto torqueElectricMinimum = testVehicleMotionManagementBrakeAxleRow2->getTorqueElectricMinimum();
testVehicleMotionManagementBrakeAxleRow2->setTorqueElectricMinimum(0);
auto torqueFrictionDifferenceMaximum = testVehicleMotionManagementBrakeAxleRow2->getTorqueFrictionDifferenceMaximum();
testVehicleMotionManagementBrakeAxleRow2->setTorqueFrictionDifferenceMaximum(0);
auto torqueDistributionFrictionRightMaximum = testVehicleMotionManagementBrakeAxleRow2->getTorqueDistributionFrictionRightMaximum();
testVehicleMotionManagementBrakeAxleRow2->setTorqueDistributionFrictionRightMaximum(0);
auto torqueDistributionFrictionRightMinimum = testVehicleMotionManagementBrakeAxleRow2->getTorqueDistributionFrictionRightMinimum();
testVehicleMotionManagementBrakeAxleRow2->setTorqueDistributionFrictionRightMinimum(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator : public IVehicleMotionManagementBrakeAxleRow2
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_MotionManagement_Brake_Axle_Row2 object to make thread safe.
    */
    explicit VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow2> impl);

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2 implementation. */
    void setTorqueElectricMinimum(int32_t TorqueElectricMinimum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2 implementation. */
    int32_t getTorqueElectricMinimum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2 implementation. */
    void setTorqueFrictionDifferenceMaximum(int32_t TorqueFrictionDifferenceMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2 implementation. */
    int32_t getTorqueFrictionDifferenceMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2 implementation. */
    void setTorqueDistributionFrictionRightMaximum(int32_t TorqueDistributionFrictionRightMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2 implementation. */
    int32_t getTorqueDistributionFrictionRightMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2 implementation. */
    void setTorqueDistributionFrictionRightMinimum(int32_t TorqueDistributionFrictionRightMinimum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2 implementation. */
    int32_t getTorqueDistributionFrictionRightMinimum() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake_Axle_Row2 changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_MotionManagement_Brake_Axle_Row2.
    */
    IVehicleMotionManagementBrakeAxleRow2Publisher& _getPublisher() const override;
private:
    /** The Vehicle_MotionManagement_Brake_Axle_Row2 object which is guarded */
    std::shared_ptr<IVehicleMotionManagementBrakeAxleRow2> m_impl;
    // Mutex for torqueElectricMinimum property
    mutable std::shared_timed_mutex m_torqueElectricMinimumMutex;
    // Mutex for torqueFrictionDifferenceMaximum property
    mutable std::shared_timed_mutex m_torqueFrictionDifferenceMaximumMutex;
    // Mutex for torqueDistributionFrictionRightMaximum property
    mutable std::shared_timed_mutex m_torqueDistributionFrictionRightMaximumMutex;
    // Mutex for torqueDistributionFrictionRightMinimum property
    mutable std::shared_timed_mutex m_torqueDistributionFrictionRightMinimumMutex;
};
} // namespace Vehicle
} // namespace Cpp
