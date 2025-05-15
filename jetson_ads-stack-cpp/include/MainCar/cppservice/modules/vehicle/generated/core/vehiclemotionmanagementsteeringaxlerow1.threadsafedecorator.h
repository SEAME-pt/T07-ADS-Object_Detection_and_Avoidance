
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_MotionManagement_Steering_Axle_Row1 interface.
* @see Vehicle_MotionManagement_Steering_Axle_Row1
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleMotionManagementSteeringAxleRow1> testVehicleMotionManagementSteeringAxleRow1 = std::make_unique<VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator>(std::make_shared<VehicleMotionManagementSteeringAxleRow1>());

// Thread safe access
auto rackPositionOffsetTarget = testVehicleMotionManagementSteeringAxleRow1->getRackPositionOffsetTarget();
testVehicleMotionManagementSteeringAxleRow1->setRackPositionOffsetTarget(0);
auto rackPositionOffsetTargetMode = testVehicleMotionManagementSteeringAxleRow1->getRackPositionOffsetTargetMode();
testVehicleMotionManagementSteeringAxleRow1->setRackPositionOffsetTargetMode(0);
auto rackPosition = testVehicleMotionManagementSteeringAxleRow1->getRackPosition();
testVehicleMotionManagementSteeringAxleRow1->setRackPosition(0);
auto rackPositionTarget = testVehicleMotionManagementSteeringAxleRow1->getRackPositionTarget();
testVehicleMotionManagementSteeringAxleRow1->setRackPositionTarget(0);
auto rackPositionTargetMode = testVehicleMotionManagementSteeringAxleRow1->getRackPositionTargetMode();
testVehicleMotionManagementSteeringAxleRow1->setRackPositionTargetMode(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator : public IVehicleMotionManagementSteeringAxleRow1
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_MotionManagement_Steering_Axle_Row1 object to make thread safe.
    */
    explicit VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementSteeringAxleRow1> impl);

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    void setRackPositionOffsetTarget(int32_t RackPositionOffsetTarget) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    int32_t getRackPositionOffsetTarget() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    void setRackPositionOffsetTargetMode(int32_t RackPositionOffsetTargetMode) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    int32_t getRackPositionOffsetTargetMode() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    void setRackPosition(int32_t RackPosition) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    int32_t getRackPosition() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    void setRackPositionTarget(int32_t RackPositionTarget) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    int32_t getRackPositionTarget() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    void setRackPositionTargetMode(int32_t RackPositionTargetMode) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_Axle_Row1 implementation. */
    int32_t getRackPositionTargetMode() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Steering_Axle_Row1 changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_MotionManagement_Steering_Axle_Row1.
    */
    IVehicleMotionManagementSteeringAxleRow1Publisher& _getPublisher() const override;
private:
    /** The Vehicle_MotionManagement_Steering_Axle_Row1 object which is guarded */
    std::shared_ptr<IVehicleMotionManagementSteeringAxleRow1> m_impl;
    // Mutex for rackPositionOffsetTarget property
    mutable std::shared_timed_mutex m_rackPositionOffsetTargetMutex;
    // Mutex for rackPositionOffsetTargetMode property
    mutable std::shared_timed_mutex m_rackPositionOffsetTargetModeMutex;
    // Mutex for rackPosition property
    mutable std::shared_timed_mutex m_rackPositionMutex;
    // Mutex for rackPositionTarget property
    mutable std::shared_timed_mutex m_rackPositionTargetMutex;
    // Mutex for rackPositionTargetMode property
    mutable std::shared_timed_mutex m_rackPositionTargetModeMutex;
};
} // namespace Vehicle
} // namespace Cpp
