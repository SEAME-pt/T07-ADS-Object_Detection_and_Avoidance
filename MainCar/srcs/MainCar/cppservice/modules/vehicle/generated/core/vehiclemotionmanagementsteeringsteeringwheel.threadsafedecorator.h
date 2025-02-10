
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_MotionManagement_Steering_SteeringWheel interface.
* @see Vehicle_MotionManagement_Steering_SteeringWheel
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleMotionManagementSteeringSteeringWheel> testVehicleMotionManagementSteeringSteeringWheel = std::make_unique<VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator>(std::make_shared<VehicleMotionManagementSteeringSteeringWheel>());

// Thread safe access
auto angle = testVehicleMotionManagementSteeringSteeringWheel->getAngle();
testVehicleMotionManagementSteeringSteeringWheel->setAngle(0);
auto angleTarget = testVehicleMotionManagementSteeringSteeringWheel->getAngleTarget();
testVehicleMotionManagementSteeringSteeringWheel->setAngleTarget(0);
auto angleTargetMode = testVehicleMotionManagementSteeringSteeringWheel->getAngleTargetMode();
testVehicleMotionManagementSteeringSteeringWheel->setAngleTargetMode(0);
auto torqueOffsetTarget = testVehicleMotionManagementSteeringSteeringWheel->getTorqueOffsetTarget();
testVehicleMotionManagementSteeringSteeringWheel->setTorqueOffsetTarget(0);
auto torqueOffsetTargetMode = testVehicleMotionManagementSteeringSteeringWheel->getTorqueOffsetTargetMode();
testVehicleMotionManagementSteeringSteeringWheel->setTorqueOffsetTargetMode(0);
auto torqueTarget = testVehicleMotionManagementSteeringSteeringWheel->getTorqueTarget();
testVehicleMotionManagementSteeringSteeringWheel->setTorqueTarget(0);
auto torqueTargetMode = testVehicleMotionManagementSteeringSteeringWheel->getTorqueTargetMode();
testVehicleMotionManagementSteeringSteeringWheel->setTorqueTargetMode(0);
auto torque = testVehicleMotionManagementSteeringSteeringWheel->getTorque();
testVehicleMotionManagementSteeringSteeringWheel->setTorque(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator : public IVehicleMotionManagementSteeringSteeringWheel
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_MotionManagement_Steering_SteeringWheel object to make thread safe.
    */
    explicit VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementSteeringSteeringWheel> impl);

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    void setAngle(int32_t Angle) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    int32_t getAngle() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    void setAngleTarget(int32_t AngleTarget) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    int32_t getAngleTarget() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    void setAngleTargetMode(int32_t AngleTargetMode) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    int32_t getAngleTargetMode() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    void setTorqueOffsetTarget(int32_t TorqueOffsetTarget) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    int32_t getTorqueOffsetTarget() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    void setTorqueOffsetTargetMode(int32_t TorqueOffsetTargetMode) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    int32_t getTorqueOffsetTargetMode() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    void setTorqueTarget(int32_t TorqueTarget) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    int32_t getTorqueTarget() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    void setTorqueTargetMode(int32_t TorqueTargetMode) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    int32_t getTorqueTargetMode() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    void setTorque(int32_t Torque) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Steering_SteeringWheel implementation. */
    int32_t getTorque() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Steering_SteeringWheel changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_MotionManagement_Steering_SteeringWheel.
    */
    IVehicleMotionManagementSteeringSteeringWheelPublisher& _getPublisher() const override;
private:
    /** The Vehicle_MotionManagement_Steering_SteeringWheel object which is guarded */
    std::shared_ptr<IVehicleMotionManagementSteeringSteeringWheel> m_impl;
    // Mutex for angle property
    mutable std::shared_timed_mutex m_angleMutex;
    // Mutex for angleTarget property
    mutable std::shared_timed_mutex m_angleTargetMutex;
    // Mutex for angleTargetMode property
    mutable std::shared_timed_mutex m_angleTargetModeMutex;
    // Mutex for torqueOffsetTarget property
    mutable std::shared_timed_mutex m_torqueOffsetTargetMutex;
    // Mutex for torqueOffsetTargetMode property
    mutable std::shared_timed_mutex m_torqueOffsetTargetModeMutex;
    // Mutex for torqueTarget property
    mutable std::shared_timed_mutex m_torqueTargetMutex;
    // Mutex for torqueTargetMode property
    mutable std::shared_timed_mutex m_torqueTargetModeMutex;
    // Mutex for torque property
    mutable std::shared_timed_mutex m_torqueMutex;
};
} // namespace Vehicle
} // namespace Cpp
