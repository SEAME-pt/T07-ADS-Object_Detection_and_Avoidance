
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right interface.
* @see Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleMotionManagementBrakeAxleRow2WheelRight> testVehicleMotionManagementBrakeAxleRow2WheelRight = std::make_unique<VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator>(std::make_shared<VehicleMotionManagementBrakeAxleRow2WheelRight>());

// Thread safe access
auto torque = testVehicleMotionManagementBrakeAxleRow2WheelRight->getTorque();
testVehicleMotionManagementBrakeAxleRow2WheelRight->setTorque(0);
auto torqueArbitrated = testVehicleMotionManagementBrakeAxleRow2WheelRight->getTorqueArbitrated();
testVehicleMotionManagementBrakeAxleRow2WheelRight->setTorqueArbitrated(0);
auto torqueFrictionMaximum = testVehicleMotionManagementBrakeAxleRow2WheelRight->getTorqueFrictionMaximum();
testVehicleMotionManagementBrakeAxleRow2WheelRight->setTorqueFrictionMaximum(0);
auto torqueFrictionMinimum = testVehicleMotionManagementBrakeAxleRow2WheelRight->getTorqueFrictionMinimum();
testVehicleMotionManagementBrakeAxleRow2WheelRight->setTorqueFrictionMinimum(0);
auto omegaUpper = testVehicleMotionManagementBrakeAxleRow2WheelRight->getOmegaUpper();
testVehicleMotionManagementBrakeAxleRow2WheelRight->setOmegaUpper(0);
auto omegaLower = testVehicleMotionManagementBrakeAxleRow2WheelRight->getOmegaLower();
testVehicleMotionManagementBrakeAxleRow2WheelRight->setOmegaLower(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator : public IVehicleMotionManagementBrakeAxleRow2WheelRight
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right object to make thread safe.
    */
    explicit VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow2WheelRight> impl);

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    void setTorque(int32_t Torque) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    int32_t getTorque() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    void setTorqueArbitrated(int32_t TorqueArbitrated) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    int32_t getTorqueArbitrated() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    void setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    int32_t getTorqueFrictionMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    void setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    int32_t getTorqueFrictionMinimum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    void setOmegaUpper(int32_t OmegaUpper) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    int32_t getOmegaUpper() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    void setOmegaLower(int32_t OmegaLower) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right implementation. */
    int32_t getOmegaLower() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right.
    */
    IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher& _getPublisher() const override;
private:
    /** The Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right object which is guarded */
    std::shared_ptr<IVehicleMotionManagementBrakeAxleRow2WheelRight> m_impl;
    // Mutex for torque property
    mutable std::shared_timed_mutex m_torqueMutex;
    // Mutex for torqueArbitrated property
    mutable std::shared_timed_mutex m_torqueArbitratedMutex;
    // Mutex for torqueFrictionMaximum property
    mutable std::shared_timed_mutex m_torqueFrictionMaximumMutex;
    // Mutex for torqueFrictionMinimum property
    mutable std::shared_timed_mutex m_torqueFrictionMinimumMutex;
    // Mutex for omegaUpper property
    mutable std::shared_timed_mutex m_omegaUpperMutex;
    // Mutex for omegaLower property
    mutable std::shared_timed_mutex m_omegaLowerMutex;
};
} // namespace Vehicle
} // namespace Cpp
