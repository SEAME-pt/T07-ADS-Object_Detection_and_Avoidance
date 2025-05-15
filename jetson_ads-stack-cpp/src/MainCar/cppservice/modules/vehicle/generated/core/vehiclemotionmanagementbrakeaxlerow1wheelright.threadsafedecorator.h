
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right interface.
* @see Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleMotionManagementBrakeAxleRow1WheelRight> testVehicleMotionManagementBrakeAxleRow1WheelRight = std::make_unique<VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator>(std::make_shared<VehicleMotionManagementBrakeAxleRow1WheelRight>());

// Thread safe access
auto torque = testVehicleMotionManagementBrakeAxleRow1WheelRight->getTorque();
testVehicleMotionManagementBrakeAxleRow1WheelRight->setTorque(0);
auto torqueArbitrated = testVehicleMotionManagementBrakeAxleRow1WheelRight->getTorqueArbitrated();
testVehicleMotionManagementBrakeAxleRow1WheelRight->setTorqueArbitrated(0);
auto torqueFrictionMaximum = testVehicleMotionManagementBrakeAxleRow1WheelRight->getTorqueFrictionMaximum();
testVehicleMotionManagementBrakeAxleRow1WheelRight->setTorqueFrictionMaximum(0);
auto torqueFrictionMinimum = testVehicleMotionManagementBrakeAxleRow1WheelRight->getTorqueFrictionMinimum();
testVehicleMotionManagementBrakeAxleRow1WheelRight->setTorqueFrictionMinimum(0);
auto omegaUpper = testVehicleMotionManagementBrakeAxleRow1WheelRight->getOmegaUpper();
testVehicleMotionManagementBrakeAxleRow1WheelRight->setOmegaUpper(0);
auto omegaLower = testVehicleMotionManagementBrakeAxleRow1WheelRight->getOmegaLower();
testVehicleMotionManagementBrakeAxleRow1WheelRight->setOmegaLower(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator : public IVehicleMotionManagementBrakeAxleRow1WheelRight
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right object to make thread safe.
    */
    explicit VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow1WheelRight> impl);

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    void setTorque(int32_t Torque) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    int32_t getTorque() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    void setTorqueArbitrated(int32_t TorqueArbitrated) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    int32_t getTorqueArbitrated() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    void setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    int32_t getTorqueFrictionMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    void setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    int32_t getTorqueFrictionMinimum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    void setOmegaUpper(int32_t OmegaUpper) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    int32_t getOmegaUpper() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    void setOmegaLower(int32_t OmegaLower) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right implementation. */
    int32_t getOmegaLower() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right.
    */
    IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher& _getPublisher() const override;
private:
    /** The Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right object which is guarded */
    std::shared_ptr<IVehicleMotionManagementBrakeAxleRow1WheelRight> m_impl;
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
