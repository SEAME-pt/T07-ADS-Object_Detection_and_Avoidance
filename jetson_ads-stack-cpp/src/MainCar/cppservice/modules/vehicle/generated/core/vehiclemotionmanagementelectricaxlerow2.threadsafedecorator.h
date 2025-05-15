
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleMotionManagementElectricAxleRow2ThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_MotionManagement_ElectricAxle_Row2 interface.
* @see Vehicle_MotionManagement_ElectricAxle_Row2
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleMotionManagementElectricAxleRow2> testVehicleMotionManagementElectricAxleRow2 = std::make_unique<VehicleMotionManagementElectricAxleRow2ThreadSafeDecorator>(std::make_shared<VehicleMotionManagementElectricAxleRow2>());

// Thread safe access
auto torque = testVehicleMotionManagementElectricAxleRow2->getTorque();
testVehicleMotionManagementElectricAxleRow2->setTorque(0);
auto torqueTarget = testVehicleMotionManagementElectricAxleRow2->getTorqueTarget();
testVehicleMotionManagementElectricAxleRow2->setTorqueTarget(0);
auto torqueMaximum = testVehicleMotionManagementElectricAxleRow2->getTorqueMaximum();
testVehicleMotionManagementElectricAxleRow2->setTorqueMaximum(0);
auto torqueMinimum = testVehicleMotionManagementElectricAxleRow2->getTorqueMinimum();
testVehicleMotionManagementElectricAxleRow2->setTorqueMinimum(0);
auto torqueLongTermMaximum = testVehicleMotionManagementElectricAxleRow2->getTorqueLongTermMaximum();
testVehicleMotionManagementElectricAxleRow2->setTorqueLongTermMaximum(0);
auto torqueLongTermMinimum = testVehicleMotionManagementElectricAxleRow2->getTorqueLongTermMinimum();
testVehicleMotionManagementElectricAxleRow2->setTorqueLongTermMinimum(0);
auto torqueShortTermMaximum = testVehicleMotionManagementElectricAxleRow2->getTorqueShortTermMaximum();
testVehicleMotionManagementElectricAxleRow2->setTorqueShortTermMaximum(0);
auto torqueShortTermMinimum = testVehicleMotionManagementElectricAxleRow2->getTorqueShortTermMinimum();
testVehicleMotionManagementElectricAxleRow2->setTorqueShortTermMinimum(0);
auto torqueMaximumLimit = testVehicleMotionManagementElectricAxleRow2->getTorqueMaximumLimit();
testVehicleMotionManagementElectricAxleRow2->setTorqueMaximumLimit(0);
auto torqueMinimumLimit = testVehicleMotionManagementElectricAxleRow2->getTorqueMinimumLimit();
testVehicleMotionManagementElectricAxleRow2->setTorqueMinimumLimit(0);
auto rotationalSpeed = testVehicleMotionManagementElectricAxleRow2->getRotationalSpeed();
testVehicleMotionManagementElectricAxleRow2->setRotationalSpeed(0);
auto rotationalSpeedTarget = testVehicleMotionManagementElectricAxleRow2->getRotationalSpeedTarget();
testVehicleMotionManagementElectricAxleRow2->setRotationalSpeedTarget(0);
auto rotationalSpeedMaximumLimit = testVehicleMotionManagementElectricAxleRow2->getRotationalSpeedMaximumLimit();
testVehicleMotionManagementElectricAxleRow2->setRotationalSpeedMaximumLimit(0);
auto rotationalSpeedMinimumLimit = testVehicleMotionManagementElectricAxleRow2->getRotationalSpeedMinimumLimit();
testVehicleMotionManagementElectricAxleRow2->setRotationalSpeedMinimumLimit(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementElectricAxleRow2ThreadSafeDecorator : public IVehicleMotionManagementElectricAxleRow2
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_MotionManagement_ElectricAxle_Row2 object to make thread safe.
    */
    explicit VehicleMotionManagementElectricAxleRow2ThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementElectricAxleRow2> impl);

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorque(int32_t Torque) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorque() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueTarget(int32_t TorqueTarget) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueTarget() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueMaximum(int32_t TorqueMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueMinimum(int32_t TorqueMinimum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueMinimum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueLongTermMaximum(int32_t TorqueLongTermMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueLongTermMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueLongTermMinimum(int32_t TorqueLongTermMinimum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueLongTermMinimum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueShortTermMaximum(int32_t TorqueShortTermMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueShortTermMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueShortTermMinimum(int32_t TorqueShortTermMinimum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueShortTermMinimum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueMaximumLimit(int32_t TorqueMaximumLimit) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueMaximumLimit() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setTorqueMinimumLimit(int32_t TorqueMinimumLimit) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getTorqueMinimumLimit() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setRotationalSpeed(int32_t RotationalSpeed) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getRotationalSpeed() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setRotationalSpeedTarget(int32_t RotationalSpeedTarget) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getRotationalSpeedTarget() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setRotationalSpeedMaximumLimit(int32_t RotationalSpeedMaximumLimit) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getRotationalSpeedMaximumLimit() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    void setRotationalSpeedMinimumLimit(int32_t RotationalSpeedMinimumLimit) override;
    /** Guards and forwards call to Vehicle_MotionManagement_ElectricAxle_Row2 implementation. */
    int32_t getRotationalSpeedMinimumLimit() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_ElectricAxle_Row2 changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_MotionManagement_ElectricAxle_Row2.
    */
    IVehicleMotionManagementElectricAxleRow2Publisher& _getPublisher() const override;
private:
    /** The Vehicle_MotionManagement_ElectricAxle_Row2 object which is guarded */
    std::shared_ptr<IVehicleMotionManagementElectricAxleRow2> m_impl;
    // Mutex for torque property
    mutable std::shared_timed_mutex m_torqueMutex;
    // Mutex for torqueTarget property
    mutable std::shared_timed_mutex m_torqueTargetMutex;
    // Mutex for torqueMaximum property
    mutable std::shared_timed_mutex m_torqueMaximumMutex;
    // Mutex for torqueMinimum property
    mutable std::shared_timed_mutex m_torqueMinimumMutex;
    // Mutex for torqueLongTermMaximum property
    mutable std::shared_timed_mutex m_torqueLongTermMaximumMutex;
    // Mutex for torqueLongTermMinimum property
    mutable std::shared_timed_mutex m_torqueLongTermMinimumMutex;
    // Mutex for torqueShortTermMaximum property
    mutable std::shared_timed_mutex m_torqueShortTermMaximumMutex;
    // Mutex for torqueShortTermMinimum property
    mutable std::shared_timed_mutex m_torqueShortTermMinimumMutex;
    // Mutex for torqueMaximumLimit property
    mutable std::shared_timed_mutex m_torqueMaximumLimitMutex;
    // Mutex for torqueMinimumLimit property
    mutable std::shared_timed_mutex m_torqueMinimumLimitMutex;
    // Mutex for rotationalSpeed property
    mutable std::shared_timed_mutex m_rotationalSpeedMutex;
    // Mutex for rotationalSpeedTarget property
    mutable std::shared_timed_mutex m_rotationalSpeedTargetMutex;
    // Mutex for rotationalSpeedMaximumLimit property
    mutable std::shared_timed_mutex m_rotationalSpeedMaximumLimitMutex;
    // Mutex for rotationalSpeedMinimumLimit property
    mutable std::shared_timed_mutex m_rotationalSpeedMinimumLimitMutex;
};
} // namespace Vehicle
} // namespace Cpp
