
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleMotionManagementBrakeThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_MotionManagement_Brake interface.
* @see Vehicle_MotionManagement_Brake
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleMotionManagementBrake> testVehicleMotionManagementBrake = std::make_unique<VehicleMotionManagementBrakeThreadSafeDecorator>(std::make_shared<VehicleMotionManagementBrake>());

// Thread safe access
auto vehicleForceMaximum = testVehicleMotionManagementBrake->getVehicleForceMaximum();
testVehicleMotionManagementBrake->setVehicleForceMaximum(0);
auto vehicleForceElectricMinimumArbitrated = testVehicleMotionManagementBrake->getVehicleForceElectricMinimumArbitrated();
testVehicleMotionManagementBrake->setVehicleForceElectricMinimumArbitrated(0);
auto vehicleForceElectric = testVehicleMotionManagementBrake->getVehicleForceElectric();
testVehicleMotionManagementBrake->setVehicleForceElectric(0);
auto vehicleForceDistributionFrontMaximum = testVehicleMotionManagementBrake->getVehicleForceDistributionFrontMaximum();
testVehicleMotionManagementBrake->setVehicleForceDistributionFrontMaximum(0);
auto vehicleForceDistributionFrontMinimum = testVehicleMotionManagementBrake->getVehicleForceDistributionFrontMinimum();
testVehicleMotionManagementBrake->setVehicleForceDistributionFrontMinimum(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeThreadSafeDecorator : public IVehicleMotionManagementBrake
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_MotionManagement_Brake object to make thread safe.
    */
    explicit VehicleMotionManagementBrakeThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrake> impl);

    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    void setVehicleForceMaximum(int32_t VehicleForceMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    int32_t getVehicleForceMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    void setVehicleForceElectricMinimumArbitrated(int32_t VehicleForceElectricMinimumArbitrated) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    int32_t getVehicleForceElectricMinimumArbitrated() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    void setVehicleForceElectric(int32_t VehicleForceElectric) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    int32_t getVehicleForceElectric() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    void setVehicleForceDistributionFrontMaximum(int32_t VehicleForceDistributionFrontMaximum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    int32_t getVehicleForceDistributionFrontMaximum() const override;

    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    void setVehicleForceDistributionFrontMinimum(int32_t VehicleForceDistributionFrontMinimum) override;
    /** Guards and forwards call to Vehicle_MotionManagement_Brake implementation. */
    int32_t getVehicleForceDistributionFrontMinimum() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_MotionManagement_Brake.
    */
    IVehicleMotionManagementBrakePublisher& _getPublisher() const override;
private:
    /** The Vehicle_MotionManagement_Brake object which is guarded */
    std::shared_ptr<IVehicleMotionManagementBrake> m_impl;
    // Mutex for vehicleForceMaximum property
    mutable std::shared_timed_mutex m_vehicleForceMaximumMutex;
    // Mutex for vehicleForceElectricMinimumArbitrated property
    mutable std::shared_timed_mutex m_vehicleForceElectricMinimumArbitratedMutex;
    // Mutex for vehicleForceElectric property
    mutable std::shared_timed_mutex m_vehicleForceElectricMutex;
    // Mutex for vehicleForceDistributionFrontMaximum property
    mutable std::shared_timed_mutex m_vehicleForceDistributionFrontMaximumMutex;
    // Mutex for vehicleForceDistributionFrontMinimum property
    mutable std::shared_timed_mutex m_vehicleForceDistributionFrontMinimumMutex;
};
} // namespace Vehicle
} // namespace Cpp
