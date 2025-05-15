
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition interface.
* @see Vehicle_CurrentLocation_GNSSReceiver_MountingPosition
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleCurrentLocationGnssReceiverMountingPosition> testVehicleCurrentLocationGnssReceiverMountingPosition = std::make_unique<VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator>(std::make_shared<VehicleCurrentLocationGnssReceiverMountingPosition>());

// Thread safe access
auto x = testVehicleCurrentLocationGnssReceiverMountingPosition->getX();
testVehicleCurrentLocationGnssReceiverMountingPosition->setX(0);
auto y = testVehicleCurrentLocationGnssReceiverMountingPosition->getY();
testVehicleCurrentLocationGnssReceiverMountingPosition->setY(0);
auto z = testVehicleCurrentLocationGnssReceiverMountingPosition->getZ();
testVehicleCurrentLocationGnssReceiverMountingPosition->setZ(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator : public IVehicleCurrentLocationGnssReceiverMountingPosition
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_CurrentLocation_GNSSReceiver_MountingPosition object to make thread safe.
    */
    explicit VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator(std::shared_ptr<IVehicleCurrentLocationGnssReceiverMountingPosition> impl);

    /** Guards and forwards call to Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation. */
    void setX(int32_t X) override;
    /** Guards and forwards call to Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation. */
    int32_t getX() const override;

    /** Guards and forwards call to Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation. */
    void setY(int32_t Y) override;
    /** Guards and forwards call to Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation. */
    int32_t getY() const override;

    /** Guards and forwards call to Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation. */
    void setZ(int32_t Z) override;
    /** Guards and forwards call to Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation. */
    int32_t getZ() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation_GNSSReceiver_MountingPosition changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_CurrentLocation_GNSSReceiver_MountingPosition.
    */
    IVehicleCurrentLocationGnssReceiverMountingPositionPublisher& _getPublisher() const override;
private:
    /** The Vehicle_CurrentLocation_GNSSReceiver_MountingPosition object which is guarded */
    std::shared_ptr<IVehicleCurrentLocationGnssReceiverMountingPosition> m_impl;
    // Mutex for x property
    mutable std::shared_timed_mutex m_xMutex;
    // Mutex for y property
    mutable std::shared_timed_mutex m_yMutex;
    // Mutex for z property
    mutable std::shared_timed_mutex m_zMutex;
};
} // namespace Vehicle
} // namespace Cpp
