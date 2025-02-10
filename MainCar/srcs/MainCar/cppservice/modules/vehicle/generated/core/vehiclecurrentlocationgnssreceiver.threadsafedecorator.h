
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleCurrentLocationGnssReceiverThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_CurrentLocation_GNSSReceiver interface.
* @see Vehicle_CurrentLocation_GNSSReceiver
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleCurrentLocationGnssReceiver> testVehicleCurrentLocationGnssReceiver = std::make_unique<VehicleCurrentLocationGnssReceiverThreadSafeDecorator>(std::make_shared<VehicleCurrentLocationGnssReceiver>());

// Thread safe access
auto fixType = testVehicleCurrentLocationGnssReceiver->getFixType();
testVehicleCurrentLocationGnssReceiver->setFixType(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::NONE);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleCurrentLocationGnssReceiverThreadSafeDecorator : public IVehicleCurrentLocationGnssReceiver
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_CurrentLocation_GNSSReceiver object to make thread safe.
    */
    explicit VehicleCurrentLocationGnssReceiverThreadSafeDecorator(std::shared_ptr<IVehicleCurrentLocationGnssReceiver> impl);

    /** Guards and forwards call to Vehicle_CurrentLocation_GNSSReceiver implementation. */
    void setFixType(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType) override;
    /** Guards and forwards call to Vehicle_CurrentLocation_GNSSReceiver implementation. */
    Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum getFixType() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation_GNSSReceiver changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_CurrentLocation_GNSSReceiver.
    */
    IVehicleCurrentLocationGnssReceiverPublisher& _getPublisher() const override;
private:
    /** The Vehicle_CurrentLocation_GNSSReceiver object which is guarded */
    std::shared_ptr<IVehicleCurrentLocationGnssReceiver> m_impl;
    // Mutex for fixType property
    mutable std::shared_timed_mutex m_fixTypeMutex;
};
} // namespace Vehicle
} // namespace Cpp
