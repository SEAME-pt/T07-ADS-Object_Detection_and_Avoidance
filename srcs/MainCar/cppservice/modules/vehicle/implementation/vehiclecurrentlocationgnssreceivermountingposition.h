
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclecurrentlocationgnssreceivermountingposition.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_CurrentLocation_GNSSReceiver_MountingPosition implementation.
*/
class CPP_VEHICLE_EXPORT VehicleCurrentLocationGnssReceiverMountingPosition : public IVehicleCurrentLocationGnssReceiverMountingPosition
{
public:
    explicit VehicleCurrentLocationGnssReceiverMountingPosition();
    ~VehicleCurrentLocationGnssReceiverMountingPosition();
public:
    /**
    * X Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = forward of rear axle. Negative values = backward of rear axle.
    */
    void setX(int32_t X) override;
    int32_t getX() const override;
    
    /**
    * Y Mounting position of GNSS receiver antenna relative to vehicle coordinate system. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = left of origin. Negative values = right of origin. Left/Right is as seen from driver perspective, i.e. by a person looking forward.
    */
    void setY(int32_t Y) override;
    int32_t getY() const override;
    
    /**
    * Z Mounting position of GNSS receiver on Z-axis. Axis definitions according to ISO 8855. Origin at center of (first) rear axle. Positive values = above center of rear axle. Negative values = below center of rear axle.
    */
    void setZ(int32_t Z) override;
    int32_t getZ() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation_GNSSReceiver_MountingPosition changes and signal emission.
    * @return The publisher for Vehicle_CurrentLocation_GNSSReceiver_MountingPosition.
    */
    IVehicleCurrentLocationGnssReceiverMountingPositionPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition. */
    std::unique_ptr<IVehicleCurrentLocationGnssReceiverMountingPositionPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_CurrentLocation_GNSSReceiver_MountingPosition. */
    VehicleCurrentLocationGnssReceiverMountingPositionData m_data;
};
} // namespace Vehicle
} // namespace Cpp
