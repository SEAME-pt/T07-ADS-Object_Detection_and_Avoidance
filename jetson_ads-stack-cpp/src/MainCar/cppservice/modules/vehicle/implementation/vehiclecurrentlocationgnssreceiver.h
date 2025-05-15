
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclecurrentlocationgnssreceiver.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_CurrentLocation_GNSSReceiver implementation.
*/
class CPP_VEHICLE_EXPORT VehicleCurrentLocationGnssReceiver : public IVehicleCurrentLocationGnssReceiver
{
public:
    explicit VehicleCurrentLocationGnssReceiver();
    ~VehicleCurrentLocationGnssReceiver();
public:
    /**
    * FixType Fix status of GNSS receiver.
    */
    void setFixType(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType) override;
    Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum getFixType() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_CurrentLocation_GNSSReceiver changes and signal emission.
    * @return The publisher for Vehicle_CurrentLocation_GNSSReceiver.
    */
    IVehicleCurrentLocationGnssReceiverPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_CurrentLocation_GNSSReceiver. */
    std::unique_ptr<IVehicleCurrentLocationGnssReceiverPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_CurrentLocation_GNSSReceiver. */
    VehicleCurrentLocationGnssReceiverData m_data;
};
} // namespace Vehicle
} // namespace Cpp
