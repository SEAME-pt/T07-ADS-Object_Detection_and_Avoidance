

#include "vehicle/implementation/vehiclecurrentlocationgnssreceiver.h"
#include "vehicle/generated/core/vehiclecurrentlocationgnssreceiver.publisher.h"
#include "vehicle/generated/core/vehiclecurrentlocationgnssreceiver.data.h"

using namespace Cpp::Vehicle;

VehicleCurrentLocationGnssReceiver::VehicleCurrentLocationGnssReceiver()
    : m_publisher(std::make_unique<VehicleCurrentLocationGnssReceiverPublisher>())
{
}
VehicleCurrentLocationGnssReceiver::~VehicleCurrentLocationGnssReceiver()
{
}

void VehicleCurrentLocationGnssReceiver::setFixType(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType)
{
    if (m_data.m_FixType != FixType) {
        m_data.m_FixType = FixType;
        m_publisher->publishFixTypeChanged(FixType);
    }
}

Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum VehicleCurrentLocationGnssReceiver::getFixType() const
{
    return m_data.m_FixType;
}

IVehicleCurrentLocationGnssReceiverPublisher& VehicleCurrentLocationGnssReceiver::_getPublisher() const
{
    return *m_publisher;
}
