

#include "vehicle/implementation/vehiclecurrentlocationgnssreceivermountingposition.h"
#include "vehicle/generated/core/vehiclecurrentlocationgnssreceivermountingposition.publisher.h"
#include "vehicle/generated/core/vehiclecurrentlocationgnssreceivermountingposition.data.h"

using namespace Cpp::Vehicle;

VehicleCurrentLocationGnssReceiverMountingPosition::VehicleCurrentLocationGnssReceiverMountingPosition()
    : m_publisher(std::make_unique<VehicleCurrentLocationGnssReceiverMountingPositionPublisher>())
{
}
VehicleCurrentLocationGnssReceiverMountingPosition::~VehicleCurrentLocationGnssReceiverMountingPosition()
{
}

void VehicleCurrentLocationGnssReceiverMountingPosition::setX(int32_t X)
{
    if (m_data.m_X != X) {
        m_data.m_X = X;
        m_publisher->publishXChanged(X);
    }
}

int32_t VehicleCurrentLocationGnssReceiverMountingPosition::getX() const
{
    return m_data.m_X;
}

void VehicleCurrentLocationGnssReceiverMountingPosition::setY(int32_t Y)
{
    if (m_data.m_Y != Y) {
        m_data.m_Y = Y;
        m_publisher->publishYChanged(Y);
    }
}

int32_t VehicleCurrentLocationGnssReceiverMountingPosition::getY() const
{
    return m_data.m_Y;
}

void VehicleCurrentLocationGnssReceiverMountingPosition::setZ(int32_t Z)
{
    if (m_data.m_Z != Z) {
        m_data.m_Z = Z;
        m_publisher->publishZChanged(Z);
    }
}

int32_t VehicleCurrentLocationGnssReceiverMountingPosition::getZ() const
{
    return m_data.m_Z;
}

IVehicleCurrentLocationGnssReceiverMountingPositionPublisher& VehicleCurrentLocationGnssReceiverMountingPosition::_getPublisher() const
{
    return *m_publisher;
}
