

#include "vehicle/implementation/vehicleconnectivity.h"
#include "vehicle/generated/core/vehicleconnectivity.publisher.h"
#include "vehicle/generated/core/vehicleconnectivity.data.h"

using namespace Cpp::Vehicle;

VehicleConnectivity::VehicleConnectivity()
    : m_publisher(std::make_unique<VehicleConnectivityPublisher>())
{
}
VehicleConnectivity::~VehicleConnectivity()
{
}

void VehicleConnectivity::setIsConnectivityAvailable(bool IsConnectivityAvailable)
{
    if (m_data.m_IsConnectivityAvailable != IsConnectivityAvailable) {
        m_data.m_IsConnectivityAvailable = IsConnectivityAvailable;
        m_publisher->publishIsConnectivityAvailableChanged(IsConnectivityAvailable);
    }
}

bool VehicleConnectivity::getIsConnectivityAvailable() const
{
    return m_data.m_IsConnectivityAvailable;
}

IVehicleConnectivityPublisher& VehicleConnectivity::_getPublisher() const
{
    return *m_publisher;
}
