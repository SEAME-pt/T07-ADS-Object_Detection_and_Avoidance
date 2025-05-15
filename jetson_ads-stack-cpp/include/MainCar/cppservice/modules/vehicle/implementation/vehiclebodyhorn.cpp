

#include "vehicle/implementation/vehiclebodyhorn.h"
#include "vehicle/generated/core/vehiclebodyhorn.publisher.h"
#include "vehicle/generated/core/vehiclebodyhorn.data.h"

using namespace Cpp::Vehicle;

VehicleBodyHorn::VehicleBodyHorn()
    : m_publisher(std::make_unique<VehicleBodyHornPublisher>())
{
}
VehicleBodyHorn::~VehicleBodyHorn()
{
}

void VehicleBodyHorn::setIsActive(bool IsActive)
{
    if (m_data.m_IsActive != IsActive) {
        m_data.m_IsActive = IsActive;
        m_publisher->publishIsActiveChanged(IsActive);
    }
}

bool VehicleBodyHorn::getIsActive() const
{
    return m_data.m_IsActive;
}

IVehicleBodyHornPublisher& VehicleBodyHorn::_getPublisher() const
{
    return *m_publisher;
}
