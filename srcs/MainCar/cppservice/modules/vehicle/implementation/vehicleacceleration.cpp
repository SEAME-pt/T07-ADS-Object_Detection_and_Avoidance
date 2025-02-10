

#include "vehicle/implementation/vehicleacceleration.h"
#include "vehicle/generated/core/vehicleacceleration.publisher.h"
#include "vehicle/generated/core/vehicleacceleration.data.h"

using namespace Cpp::Vehicle;

VehicleAcceleration::VehicleAcceleration()
    : m_publisher(std::make_unique<VehicleAccelerationPublisher>())
{
}
VehicleAcceleration::~VehicleAcceleration()
{
}

void VehicleAcceleration::setLongitudinal(float Longitudinal)
{
    if (m_data.m_Longitudinal != Longitudinal) {
        m_data.m_Longitudinal = Longitudinal;
        m_publisher->publishLongitudinalChanged(Longitudinal);
    }
}

float VehicleAcceleration::getLongitudinal() const
{
    return m_data.m_Longitudinal;
}

void VehicleAcceleration::setLateral(float Lateral)
{
    if (m_data.m_Lateral != Lateral) {
        m_data.m_Lateral = Lateral;
        m_publisher->publishLateralChanged(Lateral);
    }
}

float VehicleAcceleration::getLateral() const
{
    return m_data.m_Lateral;
}

void VehicleAcceleration::setVertical(float Vertical)
{
    if (m_data.m_Vertical != Vertical) {
        m_data.m_Vertical = Vertical;
        m_publisher->publishVerticalChanged(Vertical);
    }
}

float VehicleAcceleration::getVertical() const
{
    return m_data.m_Vertical;
}

IVehicleAccelerationPublisher& VehicleAcceleration::_getPublisher() const
{
    return *m_publisher;
}
