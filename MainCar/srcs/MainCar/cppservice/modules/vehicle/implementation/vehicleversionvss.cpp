

#include "vehicle/implementation/vehicleversionvss.h"
#include "vehicle/generated/core/vehicleversionvss.publisher.h"
#include "vehicle/generated/core/vehicleversionvss.data.h"

using namespace Cpp::Vehicle;

VehicleVersionVss::VehicleVersionVss()
    : m_publisher(std::make_unique<VehicleVersionVssPublisher>())
{
}
VehicleVersionVss::~VehicleVersionVss()
{
}

void VehicleVersionVss::setMajor(int32_t Major)
{
    if (m_data.m_Major != Major) {
        m_data.m_Major = Major;
        m_publisher->publishMajorChanged(Major);
    }
}

int32_t VehicleVersionVss::getMajor() const
{
    return m_data.m_Major;
}

void VehicleVersionVss::setMinor(int32_t Minor)
{
    if (m_data.m_Minor != Minor) {
        m_data.m_Minor = Minor;
        m_publisher->publishMinorChanged(Minor);
    }
}

int32_t VehicleVersionVss::getMinor() const
{
    return m_data.m_Minor;
}

void VehicleVersionVss::setPatch(int32_t Patch)
{
    if (m_data.m_Patch != Patch) {
        m_data.m_Patch = Patch;
        m_publisher->publishPatchChanged(Patch);
    }
}

int32_t VehicleVersionVss::getPatch() const
{
    return m_data.m_Patch;
}

void VehicleVersionVss::setLabel(const std::string& Label)
{
    if (m_data.m_Label != Label) {
        m_data.m_Label = Label;
        m_publisher->publishLabelChanged(Label);
    }
}

const std::string& VehicleVersionVss::getLabel() const
{
    return m_data.m_Label;
}

IVehicleVersionVssPublisher& VehicleVersionVss::_getPublisher() const
{
    return *m_publisher;
}
