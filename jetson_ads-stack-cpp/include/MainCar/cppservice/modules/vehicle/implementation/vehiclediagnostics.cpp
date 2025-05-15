

#include "vehicle/implementation/vehiclediagnostics.h"
#include "vehicle/generated/core/vehiclediagnostics.publisher.h"
#include "vehicle/generated/core/vehiclediagnostics.data.h"

using namespace Cpp::Vehicle;

VehicleDiagnostics::VehicleDiagnostics()
    : m_publisher(std::make_unique<VehicleDiagnosticsPublisher>())
{
}
VehicleDiagnostics::~VehicleDiagnostics()
{
}

void VehicleDiagnostics::setDtcCount(int32_t DTCCount)
{
    if (m_data.m_DTCCount != DTCCount) {
        m_data.m_DTCCount = DTCCount;
        m_publisher->publishDtcCountChanged(DTCCount);
    }
}

int32_t VehicleDiagnostics::getDtcCount() const
{
    return m_data.m_DTCCount;
}

void VehicleDiagnostics::setDtcList(const std::list<std::string>& DTCList)
{
    if (m_data.m_DTCList != DTCList) {
        m_data.m_DTCList = DTCList;
        m_publisher->publishDtcListChanged(DTCList);
    }
}

const std::list<std::string>& VehicleDiagnostics::getDtcList() const
{
    return m_data.m_DTCList;
}

IVehicleDiagnosticsPublisher& VehicleDiagnostics::_getPublisher() const
{
    return *m_publisher;
}
