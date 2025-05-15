
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclediagnostics.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Diagnostics implementation.
*/
class CPP_VEHICLE_EXPORT VehicleDiagnostics : public IVehicleDiagnostics
{
public:
    explicit VehicleDiagnostics();
    ~VehicleDiagnostics();
public:
    /**
    * DTCCount Number of Diagnostic Trouble Codes (DTC)
    */
    void setDtcCount(int32_t DTCCount) override;
    int32_t getDtcCount() const override;
    
    /**
    * DTCList List of currently active DTCs formatted according OBD II (SAE-J2012DA_201812) standard ([P|C|B|U]XXXXX )
    */
    void setDtcList(const std::list<std::string>& DTCList) override;
    const std::list<std::string>& getDtcList() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Diagnostics changes and signal emission.
    * @return The publisher for Vehicle_Diagnostics.
    */
    IVehicleDiagnosticsPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Diagnostics. */
    std::unique_ptr<IVehicleDiagnosticsPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Diagnostics. */
    VehicleDiagnosticsData m_data;
};
} // namespace Vehicle
} // namespace Cpp
