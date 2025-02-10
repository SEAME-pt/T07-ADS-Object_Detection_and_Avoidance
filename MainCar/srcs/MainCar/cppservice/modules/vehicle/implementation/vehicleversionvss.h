
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehicleversionvss.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_VersionVSS implementation.
*/
class CPP_VEHICLE_EXPORT VehicleVersionVss : public IVehicleVersionVss
{
public:
    explicit VehicleVersionVss();
    ~VehicleVersionVss();
public:
    /**
    * Major Supported Version of VSS - Major version.
    */
    void setMajor(int32_t Major) override;
    int32_t getMajor() const override;
    
    /**
    * Minor Supported Version of VSS - Minor version.
    */
    void setMinor(int32_t Minor) override;
    int32_t getMinor() const override;
    
    /**
    * Patch Supported Version of VSS - Patch version.
    */
    void setPatch(int32_t Patch) override;
    int32_t getPatch() const override;
    
    /**
    * Label Label to further describe the version.
    */
    void setLabel(const std::string& Label) override;
    const std::string& getLabel() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_VersionVSS changes and signal emission.
    * @return The publisher for Vehicle_VersionVSS.
    */
    IVehicleVersionVssPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_VersionVSS. */
    std::unique_ptr<IVehicleVersionVssPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_VersionVSS. */
    VehicleVersionVssData m_data;
};
} // namespace Vehicle
} // namespace Cpp
