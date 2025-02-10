
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehicledriveridentifierdeprecated.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Driver_Identifier_Deprecated implementation.
*/
class CPP_VEHICLE_EXPORT VehicleDriverIdentifierDeprecated : public IVehicleDriverIdentifierDeprecated
{
public:
    explicit VehicleDriverIdentifierDeprecated();
    ~VehicleDriverIdentifierDeprecated();
public:
    /**
    * Subject_Deprecated Subject for the authentication of the occupant e.g. UserID 7331677.
    */
    void setSubjectDeprecated(const std::string& Subject_Deprecated) override;
    const std::string& getSubjectDeprecated() const override;
    
    /**
    * Issuer_Deprecated Unique Issuer for the authentication of the occupant e.g. https://accounts.funcorp.com.
    */
    void setIssuerDeprecated(const std::string& Issuer_Deprecated) override;
    const std::string& getIssuerDeprecated() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Driver_Identifier_Deprecated changes and signal emission.
    * @return The publisher for Vehicle_Driver_Identifier_Deprecated.
    */
    IVehicleDriverIdentifierDeprecatedPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Driver_Identifier_Deprecated. */
    std::unique_ptr<IVehicleDriverIdentifierDeprecatedPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Driver_Identifier_Deprecated. */
    VehicleDriverIdentifierDeprecatedData m_data;
};
} // namespace Vehicle
} // namespace Cpp
