
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclebodyhorn.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Body_Horn implementation.
*/
class CPP_VEHICLE_EXPORT VehicleBodyHorn : public IVehicleBodyHorn
{
public:
    explicit VehicleBodyHorn();
    ~VehicleBodyHorn();
public:
    /**
    * IsActive Horn active or inactive. True = Active. False = Inactive.
    */
    void setIsActive(bool IsActive) override;
    bool getIsActive() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Horn changes and signal emission.
    * @return The publisher for Vehicle_Body_Horn.
    */
    IVehicleBodyHornPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Body_Horn. */
    std::unique_ptr<IVehicleBodyHornPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Body_Horn. */
    VehicleBodyHornData m_data;
};
} // namespace Vehicle
} // namespace Cpp
