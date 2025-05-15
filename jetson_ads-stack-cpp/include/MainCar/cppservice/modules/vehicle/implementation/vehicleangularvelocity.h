
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehicleangularvelocity.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_AngularVelocity implementation.
*/
class CPP_VEHICLE_EXPORT VehicleAngularVelocity : public IVehicleAngularVelocity
{
public:
    explicit VehicleAngularVelocity();
    ~VehicleAngularVelocity();
public:
    /**
    * Roll Vehicle rotation rate along X (longitudinal).
    */
    void setRoll(float Roll) override;
    float getRoll() const override;
    
    /**
    * Pitch Vehicle rotation rate along Y (lateral).
    */
    void setPitch(float Pitch) override;
    float getPitch() const override;
    
    /**
    * Yaw Vehicle rotation rate along Z (vertical).
    */
    void setYaw(float Yaw) override;
    float getYaw() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_AngularVelocity changes and signal emission.
    * @return The publisher for Vehicle_AngularVelocity.
    */
    IVehicleAngularVelocityPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_AngularVelocity. */
    std::unique_ptr<IVehicleAngularVelocityPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_AngularVelocity. */
    VehicleAngularVelocityData m_data;
};
} // namespace Vehicle
} // namespace Cpp
