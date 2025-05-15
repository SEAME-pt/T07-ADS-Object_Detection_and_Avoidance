
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehicleacceleration.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Acceleration implementation.
*/
class CPP_VEHICLE_EXPORT VehicleAcceleration : public IVehicleAcceleration
{
public:
    explicit VehicleAcceleration();
    ~VehicleAcceleration();
public:
    /**
    * Longitudinal Vehicle acceleration in X (longitudinal acceleration).
    */
    void setLongitudinal(float Longitudinal) override;
    float getLongitudinal() const override;
    
    /**
    * Lateral Vehicle acceleration in Y (lateral acceleration).
    */
    void setLateral(float Lateral) override;
    float getLateral() const override;
    
    /**
    * Vertical Vehicle acceleration in Z (vertical acceleration).
    */
    void setVertical(float Vertical) override;
    float getVertical() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Acceleration changes and signal emission.
    * @return The publisher for Vehicle_Acceleration.
    */
    IVehicleAccelerationPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Acceleration. */
    std::unique_ptr<IVehicleAccelerationPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Acceleration. */
    VehicleAccelerationData m_data;
};
} // namespace Vehicle
} // namespace Cpp
