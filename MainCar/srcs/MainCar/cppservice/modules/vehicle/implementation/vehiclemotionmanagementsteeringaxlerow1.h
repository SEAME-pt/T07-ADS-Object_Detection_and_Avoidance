
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclemotionmanagementsteeringaxlerow1.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_MotionManagement_Steering_Axle_Row1 implementation.
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementSteeringAxleRow1 : public IVehicleMotionManagementSteeringAxleRow1
{
public:
    explicit VehicleMotionManagementSteeringAxleRow1();
    ~VehicleMotionManagementSteeringAxleRow1();
public:
    /**
    * RackPositionOffsetTarget Rack position offset request to the axle steering actuator (for steer-by-wire), added to the actuator internal calculated set-point. Positive values without internal calculated set point change lead to a left movement of the vehicle (based on ISO8855).
    */
    void setRackPositionOffsetTarget(int32_t RackPositionOffsetTarget) override;
    int32_t getRackPositionOffsetTarget() const override;
    
    /**
    * RackPositionOffsetTargetMode Mode used for controlling rack position offset interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    void setRackPositionOffsetTargetMode(int32_t RackPositionOffsetTargetMode) override;
    int32_t getRackPositionOffsetTargetMode() const override;
    
    /**
    * RackPosition Represents the current position of the steering rack on axle steering actuator. Positive values leads to a left turn of the vehicle (based on ISO8855).
    */
    void setRackPosition(int32_t RackPosition) override;
    int32_t getRackPosition() const override;
    
    /**
    * RackPositionTarget Rack position request to the axle steering actuator (external set-point). Positive values lead to a left turn of the vehicle (based on ISO8855).
    */
    void setRackPositionTarget(int32_t RackPositionTarget) override;
    int32_t getRackPositionTarget() const override;
    
    /**
    * RackPositionTargetMode Mode used for controlling rack position interface of axle steering actuator. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    void setRackPositionTargetMode(int32_t RackPositionTargetMode) override;
    int32_t getRackPositionTargetMode() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Steering_Axle_Row1 changes and signal emission.
    * @return The publisher for Vehicle_MotionManagement_Steering_Axle_Row1.
    */
    IVehicleMotionManagementSteeringAxleRow1Publisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_MotionManagement_Steering_Axle_Row1. */
    std::unique_ptr<IVehicleMotionManagementSteeringAxleRow1Publisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_MotionManagement_Steering_Axle_Row1. */
    VehicleMotionManagementSteeringAxleRow1Data m_data;
};
} // namespace Vehicle
} // namespace Cpp
