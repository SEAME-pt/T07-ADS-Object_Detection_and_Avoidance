
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclemotionmanagementsteeringsteeringwheel.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_MotionManagement_Steering_SteeringWheel implementation.
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementSteeringSteeringWheel : public IVehicleMotionManagementSteeringSteeringWheel
{
public:
    explicit VehicleMotionManagementSteeringSteeringWheel();
    ~VehicleMotionManagementSteeringSteeringWheel();
public:
    /**
    * Angle Represents the current input angle of the steering system, typically corresponds to angle applied by driver on the steering-wheel. Positive for angle in counterclockwise direction (based on ISO8855).
    */
    void setAngle(int32_t Angle) override;
    int32_t getAngle() const override;
    
    /**
    * AngleTarget Steering-wheel angle request to the steering actuator (external set-point for steer-by-wire). Positive sign for angle in counterclockwise direction (based on ISO8855).
    */
    void setAngleTarget(int32_t AngleTarget) override;
    int32_t getAngleTarget() const override;
    
    /**
    * AngleTargetMode Mode used for controlling Steering-wheel angle interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    void setAngleTargetMode(int32_t AngleTargetMode) override;
    int32_t getAngleTargetMode() const override;
    
    /**
    * TorqueOffsetTarget Steering-wheel torque offset request to the steering actuator, added to the actuator internal calculated target value. Positive sign for torque in counterclockwise direction (based on ISO8855).
    */
    void setTorqueOffsetTarget(int32_t TorqueOffsetTarget) override;
    int32_t getTorqueOffsetTarget() const override;
    
    /**
    * TorqueOffsetTargetMode Mode used for controlling steering-wheel torque offset interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    void setTorqueOffsetTargetMode(int32_t TorqueOffsetTargetMode) override;
    int32_t getTorqueOffsetTargetMode() const override;
    
    /**
    * TorqueTarget Steering-wheel torque request to the steering actuator (external set-point for steer-by-wire). Positive sign for torque in counterclockwise direction (based on ISO8855).
    */
    void setTorqueTarget(int32_t TorqueTarget) override;
    int32_t getTorqueTarget() const override;
    
    /**
    * TorqueTargetMode Mode used for controlling Steering-wheel torque interface. 0 indicates interface disabled. Other values activate vehicle specific modes.
    */
    void setTorqueTargetMode(int32_t TorqueTargetMode) override;
    int32_t getTorqueTargetMode() const override;
    
    /**
    * Torque Represents the current input torque for steering system, typically corresponds to torque applied by driver on the steering wheel. Positive for torque in counterclockwise direction (based on ISO8855).
    */
    void setTorque(int32_t Torque) override;
    int32_t getTorque() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Steering_SteeringWheel changes and signal emission.
    * @return The publisher for Vehicle_MotionManagement_Steering_SteeringWheel.
    */
    IVehicleMotionManagementSteeringSteeringWheelPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_MotionManagement_Steering_SteeringWheel. */
    std::unique_ptr<IVehicleMotionManagementSteeringSteeringWheelPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_MotionManagement_Steering_SteeringWheel. */
    VehicleMotionManagementSteeringSteeringWheelData m_data;
};
} // namespace Vehicle
} // namespace Cpp
