
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2wheelleft.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Left implementation.
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow2WheelLeft : public IVehicleMotionManagementBrakeAxleRow2WheelLeft
{
public:
    explicit VehicleMotionManagementBrakeAxleRow2WheelLeft();
    ~VehicleMotionManagementBrakeAxleRow2WheelLeft();
public:
    /**
    * Torque Estimated friction brake torque. Brake Torque < 0Nm.
    */
    void setTorque(int32_t Torque) override;
    int32_t getTorque() const override;
    
    /**
    * TorqueArbitrated Brake system internally calculated friction brake torque target. Brake Torque < 0Nm.
    */
    void setTorqueArbitrated(int32_t TorqueArbitrated) override;
    int32_t getTorqueArbitrated() const override;
    
    /**
    * TorqueFrictionMaximum Maximum wheel torque request for friction brake. Brake Torque < 0Nm.
    */
    void setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum) override;
    int32_t getTorqueFrictionMaximum() const override;
    
    /**
    * TorqueFrictionMinimum Minimum wheel torque request for friction brake. Brake Torque < 0Nm.
    */
    void setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum) override;
    int32_t getTorqueFrictionMinimum() const override;
    
    /**
    * OmegaUpper Upper wheel speed limit request controlled by friction brake. ISO 8855 wheel-spin velocity.
    */
    void setOmegaUpper(int32_t OmegaUpper) override;
    int32_t getOmegaUpper() const override;
    
    /**
    * OmegaLower Lower wheel speed limit request controlled by friction brake. ISO 8855 wheel-spin velocity.
    */
    void setOmegaLower(int32_t OmegaLower) override;
    int32_t getOmegaLower() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Left changes and signal emission.
    * @return The publisher for Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Left.
    */
    IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Left. */
    std::unique_ptr<IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Left. */
    VehicleMotionManagementBrakeAxleRow2WheelLeftData m_data;
};
} // namespace Vehicle
} // namespace Cpp
