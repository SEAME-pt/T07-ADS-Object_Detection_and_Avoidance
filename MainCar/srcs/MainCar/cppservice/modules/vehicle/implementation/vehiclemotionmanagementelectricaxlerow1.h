
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclemotionmanagementelectricaxlerow1.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_MotionManagement_ElectricAxle_Row1 implementation.
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementElectricAxleRow1 : public IVehicleMotionManagementElectricAxleRow1
{
public:
    explicit VehicleMotionManagementElectricAxleRow1();
    ~VehicleMotionManagementElectricAxleRow1();
public:
    /**
    * Torque Axle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    void setTorque(int32_t Torque) override;
    int32_t getTorque() const override;
    
    /**
    * TorqueTarget Target axle torque in torque control mode, positive sign for torque in forward direction, negative sign for torque in backward direction (ISO8855).
    */
    void setTorqueTarget(int32_t TorqueTarget) override;
    int32_t getTorqueTarget() const override;
    
    /**
    * TorqueMaximum Maximum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    void setTorqueMaximum(int32_t TorqueMaximum) override;
    int32_t getTorqueMaximum() const override;
    
    /**
    * TorqueMinimum Minimum momentarily available eAxle torque, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    void setTorqueMinimum(int32_t TorqueMinimum) override;
    int32_t getTorqueMinimum() const override;
    
    /**
    * TorqueLongTermMaximum Maximum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    void setTorqueLongTermMaximum(int32_t TorqueLongTermMaximum) override;
    int32_t getTorqueLongTermMaximum() const override;
    
    /**
    * TorqueLongTermMinimum Minimum long-term available eAxle torque, default time horizon = 10 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    void setTorqueLongTermMinimum(int32_t TorqueLongTermMinimum) override;
    int32_t getTorqueLongTermMinimum() const override;
    
    /**
    * TorqueShortTermMaximum Maximum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    void setTorqueShortTermMaximum(int32_t TorqueShortTermMaximum) override;
    int32_t getTorqueShortTermMaximum() const override;
    
    /**
    * TorqueShortTermMinimum Minimum short-term available eAxle torque, default time horizon = 1 sec, positive sign for torque in forward direction, negative sign for torque in backward direction.
    */
    void setTorqueShortTermMinimum(int32_t TorqueShortTermMinimum) override;
    int32_t getTorqueShortTermMinimum() const override;
    
    /**
    * TorqueMaximumLimit Maximum allowed eAxle torque in rotation speed control mode, positive sign for torque in forward direction, negative sign unused.
    */
    void setTorqueMaximumLimit(int32_t TorqueMaximumLimit) override;
    int32_t getTorqueMaximumLimit() const override;
    
    /**
    * TorqueMinimumLimit Minimum allowed axle torque in rotation speed control mode, positive sign unused, negative sign for torque in backward direction (ISO8855).
    */
    void setTorqueMinimumLimit(int32_t TorqueMinimumLimit) override;
    int32_t getTorqueMinimumLimit() const override;
    
    /**
    * RotationalSpeed Rotational speed for the specified axle, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    void setRotationalSpeed(int32_t RotationalSpeed) override;
    int32_t getRotationalSpeed() const override;
    
    /**
    * RotationalSpeedTarget Target axle rotational speed in rotation speed control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    void setRotationalSpeedTarget(int32_t RotationalSpeedTarget) override;
    int32_t getRotationalSpeedTarget() const override;
    
    /**
    * RotationalSpeedMaximumLimit Maximum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    void setRotationalSpeedMaximumLimit(int32_t RotationalSpeedMaximumLimit) override;
    int32_t getRotationalSpeedMaximumLimit() const override;
    
    /**
    * RotationalSpeedMinimumLimit Minimum allowed axle rotational speed in torque control mode, positive sign for rotation in forward direction, negative sign for rotation in backward direction.
    */
    void setRotationalSpeedMinimumLimit(int32_t RotationalSpeedMinimumLimit) override;
    int32_t getRotationalSpeedMinimumLimit() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_ElectricAxle_Row1 changes and signal emission.
    * @return The publisher for Vehicle_MotionManagement_ElectricAxle_Row1.
    */
    IVehicleMotionManagementElectricAxleRow1Publisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_MotionManagement_ElectricAxle_Row1. */
    std::unique_ptr<IVehicleMotionManagementElectricAxleRow1Publisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_MotionManagement_ElectricAxle_Row1. */
    VehicleMotionManagementElectricAxleRow1Data m_data;
};
} // namespace Vehicle
} // namespace Cpp
