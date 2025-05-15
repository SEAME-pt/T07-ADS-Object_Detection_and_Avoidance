
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrake.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_MotionManagement_Brake implementation.
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrake : public IVehicleMotionManagementBrake
{
public:
    explicit VehicleMotionManagementBrake();
    ~VehicleMotionManagementBrake();
public:
    /**
    * VehicleForceMaximum Maximum longitudinal brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    */
    void setVehicleForceMaximum(int32_t VehicleForceMaximum) override;
    int32_t getVehicleForceMaximum() const override;
    
    /**
    * VehicleForceElectricMinimumArbitrated Brake system internally calculated regenerative force limit at vehicle level for eAxle actuation. Brake Force < 0N.
    */
    void setVehicleForceElectricMinimumArbitrated(int32_t VehicleForceElectricMinimumArbitrated) override;
    int32_t getVehicleForceElectricMinimumArbitrated() const override;
    
    /**
    * VehicleForceElectric Regenerative brake force request (FxWhlSum). Sum of all tyre longitudinal forces. Brake Force < 0N.
    */
    void setVehicleForceElectric(int32_t VehicleForceElectric) override;
    int32_t getVehicleForceElectric() const override;
    
    /**
    * VehicleForceDistributionFrontMaximum Maximum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force shall be shifted to rear axle. 50% = At most 50% shall be shifted to front axle. 100%  = Complete longitudinal brake force may be shifted to front axle.
    */
    void setVehicleForceDistributionFrontMaximum(int32_t VehicleForceDistributionFrontMaximum) override;
    int32_t getVehicleForceDistributionFrontMaximum() const override;
    
    /**
    * VehicleForceDistributionFrontMinimum Minimum distribution range request of FxWhlSum to front axle. 0% = Complete longitudinal brake force may be shifted to rear axle. 50% = At least 50% shall be shifted to front axle. 100% = Complete longitudinal brake force shall be shifted to front axle.
    */
    void setVehicleForceDistributionFrontMinimum(int32_t VehicleForceDistributionFrontMinimum) override;
    int32_t getVehicleForceDistributionFrontMinimum() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake changes and signal emission.
    * @return The publisher for Vehicle_MotionManagement_Brake.
    */
    IVehicleMotionManagementBrakePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_MotionManagement_Brake. */
    std::unique_ptr<IVehicleMotionManagementBrakePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_MotionManagement_Brake. */
    VehicleMotionManagementBrakeData m_data;
};
} // namespace Vehicle
} // namespace Cpp
