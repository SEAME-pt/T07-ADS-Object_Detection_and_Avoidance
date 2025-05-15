
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_MotionManagement_Brake_Axle_Row1 implementation.
*/
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow1 : public IVehicleMotionManagementBrakeAxleRow1
{
public:
    explicit VehicleMotionManagementBrakeAxleRow1();
    ~VehicleMotionManagementBrakeAxleRow1();
public:
    /**
    * TorqueElectricMinimum Limit for regenerative brake torque at given axle. Brake Torque < 0Nm.
    */
    void setTorqueElectricMinimum(int32_t TorqueElectricMinimum) override;
    int32_t getTorqueElectricMinimum() const override;
    
    /**
    * TorqueFrictionDifferenceMaximum Maximum absolute wheel torque difference between left and right wheel for friction brake.
    */
    void setTorqueFrictionDifferenceMaximum(int32_t TorqueFrictionDifferenceMaximum) override;
    int32_t getTorqueFrictionDifferenceMaximum() const override;
    
    /**
    * TorqueDistributionFrictionRightMaximum Maximum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque shall be shifted to the left wheel. 50% = At most 50% friction torque may be shifted to the right wheel. 100% = Complete friction torque may be shifted to the right wheel.
    */
    void setTorqueDistributionFrictionRightMaximum(int32_t TorqueDistributionFrictionRightMaximum) override;
    int32_t getTorqueDistributionFrictionRightMaximum() const override;
    
    /**
    * TorqueDistributionFrictionRightMinimum Minimum distribution range of the friction brake request on the axle to the right wheel. 0% = Complete friction torque may be shifted to the left wheel. 50% = At least 50% friction torque shall be shifted to the right wheel. 100% = Complete friction torque shall be shifted to the right wheel.
    */
    void setTorqueDistributionFrictionRightMinimum(int32_t TorqueDistributionFrictionRightMinimum) override;
    int32_t getTorqueDistributionFrictionRightMinimum() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_MotionManagement_Brake_Axle_Row1 changes and signal emission.
    * @return The publisher for Vehicle_MotionManagement_Brake_Axle_Row1.
    */
    IVehicleMotionManagementBrakeAxleRow1Publisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_MotionManagement_Brake_Axle_Row1. */
    std::unique_ptr<IVehicleMotionManagementBrakeAxleRow1Publisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_MotionManagement_Brake_Axle_Row1. */
    VehicleMotionManagementBrakeAxleRow1Data m_data;
};
} // namespace Vehicle
} // namespace Cpp
