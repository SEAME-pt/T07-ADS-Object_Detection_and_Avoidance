
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertrain.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrain : public IVehiclePowertrain
{
public:
    explicit VehiclePowertrain();
    ~VehiclePowertrain();
public:
    /**
    * AccumulatedBrakingEnergy The accumulated energy from regenerative braking over lifetime.
    */
    void setAccumulatedBrakingEnergy(float AccumulatedBrakingEnergy) override;
    float getAccumulatedBrakingEnergy() const override;
    
    /**
    * Range Remaining range in meters using all energy sources available in the vehicle.
    */
    void setRange(int32_t Range) override;
    int32_t getRange() const override;
    
    /**
    * TimeRemaining Time remaining in seconds before all energy sources available in the vehicle are empty.
    */
    void setTimeRemaining(int32_t TimeRemaining) override;
    int32_t getTimeRemaining() const override;
    
    /**
    * Type Defines the powertrain type of the vehicle.
    */
    void setType(Vehicle_Powertrain_Type_ValueEnum Type) override;
    Vehicle_Powertrain_Type_ValueEnum getType() const override;
    
    /**
    * PowerOptimizeLevel Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    void setPowerOptimizeLevel(int32_t PowerOptimizeLevel) override;
    int32_t getPowerOptimizeLevel() const override;
    
    /**
    * IsAutoPowerOptimize Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    void setIsAutoPowerOptimize(bool IsAutoPowerOptimize) override;
    bool getIsAutoPowerOptimize() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain changes and signal emission.
    * @return The publisher for Vehicle_Powertrain.
    */
    IVehiclePowertrainPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain. */
    std::unique_ptr<IVehiclePowertrainPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain. */
    VehiclePowertrainData m_data;
};
} // namespace Vehicle
} // namespace Cpp
