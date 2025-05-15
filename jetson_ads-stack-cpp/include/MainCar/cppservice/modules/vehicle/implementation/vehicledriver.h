
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehicledriver.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Driver implementation.
*/
class CPP_VEHICLE_EXPORT VehicleDriver : public IVehicleDriver
{
public:
    explicit VehicleDriver();
    ~VehicleDriver();
public:
    /**
    * DistractionLevel Distraction level of the driver, which can be evaluated by multiple factors e.g. driving situation, acoustical or optical signals inside the cockpit, ongoing phone calls.
    */
    void setDistractionLevel(float DistractionLevel) override;
    float getDistractionLevel() const override;
    
    /**
    * IsEyesOnRoad Has driver the eyes on road or not?
    */
    void setIsEyesOnRoad(bool IsEyesOnRoad) override;
    bool getIsEyesOnRoad() const override;
    
    /**
    * IsHandsOnWheel Are the driver's hands on the steering wheel or not?
    */
    void setIsHandsOnWheel(bool IsHandsOnWheel) override;
    bool getIsHandsOnWheel() const override;
    
    /**
    * AttentiveProbability Probability of attentiveness of the driver.
    */
    void setAttentiveProbability(float AttentiveProbability) override;
    float getAttentiveProbability() const override;
    
    /**
    * FatigueLevel Fatigue level of the driver, which can be evaluated by multiple factors e.g. trip time, behaviour of steering, eye status.
    */
    void setFatigueLevel(float FatigueLevel) override;
    float getFatigueLevel() const override;
    
    /**
    * HeartRate Heart rate of the driver.
    */
    void setHeartRate(int32_t HeartRate) override;
    int32_t getHeartRate() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Driver changes and signal emission.
    * @return The publisher for Vehicle_Driver.
    */
    IVehicleDriverPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Driver. */
    std::unique_ptr<IVehicleDriverPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Driver. */
    VehicleDriverData m_data;
};
} // namespace Vehicle
} // namespace Cpp
