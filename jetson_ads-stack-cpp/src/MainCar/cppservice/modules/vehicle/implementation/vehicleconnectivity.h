
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehicleconnectivity.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Connectivity implementation.
*/
class CPP_VEHICLE_EXPORT VehicleConnectivity : public IVehicleConnectivity
{
public:
    explicit VehicleConnectivity();
    ~VehicleConnectivity();
public:
    /**
    * IsConnectivityAvailable Indicates if connectivity between vehicle and cloud is available. True = Connectivity is available. False = Connectivity is not available.
    */
    void setIsConnectivityAvailable(bool IsConnectivityAvailable) override;
    bool getIsConnectivityAvailable() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Connectivity changes and signal emission.
    * @return The publisher for Vehicle_Connectivity.
    */
    IVehicleConnectivityPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Connectivity. */
    std::unique_ptr<IVehicleConnectivityPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Connectivity. */
    VehicleConnectivityData m_data;
};
} // namespace Vehicle
} // namespace Cpp
