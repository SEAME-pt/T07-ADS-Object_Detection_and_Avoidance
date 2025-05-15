
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleDriverIdentifierDeprecatedThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Driver_Identifier_Deprecated interface.
* @see Vehicle_Driver_Identifier_Deprecated
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleDriverIdentifierDeprecated> testVehicleDriverIdentifierDeprecated = std::make_unique<VehicleDriverIdentifierDeprecatedThreadSafeDecorator>(std::make_shared<VehicleDriverIdentifierDeprecated>());

// Thread safe access
auto subjectDeprecated = testVehicleDriverIdentifierDeprecated->getSubjectDeprecated();
testVehicleDriverIdentifierDeprecated->setSubjectDeprecated(std::string());
auto issuerDeprecated = testVehicleDriverIdentifierDeprecated->getIssuerDeprecated();
testVehicleDriverIdentifierDeprecated->setIssuerDeprecated(std::string());
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleDriverIdentifierDeprecatedThreadSafeDecorator : public IVehicleDriverIdentifierDeprecated
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Driver_Identifier_Deprecated object to make thread safe.
    */
    explicit VehicleDriverIdentifierDeprecatedThreadSafeDecorator(std::shared_ptr<IVehicleDriverIdentifierDeprecated> impl);

    /** Guards and forwards call to Vehicle_Driver_Identifier_Deprecated implementation. */
    void setSubjectDeprecated(const std::string& Subject_Deprecated) override;
    /** Guards and forwards call to Vehicle_Driver_Identifier_Deprecated implementation. */
    const std::string& getSubjectDeprecated() const override;

    /** Guards and forwards call to Vehicle_Driver_Identifier_Deprecated implementation. */
    void setIssuerDeprecated(const std::string& Issuer_Deprecated) override;
    /** Guards and forwards call to Vehicle_Driver_Identifier_Deprecated implementation. */
    const std::string& getIssuerDeprecated() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Driver_Identifier_Deprecated changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Driver_Identifier_Deprecated.
    */
    IVehicleDriverIdentifierDeprecatedPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Driver_Identifier_Deprecated object which is guarded */
    std::shared_ptr<IVehicleDriverIdentifierDeprecated> m_impl;
    // Mutex for subjectDeprecated property
    mutable std::shared_timed_mutex m_subjectDeprecatedMutex;
    // Mutex for issuerDeprecated property
    mutable std::shared_timed_mutex m_issuerDeprecatedMutex;
};
} // namespace Vehicle
} // namespace Cpp
