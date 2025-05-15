
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleVersionVssThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_VersionVSS interface.
* @see Vehicle_VersionVSS
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleVersionVss> testVehicleVersionVss = std::make_unique<VehicleVersionVssThreadSafeDecorator>(std::make_shared<VehicleVersionVss>());

// Thread safe access
auto major = testVehicleVersionVss->getMajor();
testVehicleVersionVss->setMajor(0);
auto minor = testVehicleVersionVss->getMinor();
testVehicleVersionVss->setMinor(0);
auto patch = testVehicleVersionVss->getPatch();
testVehicleVersionVss->setPatch(0);
auto label = testVehicleVersionVss->getLabel();
testVehicleVersionVss->setLabel(std::string());
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleVersionVssThreadSafeDecorator : public IVehicleVersionVss
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_VersionVSS object to make thread safe.
    */
    explicit VehicleVersionVssThreadSafeDecorator(std::shared_ptr<IVehicleVersionVss> impl);

    /** Guards and forwards call to Vehicle_VersionVSS implementation. */
    void setMajor(int32_t Major) override;
    /** Guards and forwards call to Vehicle_VersionVSS implementation. */
    int32_t getMajor() const override;

    /** Guards and forwards call to Vehicle_VersionVSS implementation. */
    void setMinor(int32_t Minor) override;
    /** Guards and forwards call to Vehicle_VersionVSS implementation. */
    int32_t getMinor() const override;

    /** Guards and forwards call to Vehicle_VersionVSS implementation. */
    void setPatch(int32_t Patch) override;
    /** Guards and forwards call to Vehicle_VersionVSS implementation. */
    int32_t getPatch() const override;

    /** Guards and forwards call to Vehicle_VersionVSS implementation. */
    void setLabel(const std::string& Label) override;
    /** Guards and forwards call to Vehicle_VersionVSS implementation. */
    const std::string& getLabel() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_VersionVSS changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_VersionVSS.
    */
    IVehicleVersionVssPublisher& _getPublisher() const override;
private:
    /** The Vehicle_VersionVSS object which is guarded */
    std::shared_ptr<IVehicleVersionVss> m_impl;
    // Mutex for major property
    mutable std::shared_timed_mutex m_majorMutex;
    // Mutex for minor property
    mutable std::shared_timed_mutex m_minorMutex;
    // Mutex for patch property
    mutable std::shared_timed_mutex m_patchMutex;
    // Mutex for label property
    mutable std::shared_timed_mutex m_labelMutex;
};
} // namespace Vehicle
} // namespace Cpp
