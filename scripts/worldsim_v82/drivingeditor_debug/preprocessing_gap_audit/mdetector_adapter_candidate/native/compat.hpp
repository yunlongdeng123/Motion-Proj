#pragma once
// Candidate-only ROS transport facade. No ROS master, subscribers, GPU or math.
// Geometry storage remains double precision; author algorithms use the fields.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ros {
struct Duration {
    double seconds = 0;
    double toSec() const { return seconds; }
};
struct Time {
    double seconds = 0;
    Time& fromSec(double value) { seconds = value; return *this; }
    double toSec() const { return seconds; }
    static Time now() {
        Time t;
        t.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return t;
    }
};
inline Duration operator-(const Time& a, const Time& b) { return {a.seconds-b.seconds}; }
struct Publisher {
    template<class Message> void publish(const Message&) const {} // transport only
};
using Parameter = std::variant<bool, int, double, std::string>;
using ParameterMap = std::unordered_map<std::string, Parameter>;
class NodeHandle {
    ParameterMap values_;
public:
    explicit NodeHandle(ParameterMap values) : values_(std::move(values)) {}
    template<class T> void param(const std::string& key, T& destination, const T& fallback) {
        auto it = values_.find(key);
        if (it == values_.end()) { destination = fallback; return; }
        std::visit([&](const auto& value) {
            using V = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T,std::string> && std::is_same_v<V,std::string>)
                destination = value;
            else if constexpr (std::is_arithmetic_v<T> && std::is_arithmetic_v<V>)
                destination = static_cast<T>(value);
            else throw std::runtime_error("parameter type mismatch: " + key);
        }, it->second);
    }
};
} // namespace ros

namespace std_msgs {
struct Header { std::uint32_t seq = 0; ros::Time stamp; std::string frame_id; };
}
namespace geometry_msgs {
struct Point { double x=0, y=0, z=0; };
struct Quaternion { double x=0, y=0, z=0, w=0; };
struct Pose { Point position; Quaternion orientation; };
struct PoseWithCovariance { Pose pose; std::array<double,36> covariance{}; };
struct PoseWithCovarianceStamped { std_msgs::Header header; PoseWithCovariance pose; };
}
namespace sensor_msgs {
struct PointCloud2 { std_msgs::Header header; };
struct CompressedImage {};
}
namespace nav_msgs { struct Path {}; struct Odometry {}; }
namespace visualization_msgs {
struct Marker {};
struct MarkerArray { std::vector<Marker> markers; };
}
namespace tf {} // included transport header; no tf math is called by the core
namespace pcl {
template<class Cloud> inline void toROSMsg(const Cloud&, sensor_msgs::PointCloud2&) {}
}
namespace cv {} // fixed source includes OpenCV headers but uses no OpenCV symbols
