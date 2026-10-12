#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mdetector_candidate {
struct FrameKey {
    std::string segment_sha256;
    std::int64_t raw_record_index=0, frame_timestamp_micros=0;
    bool operator==(const FrameKey& b) const {
        return segment_sha256==b.segment_sha256 && raw_record_index==b.raw_record_index
            && frame_timestamp_micros==b.frame_timestamp_micros;
    }
};
struct PointLineage {
    std::int64_t bin_row=0;
    int return_index=0, laser_name=0, range_row=0, range_col=0;
};
struct InputPoint {
    PointLineage id;
    std::array<float,3> body_xyz_m{};
    float intensity=0, elongation=0; // elongation carried in transport, unused by author
    std::int64_t top_range_flat_index=-1; // lineage only, never time or curvature
};
struct SynchronizedFrame {
    FrameKey cloud_key, pose_key;
    std::int64_t cloud_reference_micros=0, pose_reference_micros=0, origin_micros=0;
    std::array<double,16> body_to_world_row_major{};
    int top_range_height=0, top_range_width=0;
    std::vector<InputPoint> points;
    std::string coordinate_receipt, lineage_receipt, author_time_binding_receipt;
};
struct OutputPoint {
    PointLineage id;
    int origin_tag=0, cluster_tag=0, final_author_state=0, author_binary_label=9;
    enum Partition { Static, Dynamic, Invalid, Other } partition=Other;
};
struct Result { FrameKey key; std::vector<OutputPoint> points; };
class NativeBackend {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    // This constructor runs author init and allocates large native buffers.
    // The candidate has NO executable which invokes it during verification.
    NativeBackend();
    ~NativeBackend();
    Result process(const SynchronizedFrame& frame);
};
} // namespace mdetector_candidate
