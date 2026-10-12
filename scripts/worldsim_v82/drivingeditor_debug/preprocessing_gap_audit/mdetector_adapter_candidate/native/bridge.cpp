#include "prelude.hpp"
#include "bridge.hpp"
#include "author_waymo_parameters.hpp"
#include <m-detector/DynObjFilter.h>
#include <tuple>

namespace mdetector_candidate {
struct NativeBackend::Impl {
    DynObjFilter filter;
    bool processed=false;
    FrameKey previous;
    std::int64_t previous_reference=0, time_origin=0;
    Impl() {
        ros::NodeHandle parameters(author_waymo_parameters());
        filter.init(parameters); // all defaults/thresholds remain the author's code
    }
};
NativeBackend::NativeBackend() : impl_(new Impl) {}
NativeBackend::~NativeBackend() = default;

Result NativeBackend::process(const SynchronizedFrame& f) {
    if (!(f.cloud_key==f.pose_key) || f.cloud_reference_micros!=f.pose_reference_micros)
        throw std::invalid_argument("cloud/pose exact frame and reference stamp mismatch");
    if (f.coordinate_receipt.empty() || f.lineage_receipt.empty() || f.author_time_binding_receipt.empty())
        throw std::invalid_argument("unresolved coordinate/lineage/scan-time contract");
    if (f.points.empty() || f.points.size()>static_cast<std::size_t>(impl_->filter.points_num_perframe))
        throw std::invalid_argument("author capacity exceeded or empty cloud; no truncation");
    if (f.cloud_reference_micros < f.origin_micros)
        throw std::invalid_argument("negative segment-relative reference time");
    if (f.top_range_height<=0 || f.top_range_width<=0)
        throw std::invalid_argument("native TOP range-image H/W required");
    if (impl_->processed && (f.cloud_key.segment_sha256!=impl_->previous.segment_sha256
        || f.cloud_key.raw_record_index!=impl_->previous.raw_record_index+1
        || f.cloud_key.frame_timestamp_micros<=impl_->previous.frame_timestamp_micros
        || f.cloud_reference_micros<=impl_->previous_reference || f.origin_micros!=impl_->time_origin))
        throw std::invalid_argument("native backend requires one contiguous chronological segment");
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    cloud->reserve(f.points.size());
    std::tuple<int,int,int,int> previous_id{-1,-1,-1,-1};
    for (std::size_t i=0; i<f.points.size(); ++i) {
        const auto& source=f.points[i];
        const auto& id=source.id;
        auto key=std::make_tuple(id.return_index,id.laser_name,id.range_row,id.range_col);
        if (id.bin_row!=static_cast<std::int64_t>(i) || key<=previous_id
            || id.return_index<0 || id.return_index>1 || id.laser_name<1 || id.laser_name>5
            || id.range_row<0 || id.range_col<0 || (id.laser_name!=1 && source.top_range_flat_index!=-1))
            throw std::invalid_argument("raw point identity/order mismatch");
        previous_id=key;
        if (id.laser_name==1 && (id.range_row>=f.top_range_height || id.range_col>=f.top_range_width
            || source.top_range_flat_index!=(static_cast<std::int64_t>(id.return_index)*f.top_range_height
                +id.range_row)*f.top_range_width+id.range_col))
            throw std::invalid_argument("TOP return offset/pixel flat-index mismatch");
        PointType point{};
        point.x=source.body_xyz_m[0]; point.y=source.body_xyz_m[1]; point.z=source.body_xyz_m[2];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)
            || !std::isfinite(source.intensity) || !std::isfinite(source.elongation))
            throw std::invalid_argument("non-finite native point");
        point.intensity=source.intensity;
        point.curvature=0; // author uses curvature as KITTI distortion marker only, dataset=2
        cloud->push_back(point);
    }
    M3D rotation; V3D position;
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) rotation(i,j)=f.body_to_world_row_major[4*i+j];
        position(i)=f.body_to_world_row_major[4*i+3];
    }
    for (int i=0; i<16; ++i)
        if (!std::isfinite(f.body_to_world_row_major[i]))
            throw std::invalid_argument("non-finite pose");
    for (int i=0; i<4; ++i)
        if (std::abs(f.body_to_world_row_major[12+i]-(i==3 ? 1.0 : 0.0))>1e-8)
            throw std::invalid_argument("invalid homogeneous transform row");
    // Input-format tolerances only; none of the author detector thresholds change.
    if (!(rotation.transpose()*rotation).isApprox(M3D::Identity(),1e-6)
        || std::abs(rotation.determinant()-1.0)>1e-6)
        throw std::invalid_argument("pose is not an SE(3) transform");
    const double scan_time=static_cast<double>(f.cloud_reference_micros-f.origin_micros)/1e6;
    const int active_slot=impl_->filter.cur_point_soph_pointers;
    impl_->filter.filter(cloud, rotation, position, scan_time);
    // Read native label vectors by the exact input slot, never compact output clouds.
    const auto& event=impl_->filter.dyn_tag_origin;
    const auto& cluster=impl_->filter.dyn_tag_cluster;
    if (event.size()!=f.points.size() || cluster.size()!=f.points.size())
        throw std::runtime_error("author output vector/index correspondence broken");
    Result result; result.key=f.cloud_key; result.points.reserve(f.points.size());
    const auto* states=impl_->filter.point_soph_pointers[active_slot];
    for (std::size_t i=0; i<f.points.size(); ++i) {
        if (event[i]<0 || event[i]>2 || cluster[i]<-1 || cluster[i]>1
            || static_cast<int>(states[i].dyn)<0 || static_cast<int>(states[i].dyn)>6)
            throw std::runtime_error("unknown author tag/state; no guessed mapping");
        OutputPoint out;
        out.id=f.points[i].id; out.origin_tag=event[i]; out.cluster_tag=cluster[i];
        out.final_author_state=static_cast<int>(states[i].dyn);
        out.author_binary_label=cluster[i]==1 ? 251 : 9;
        if (states[i].dyn==INVALID || cluster[i]==-1) out.partition=OutputPoint::Invalid;
        else if (cluster[i]==1) out.partition=OutputPoint::Dynamic;
        else if (states[i].dyn==STATIC) out.partition=OutputPoint::Static;
        else out.partition=OutputPoint::Other;
        result.points.push_back(out);
    }
    impl_->processed=true; impl_->previous=f.cloud_key;
    impl_->previous_reference=f.cloud_reference_micros; impl_->time_origin=f.origin_micros;
    return result;
}
} // namespace mdetector_candidate
