#include "compat.hpp"
#include "author_waymo_parameters.hpp"
#include <iostream>
int main() {
    ros::NodeHandle nh(mdetector_candidate::author_waymo_parameters());
    int dataset=-1; double duration=0; float horizontal=0, vertical=0;
    bool coupled=false; std::string out_file="not empty";
    nh.param<int>("dyn_obj/dataset",dataset,0);
    nh.param<double>("dyn_obj/frame_dur",duration,0.2);
    // Exact author assignments: intentionally do not rename/correct these members.
    nh.param<float>("dyn_obj/ver_resolution_max",horizontal,0.0025f);
    nh.param<float>("dyn_obj/hor_resolution_max",vertical,0.0025f);
    nh.param<bool>("dyn_obj/cluster_coupled",coupled,false);
    nh.param<std::string>("dyn_obj/time_file",out_file,"");
    if (dataset!=2 || duration!=0.1 || !coupled || out_file!="") return 1;
    if (std::abs(horizontal-0.01f)>1e-8 || std::abs(vertical-0.005f)>1e-8) return 2;
    auto delta=ros::Time().fromSec(0.2)-ros::Time().fromSec(0.1);
    if (std::abs(delta.toSec()-0.1)>1e-10) return 3;
    std::cout << "transport parameter/time facade only: dataset=" << dataset
              << " actual author horizontal=" << horizontal << " vertical=" << vertical << '\n';
}
