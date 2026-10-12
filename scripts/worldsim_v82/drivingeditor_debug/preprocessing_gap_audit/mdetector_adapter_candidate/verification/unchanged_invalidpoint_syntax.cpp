#include <cmath>
using std::pow; using std::fabs;
struct V3D { double operator()(int) const; };
struct DynObjFilter { double blind_dis; int dataset; bool InvalidPointCheck(const V3D&, const int); };
bool  DynObjFilter::InvalidPointCheck(const V3D &body, const int intensity)
{
    if ((pow(body(0), 2) + pow(body(1), 2) + pow(body(2), 2)) < blind_dis*blind_dis || (dataset == 1 && fabs(body(0)) < 0.1 && fabs(body(1)) < 1.0) && fabs(body(2)) < 0.1)
    {
        return true;
    } 
    else
    {
        return false;
    }
}

