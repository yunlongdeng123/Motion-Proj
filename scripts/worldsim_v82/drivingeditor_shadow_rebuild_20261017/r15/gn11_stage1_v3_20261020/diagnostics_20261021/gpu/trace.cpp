#include <torch/extension.h>
std::vector<torch::Tensor> trace_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("trace", &trace_cuda, "Read-only native-buffer sparse ray trace"); }
