# 原生材质与表面依赖

这里保存固定稀疏矩阵乘法、alpha×透射率重放、表面光栅化以及 C2 曲面求解所需源码。它们不读取场景号或车辆编号。同一份算子用于所有对象，数据清单只指定已安装算子的位置。

`surface_rasterizer/` 保留原许可证和 GLM 头文件许可。其 CUDA/C++ 数学代码与原先实际使用的算子一致，只把 Python 安装包名改为 `v44_surface_rasterization`。`surface_helpers/` 仅包含曲面求解实际调用的公共函数，没有旧场景默认资产。

在与原生引擎相容的 PyTorch、CUDA 和编译器环境中构建：

```bash
nvcc -O3 --shared -Xcompiler -fPIC v44_deterministic_csr.cu -o libv44_deterministic_csr.so
nvcc -O3 --shared -Xcompiler -fPIC v44_replay.cu -o libv44_replay.so
cd surface_rasterizer
python setup.py build_ext --inplace
```

在场景清单中登记绝对路径：

```json
{
  "material_backend": {
    "csr_library": "/installed/libv44_deterministic_csr.so",
    "replay_library": "/installed/libv44_replay.so",
    "raster_extension": "/installed/v44_surface_rasterization/_C.cpython-39-x86_64-linux-gnu.so"
  }
}
```

扩展文件后缀必须匹配实际 Python ABI。构建后先运行 `v44_material_probe`：它检查真实相机、前向 RGB/alpha、稀疏矩阵及材质梯度，再进入正式 1000 步拟合。不能把成功加载动态库当作数值兼容。

本轮实际 GPU 检查使用了已安装的原始兼容动态库；共享源码的数学一致性及 CPU 检查已经核对。重新命名后的全新安装尚未作为独立环境进行验收。
