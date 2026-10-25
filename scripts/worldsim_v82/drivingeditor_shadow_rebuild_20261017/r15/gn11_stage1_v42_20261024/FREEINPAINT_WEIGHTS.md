# V4.2 外部权重准备

用户本轮范围：先下载权重，GPU 就绪后再做实验。当前只准备 FreeInpaint 的 **object-removal** 路线，不启动推理、训练、模型评价或自动等待 GPU 的控制器。a37/a6 中心补全效果尚未验证。

![后续实验组件；本轮仅下载权重](pretrained_center_architecture.svg)

## 固定版本与文件范围

官方源码：`Rorisis/FreeInpaint@9ac03f14b652c8d03d65f770bdf61da253da0fef`。所有权重固定到官方 Hugging Face 仓库提交；远端通过 hf-mirror 直连获取字节，仍按官方公布的大小与哈希验收。下列依赖来自该源码的 `network/freeinpaint.py`、`guidance/inpainting.py`、`test.py` 的实际加载路径。

| 仓库 | 固定提交 | 下载范围 |
|---|---|---|
| jyproris/FreeInpaint | d9bf8acc084bd3fdcb3e82f01b6af9fd73e5b157 | 完整 FreeInpaint（含 DA3 骨干）、PowerPaint ppt-v1 UNet/文本编码器、说明 |
| stable-diffusion-v1-5/stable-diffusion-inpainting | 8a4288a76071f7280aedbdb3253bdb9e9d5d84bb | 参考补图入口需要的原精度 `.bin`；支持视图入口需要的 fp16 safetensors；配置、分词器、安全检查器 |
| stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | PowerPaint 使用的分词器与说明，不下载整套基础权重 |
| latent-consistency/lcm-lora-sdv1-5 | cf2fced511dbe7e26c8d1d397e728fbab875db4b | 支持视图使用的 LCM LoRA |
| h94/IP-Adapter | 018e402774aeeddd60609b4ecdb7e298259dc729 | SD1.5 adapter、CLIP 图像编码器及配置 |

合计 **36 文件、19,685,148,658 字节（19.69 GB / 18.33 GiB）**。原精度基础模型由官方参考图入口先载入，再加载 PowerPaint 权重，因此这里保留所需基础文件，不擅自改官方加载方法。`runwayml/stable-diffusion-inpainting` 已经重定向到上述 stable-diffusion-v1-5 仓库，后续配置兼容缓存别名，不再重复下载。

本轮使用已有车辆掩码，因此不下载 LangSAM/GroundingDINO/SAM 权重。SDXL/ControlNet 为官方 object-replacement 路线的依赖，不属于本轮删车任务。完整 benchmark/demo 数据、评价器模型也不属于此次推理权重集合。完整 FreeInpaint 已含 DA3，无须再下载 DA3 原版。

## 存放与校验

- 本地备用权重目录：`outputs/V42_Pretrained_Weights_20261024/`，相对当前 workspace。2026-10-24 23:55 SGT 全部36项大小与哈希验证通过，19,685,148,658字节；凭据为目录内 `download_receipt.json`，status=complete，inference_run=false。
- 远端主交付目录：`/root/autodl-tmp/models/worldsim_v82_v42_freeinpaint_20261024/`。官方源码已保存在其 `code/`，下载与一次签名重定向过期修复记录保留。远端最终36文件完整，19,685,148,658字节。
- 按用户要求复测镜像：远端直连 hf-mirror 的64 MiB样本平均1.97 MB/s；4文件、每文件最多8连接下载开始时合计约8.4 MiB/s，之后回落约1.3–2 MiB/s，不能将早期峰值当作稳定速度。代理路线仅约31 KB/s，故远端改为直连镜像并作为主交付位置。远端下载于 2026-10-25T02:35:02.682719+08:00 全部36项大小与官方哈希验收通过，下载凭据 status=complete；2026-10-26再次读取凭据并检查文件存在与大小。完整性不代表推理环境或GPU执行已经通过。初次短测速和代理超时记录均保留，不能把初次慢速结果泛化为镜像不可用。
- `download_plan.json` 固定每个文件的仓库、提交、相对路径、大小和预期哈希。大文件对照官方 LFS SHA256，小文本对照 Git blob SHA1；`download_receipt.json` 只有全部匹配才记录 `complete`。未完成的 `.aria2` 不计为通过。
- 远端入口为 `download_freeinpaint_local.py`（脚本名保留，支持Windows/Linux），设置 `V42_WEIGHTS_ROOT` 为上述远端目录、`V42_DOWNLOAD_ENDPOINT=https://hf-mirror.com`；Linux默认直连、不继承代理，下载后自动逐文件验收并退出。已有下载器活跃时不要重复启动。元数据缺失时从官方固定revision API获取，须能访问官方元数据；当前已缓存全部元数据。
- Windows断点续传需使用短目录别名，避免aria2控制文件触发路径长度限制：当前目录联接 `C:/Users/dengyunlong/.cache/v42-weights` 指向上述本地备用目录，设置 `V42_WEIGHTS_ROOT` 为该短路径后运行 `python work/v42/download_freeinpaint_local.py`。首轮未完成主权重的凭据保存在 `local_attempt1_receipt.json`；续传后全部通过，没有丢弃已下载字节。
- 完整性复查：`python work/v42/verify_freeinpaint_weights.py --root outputs/V42_Pretrained_Weights_20261024`。该操作只读原始字节，不反序列化 `.pth/.bin`，也不导入 PyTorch。

模型、缓存、下载日志和凭据不进入 Git；只提交下载/验证工具及本文档。GPU 就绪前不自动启动实验。

## GPU 就绪后的接线

远端镜像下载已完成；GPU接线前复核固定版本和下载凭据。若仅使用本地备用缓存，则先同步并按同一 verifier 复核全部文件。Linux 下载器会创建 `models--runwayml--stable-diffusion-inpainting` 指向 canonical HF 缓存目录的别名，并让 `code/checkpoints` 指向 FreeInpaint 固定 snapshot 的 `checkpoints`。未来运行应显式设置 `HF_HUB_CACHE=<根目录>/hub`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，防止意外更新或在线补权重。

离线接线还需把加载入口绑定到固定 snapshot 的本地路径；LoRA 指定 `pytorch_lora_weights.safetensors`，IP-Adapter 指定已下载的文件名。不要依赖未指定文件名时的在线仓库文件枚举。上述属于后续推理适配，当前没有宣称整套官方入口已通过离线加载。

官方环境要求 Python 3.12、PyTorch 2.9.1/CUDA 12.8、可编译 gsplat 的 CUDA toolkit；另建隔离环境，保持现有 drivingeditor 环境不变。当前没有安装或验证这套推理运行时，权重完整性不等于 GPU 推理通过。先适配 a37/a6 多视图输入与世界坐标，再做真实 GPU smoke；其中显存、保存/导出、评价、恢复与下一入口按实际实验形态逐项覆盖。文件大小不是显存峰值。

生成中心仍属带来源与置信度的伪监督；固定 V41-B master 继续独立，只有编辑副本接入生成道路与已定位静态污染抑制。`quality_accepted=false`，`human_verdict=null`。
