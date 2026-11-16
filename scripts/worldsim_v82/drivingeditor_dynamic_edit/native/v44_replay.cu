#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 128;
constexpr int kTileWidth = 16;
constexpr int kTileHeight = 16;
constexpr int kFeatureStride = 24;
constexpr int kColorChannels = 3;
constexpr int kOutputFeatures = 6;
constexpr int kRaySamples = 3;
constexpr float kMinAlpha = 1.0f / 255.0f;
constexpr float kMinTransmittance = 0.0001f;
constexpr float kSqrtTwo = 1.4142135623730951f;
constexpr float kInvSqrtTwo = 0.7071067811865475f;

struct RayProfile {
    float mean;
    float sigma;
};

struct ClipProfile {
    RayProfile ray;
    float normal_ray;
    float retained;
};

__device__ RayProfile ray_profile(const float* terms, float x, float y)
{
    const float px = fmaf(terms[0], x, fmaf(terms[1], y, terms[2]));
    const float py = fmaf(terms[1], x, fmaf(terms[3], y, terms[4]));
    const float pz = fmaf(terms[2], x, fmaf(terms[4], y, terms[5]));
    const float precision = fmaxf(fmaf(x, px, fmaf(y, py, pz)), 1e-12f);
    const float mean = fmaf(x, terms[6], fmaf(y, terms[7], terms[8])) / precision;
    return {mean, rsqrtf(precision)};
}

__device__ ClipProfile clip_profile(const float* q, float pix_x, float pix_y)
{
    const float x = fmaf(q[14], pix_x, q[15]);
    const float y = fmaf(q[16], pix_y, q[17]);
    const RayProfile profile = ray_profile(q + 1, x, y);
    const float normal_ray = fmaf(x, q[10], fmaf(y, q[11], q[12]));
    const float side = fmaf(normal_ray, profile.mean, -q[13]);
    const float deviation = fabsf(normal_ray) * profile.sigma;
    const float retained = deviation > 1e-10f
        ? 0.5f * erfcf(-kInvSqrtTwo * side / deviation)
        : (side >= 0.0f ? 1.0f : 0.0f);
    return {profile, normal_ray, retained};
}

__device__ bool uses_ray_order(int mode, int kind)
{
    return kind != 0 && (mode == 2 || mode == 4 || kind == 1);
}

__device__ float ordering_depth(
    float center, float max_shift, const RayProfile& profile,
    bool clipped, const ClipProfile& clip, float quantile)
{
    bool far_side = quantile > 0.5f;
    float tail = far_side ? 1.0f - quantile : quantile;
    if (clipped && clip.retained < 1.0f) {
        far_side = clip.normal_ray > 0.0f;
        tail = (far_side ? 1.0f - quantile : quantile) * clip.retained;
    }
    const float displacement = profile.sigma * kSqrtTwo * erfcinvf(2.0f * tail);
    const float depth = profile.mean + (far_side ? displacement : -displacement);
    if (!isfinite(depth) || depth <= 0.0f) {
        return center;
    }
    return fminf(fmaxf(depth, center - max_shift), center + max_shift);
}

template <bool Gather>
__global__ void scan_hits(
    int pixels, int width, int tiles_x, const int* pixel_ids, const int* ranges,
    const int* row_ids, const float* xy, const float* conic, const float* depths,
    const float* features, const int* kind, const float* terms, const float* ray,
    int mode, float max_shift, int* counts, const int* offsets,
    long long* sort_keys, int* hit_ids, float* alphas)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= pixels) {
        return;
    }
    const int pixel = pixel_ids[slot];
    const int x = pixel % width;
    const int y = pixel / width;
    const float pix_x = static_cast<float>(x);
    const float pix_y = static_cast<float>(y);
    const int tile = (y / kTileHeight) * tiles_x + x / kTileWidth;
    const int begin = ranges[2 * tile];
    const int end = ranges[2 * tile + 1];
    int hit = Gather ? offsets[slot] : 0;

    for (int entry = begin; entry < end; ++entry) {
        const int row = row_ids[entry];
        const float dx = xy[2 * row] - pix_x;
        const float dy = xy[2 * row + 1] - pix_y;
        const float* con = conic + 4 * row;
        const float power = -0.5f * (con[0] * dx * dx + con[2] * dy * dy)
            - con[1] * dx * dy;
        if (power > 0.0f) {
            continue;
        }
        float alpha = fminf(0.99f, con[3] * expf(power));
        if (alpha < kMinAlpha) {
            continue;
        }
        const float* q = features + row * kFeatureStride + kOutputFeatures;
        const bool clipped = q[0] > 0.5f;
        ClipProfile clip = {};
        if (clipped) {
            clip = clip_profile(q, pix_x, pix_y);
            alpha *= clip.retained;
            if (alpha < kMinAlpha) {
                continue;
            }
        }
        const bool ray_order = mode != 0 && (Gather || mode >= 3) && uses_ray_order(mode, kind[row]);
        const int samples = ray_order && mode >= 3 ? kRaySamples : 1;
        if (Gather) {
            const float center = depths[row];
            RayProfile profile = clip.ray;
            if (ray_order && !clipped) {
                const float ray_x = fmaf(ray[0], pix_x, ray[1]);
                const float ray_y = fmaf(ray[2], pix_y, ray[3]);
                profile = ray_profile(terms + 9 * row, ray_x, ray_y);
            }
            const float slice_alpha = samples == 1 ? alpha : -expm1f(log1pf(-alpha) / samples);
            for (int i = 0; i < samples; ++i) {
                const float quantile = (i + 0.5f) / samples;
                const float depth = ray_order
                    ? ordering_depth(center, max_shift, profile, clipped, clip, quantile)
                    : center;
                // Sorting stably by this key keeps the native tile order at ties.
                sort_keys[hit + i] = (static_cast<long long>(slot) << 32) | __float_as_uint(depth);
                hit_ids[hit + i] = row;
                alphas[hit + i] = slice_alpha;
            }
        }
        hit += samples;
    }
    if (!Gather) {
        counts[slot] = hit;
    }
}

__global__ void compose(
    int pixels, const int* offsets, const long long* permutation, const int* hit_ids,
    const float* alphas, const float* colors, const float* features,
    const float* background, float* result)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= pixels) {
        return;
    }
    float transmittance = 1.0f;
    float rgb[kColorChannels] = {};
    float auxiliary[kOutputFeatures] = {};
    for (int index = offsets[slot]; index < offsets[slot + 1]; ++index) {
        const long long hit = permutation[index];
        const float alpha = alphas[hit];
        const float test_t = transmittance * (1.0f - alpha);
        if (test_t < kMinTransmittance) {
            break;
        }
        const int row = hit_ids[hit];
        const float weight = alpha * transmittance;
        #pragma unroll
        for (int ch = 0; ch < kColorChannels; ++ch) {
            rgb[ch] += colors[row * kColorChannels + ch] * weight;
        }
        #pragma unroll
        for (int ch = 0; ch < kOutputFeatures; ++ch) {
            auxiliary[ch] += features[row * kFeatureStride + ch] * weight;
        }
        transmittance = test_t;
    }

    float* output = result + slot * (kColorChannels + kOutputFeatures);
    #pragma unroll
    for (int ch = 0; ch < kColorChannels; ++ch) {
        output[ch] = rgb[ch] + transmittance * background[ch];
    }
    #pragma unroll
    for (int ch = 0; ch < kOutputFeatures; ++ch) {
        output[kColorChannels + ch] = auxiliary[ch];
    }
}

} // namespace

extern "C" int count_pass(
    int pixels, int width, int tiles_x, const int* pixel_ids, const int* ranges,
    const int* row_ids, const float* xy, const float* conic, const float* depths,
    const float* features, const int* kind, const float* terms, const float* ray,
    int mode, float max_shift, int* counts, void* stream)
{
    if (pixels == 0) {
        return cudaSuccess;
    }
    scan_hits<false><<<(pixels - 1) / kThreads + 1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        pixels, width, tiles_x, pixel_ids, ranges, row_ids, xy, conic, depths,
        features, kind, terms, ray, mode, max_shift, counts, nullptr, nullptr, nullptr, nullptr);
    return cudaGetLastError();
}

extern "C" int gather_pass(
    int pixels, int width, int tiles_x, const int* pixel_ids, const int* ranges,
    const int* row_ids, const float* xy, const float* conic, const float* depths,
    const float* features, const int* kind, const float* terms, const float* ray,
    int mode, float max_shift, const int* offsets, long long* sort_keys,
    int* hit_ids, float* alphas, void* stream)
{
    if (pixels == 0) {
        return cudaSuccess;
    }
    scan_hits<true><<<(pixels - 1) / kThreads + 1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        pixels, width, tiles_x, pixel_ids, ranges, row_ids, xy, conic, depths,
        features, kind, terms, ray, mode, max_shift, nullptr, offsets, sort_keys, hit_ids, alphas);
    return cudaGetLastError();
}

extern "C" int compose_pass(
    int pixels, const int* offsets, const long long* permutation, const int* hit_ids,
    const float* alphas, const float* colors, const float* features,
    const float* background, float* result, void* stream)
{
    if (pixels == 0) {
        return cudaSuccess;
    }
    compose<<<(pixels - 1) / kThreads + 1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        pixels, offsets, permutation, hit_ids, alphas, colors, features, background, result);
    return cudaGetLastError();
}
