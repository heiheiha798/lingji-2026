#include "field_generator.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace icecarver {
namespace {

constexpr float kPi = 3.14159265358979323846f;

__device__ __forceinline__ std::uint32_t Mix32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ float UnitHash(std::uint64_t seed,
                                           std::uint64_t index,
                                           std::uint32_t lane) {
  const std::uint32_t lo = static_cast<std::uint32_t>(seed);
  const std::uint32_t hi = static_cast<std::uint32_t>(seed >> 32);
  const std::uint32_t i0 = static_cast<std::uint32_t>(index);
  const std::uint32_t i1 = static_cast<std::uint32_t>(index >> 32);
  const std::uint32_t bits =
      Mix32(lo ^ Mix32(hi + 0x9e3779b9u * (lane + 1u)) ^ Mix32(i0) ^
            Mix32(i1 + 0x85ebca6bu));
  return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float SeedParameter(std::uint64_t seed,
                                                std::uint32_t lane) {
  return UnitHash(seed, 0x243f6a8885a308d3ull, lane);
}

__device__ __forceinline__ float SdfSphere(float x, float y, float z,
                                            float cx, float cy, float cz,
                                            float radius) {
  const float dx = x - cx;
  const float dy = y - cy;
  const float dz = z - cz;
  return sqrtf(dx * dx + dy * dy + dz * dz) - radius;
}

__device__ __forceinline__ float SdfTorus(float x, float y, float z,
                                           float major_radius,
                                           float minor_radius) {
  const float radial = sqrtf(x * x + y * y) - major_radius;
  return sqrtf(radial * radial + z * z) - minor_radius;
}

__device__ float EvaluateField(FieldKind field, float x, float y, float z,
                               std::uint64_t seed, std::uint64_t linear_id) {
  const float p0 = SeedParameter(seed, 0);
  const float p1 = SeedParameter(seed, 1);
  const float p2 = SeedParameter(seed, 2);
  const float phase0 = 2.0f * kPi * SeedParameter(seed, 3);
  const float phase1 = 2.0f * kPi * SeedParameter(seed, 4);
  const float phase2 = 2.0f * kPi * SeedParameter(seed, 5);
  float value = 0.0f;

  switch (field) {
    case FieldKind::kSphere: {
      const float cx = 0.16f * (p0 - 0.5f);
      const float cy = 0.16f * (p1 - 0.5f);
      const float cz = 0.16f * (p2 - 0.5f);
      const float radius = 0.53f + 0.08f * (SeedParameter(seed, 6) - 0.5f);
      value = SdfSphere(x, y, z, cx, cy, cz, radius);
      break;
    }
    case FieldKind::kMetaball: {
      float potential = 0.0f;
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const float cx = 1.25f * (SeedParameter(seed, 10 + i * 4) - 0.5f);
        const float cy = 1.25f * (SeedParameter(seed, 11 + i * 4) - 0.5f);
        const float cz = 1.25f * (SeedParameter(seed, 12 + i * 4) - 0.5f);
        const float strength =
            0.10f + 0.07f * SeedParameter(seed, 13 + i * 4);
        const float dx = x - cx;
        const float dy = y - cy;
        const float dz = z - cz;
        potential += strength / (dx * dx + dy * dy + dz * dz + 0.055f);
      }
      value = 1.0f - potential;
      break;
    }
    case FieldKind::kGyroid: {
      const float frequency = 2.5f + 0.5f * p0;
      const float ax = frequency * kPi * x + phase0;
      const float ay = frequency * kPi * y + phase1;
      const float az = frequency * kPi * z + phase2;
      value = sinf(ax) * cosf(ay) + sinf(ay) * cosf(az) +
              sinf(az) * cosf(ax);
      break;
    }
    case FieldKind::kMixed: {
      const float sphere =
          SdfSphere(x, y, z, 0.22f * (p0 - 0.5f),
                    0.22f * (p1 - 0.5f), 0.18f * (p2 - 0.5f), 0.58f);
      const float torus =
          SdfTorus(x + 0.18f, y - 0.10f, z + 0.08f, 0.48f, 0.16f);
      const float waves = 0.10f *
                          (sinf(5.0f * x + phase0) *
                               sinf(4.0f * y + phase1) +
                           sinf(4.5f * z + phase2));
      value = fminf(sphere, torus) + waves;
      break;
    }
    case FieldKind::kMultiscale: {
      value = 0.64f * sinf(2.0f * kPi * x + phase0) *
                  cosf(2.0f * kPi * y - phase1) +
              0.38f * sinf(4.5f * kPi * y + phase1) *
                  cosf(3.5f * kPi * z + phase2) +
              0.20f * sinf(9.0f * kPi * (x + y + z) + phase2) +
              0.12f * cosf(15.0f * kPi * (x - z) - phase0);
      break;
    }
    case FieldKind::kDense: {
      value = 0.52f * sinf(7.0f * kPi * x + phase0) +
              0.49f * sinf(8.0f * kPi * y + phase1) +
              0.46f * sinf(9.0f * kPi * z + phase2) +
              0.31f * sinf(5.0f * kPi * (x + y - z) - phase0) +
              0.18f * cosf(13.0f * kPi * (x - y) + phase1);
      break;
    }
  }

  // A deterministic perturbation prevents grid vertices from landing exactly
  // on an isovalue and ensures every timing variant has distinct input data.
  value += 2.0e-4f * (UnitHash(seed, linear_id, 31) - 0.5f);
  return value;
}

__global__ void GenerateFieldKernel(FieldKind field, int nx, int ny, int nz,
                                    std::uint64_t seed, float* volume,
                                    std::uint64_t sample_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= sample_count) {
    return;
  }

  const int ix = static_cast<int>(index % static_cast<std::uint64_t>(nx));
  const std::uint64_t yz = index / static_cast<std::uint64_t>(nx);
  const int iy = static_cast<int>(yz % static_cast<std::uint64_t>(ny));
  const int iz = static_cast<int>(yz / static_cast<std::uint64_t>(ny));
  const float x = 2.0f * static_cast<float>(ix) / static_cast<float>(nx - 1) -
                  1.0f;
  const float y = 2.0f * static_cast<float>(iy) / static_cast<float>(ny - 1) -
                  1.0f;
  const float z = 2.0f * static_cast<float>(iz) / static_cast<float>(nz - 1) -
                  1.0f;
  volume[index] = EvaluateField(field, x, y, z, seed, index);
}

}  // namespace

std::uint64_t DeriveVariantSeed(std::uint64_t case_seed,
                                std::uint64_t variant_index) {
  std::uint64_t z =
      case_seed + 0x9e3779b97f4a7c15ull * (variant_index + 1ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

cudaError_t GenerateField(const CaseConfig& config, std::uint64_t variant_seed,
                          float* device_volume, cudaStream_t stream) {
  if (device_volume == nullptr || config.nx < 2 || config.ny < 2 ||
      config.nz < 2) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t sample_count =
      static_cast<std::uint64_t>(config.nx) *
      static_cast<std::uint64_t>(config.ny) *
      static_cast<std::uint64_t>(config.nz);
  constexpr int kThreads = 256;
  const std::uint64_t block_count =
      (sample_count + kThreads - 1) / kThreads;
  if (block_count > static_cast<std::uint64_t>(0x7fffffff)) {
    return cudaErrorInvalidConfiguration;
  }
  GenerateFieldKernel<<<static_cast<unsigned int>(block_count), kThreads, 0,
                        stream>>>(config.field, config.nx, config.ny, config.nz,
                                  variant_seed, device_volume, sample_count);
  return cudaGetLastError();
}

}  // namespace icecarver
