#include "icecarver/api.h"

#include "icecarver/cuda_check.h"
#include "icecarver/mc_tables.cuh"

#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

namespace {

constexpr std::size_t kWorkspaceAlignment = 256;
constexpr int kThreads = 256;
constexpr int kRowThreads = 64;
constexpr int kEmitWarps = 2;
constexpr std::uint32_t kImplementationId = 13;

__device__ __constant__ std::int8_t g_triangle_table[256][16];
__device__ __constant__ std::uint8_t g_triangle_count[256];

bool checked_add(std::size_t a, std::size_t b, std::size_t* result) {
  if (a > std::numeric_limits<std::size_t>::max() - b) {
    return false;
  }
  *result = a + b;
  return true;
}

bool checked_multiply(std::size_t a, std::size_t b, std::size_t* result) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    return false;
  }
  *result = a * b;
  return true;
}

bool append_region(std::size_t bytes, std::size_t alignment,
                   std::size_t* cursor, std::size_t* offset) {
  const std::size_t mask = alignment - 1;
  std::size_t padded = 0;
  if (!checked_add(*cursor, mask, &padded)) {
    return false;
  }
  padded &= ~mask;
  *offset = padded;
  return checked_add(padded, bytes, cursor);
}

int validate_and_count_cells(const icecarver::Input* input,
                             icecarver::Output* output,
                             std::uint32_t* num_cells) {
  if (input == nullptr || output == nullptr || input->volume == nullptr ||
      input->isovalues == nullptr || output->triangle_counts == nullptr ||
      input->nx < 2 || input->ny < 2 || input->nz < 2 ||
      input->num_isovalues < 1 ||
      input->num_isovalues > icecarver::kMaxIsovalues) {
    return icecarver::kInvalidArgument;
  }
  if (input->emit_triangles != 0) {
    for (int iso = 0; iso < input->num_isovalues; ++iso) {
      if (output->triangles[iso] == nullptr) {
        return icecarver::kInvalidArgument;
      }
    }
  }

  std::size_t xy = 0;
  std::size_t cells = 0;
  if (!checked_multiply(static_cast<std::size_t>(input->nx - 1),
                        static_cast<std::size_t>(input->ny - 1), &xy) ||
      !checked_multiply(xy, static_cast<std::size_t>(input->nz - 1),
                        &cells) ||
      cells > std::numeric_limits<std::uint32_t>::max() / 5U ||
      cells > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return icecarver::kSizeOverflow;
  }
  *num_cells = static_cast<std::uint32_t>(cells);
  return icecarver::kSuccess;
}

__global__ void store_workspace_descriptor(
    icecarver::WorkspaceDescriptor* destination,
    icecarver::WorkspaceDescriptor descriptor) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *destination = descriptor;
  }
}

__global__ void classify_cells(const float* __restrict__ volume, int nx,
                               int ny, int cx, int cy,
                               std::uint32_t num_cells,
                               const float* __restrict__ isovalues,
                               int num_isovalues,
                               std::uint8_t* __restrict__ counts) {
  const std::uint32_t cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells) {
    return;
  }
  int x = 0;
  int y = 0;
  int z = 0;
  icecarver::mc::decode_cell(cell_id, cx, cy, x, y, z);
  float values[8];
  icecarver::mc::load_corners(volume, nx, ny, x, y, z, values);
  for (int iso = 0; iso < num_isovalues; ++iso) {
    const std::uint8_t cube_case =
        icecarver::mc::case_index(values, isovalues[iso]);
    counts[static_cast<std::size_t>(iso) * num_cells + cell_id] =
        g_triangle_count[cube_case];
  }
}

template <int NumIsovalues>
__global__ void classify_cells_by_row(
    const float* __restrict__ volume, int nx, int ny, int cx, int cy,
    std::uint32_t num_cells, const float* __restrict__ isovalues,
    std::uint8_t* __restrict__ counts,
    std::uint32_t* __restrict__ row_counts) {
  __shared__ std::uint32_t warp_totals[2][NumIsovalues];

  const std::uint32_t row_id = blockIdx.x;
  const int y = static_cast<int>(row_id % static_cast<std::uint32_t>(cy));
  const int z = static_cast<int>(row_id / static_cast<std::uint32_t>(cy));
  std::uint32_t thread_totals[NumIsovalues]{};
  for (int x = threadIdx.x; x < cx; x += blockDim.x) {
    const std::uint32_t cell_id = row_id * static_cast<std::uint32_t>(cx) +
                                  static_cast<std::uint32_t>(x);
    float values[8];
    icecarver::mc::load_corners(volume, nx, ny, x, y, z, values);
#pragma unroll
    for (int iso = 0; iso < NumIsovalues; ++iso) {
      const std::uint8_t cube_case =
          icecarver::mc::case_index(values, isovalues[iso]);
      const std::uint8_t count = g_triangle_count[cube_case];
      counts[static_cast<std::size_t>(iso) * num_cells + cell_id] = count;
      thread_totals[iso] += count;
    }
  }

#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
#pragma unroll
    for (int iso = 0; iso < NumIsovalues; ++iso) {
      thread_totals[iso] +=
          __shfl_down_sync(0xffffffffU, thread_totals[iso], delta);
    }
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
#pragma unroll
    for (int iso = 0; iso < NumIsovalues; ++iso) {
      warp_totals[warp][iso] = thread_totals[iso];
    }
  }
  __syncthreads();
  if (threadIdx.x < NumIsovalues) {
    const int iso = threadIdx.x;
    row_counts[static_cast<std::size_t>(iso) * gridDim.x + row_id] =
        warp_totals[0][iso] + warp_totals[1][iso];
  }
}

__global__ void publish_total(const std::uint8_t* __restrict__ counts,
                              const std::uint32_t* __restrict__ offsets,
                              std::uint32_t num_cells,
                              std::uint64_t* __restrict__ totals, int iso) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint32_t last = num_cells - 1;
    totals[iso] = static_cast<std::uint64_t>(offsets[last]) + counts[last];
  }
}

__global__ void publish_row_total(
    const std::uint32_t* __restrict__ row_counts,
    const std::uint32_t* __restrict__ row_offsets, std::uint32_t num_rows,
    std::uint64_t* __restrict__ totals, int iso) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint32_t last = num_rows - 1;
    totals[iso] =
        static_cast<std::uint64_t>(row_offsets[last]) + row_counts[last];
  }
}

__global__ void generate_triangles(
    const float* __restrict__ volume, int nx, int ny, int cx, int cy,
    std::uint32_t num_cells, const float* __restrict__ isovalues, int iso,
    const std::uint8_t* __restrict__ counts,
    const std::uint32_t* __restrict__ offsets,
    const std::uint64_t* __restrict__ totals,
    icecarver::Triangle* __restrict__ triangles, std::uint64_t capacity) {
  const std::uint32_t cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (totals[iso] > capacity || cell_id >= num_cells ||
      counts[cell_id] == 0) {
    return;
  }

  const std::uint32_t count = counts[cell_id];
  const std::uint32_t offset = offsets[cell_id];
  if (static_cast<std::uint64_t>(offset) + count > capacity) {
    return;
  }

  int x = 0;
  int y = 0;
  int z = 0;
  icecarver::mc::decode_cell(cell_id, cx, cy, x, y, z);
  float values[8];
  icecarver::mc::load_corners(volume, nx, ny, x, y, z, values);
  const float isovalue = isovalues[iso];
  const std::uint8_t cube_case =
      icecarver::mc::case_index(values, isovalue);
  const std::int8_t* row = g_triangle_table[cube_case];
  for (std::uint32_t triangle = 0; triangle < count; ++triangle) {
    triangles[offset + triangle] = icecarver::mc::make_triangle(
        row, static_cast<int>(triangle), x, y, z, values, isovalue);
  }
}

template <int NumIsovalues>
__global__ __launch_bounds__(kRowThreads, 16) void generate_triangles_by_row(
    const float* __restrict__ volume, int nx, int ny, int cx, int cy,
    std::uint32_t num_rows, const float* __restrict__ isovalues,
    const std::uint8_t* __restrict__ counts,
    const std::uint32_t* __restrict__ row_offsets,
    icecarver::Output output) {
  __shared__ std::uint32_t warp_prefixes[kEmitWarps][NumIsovalues];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const std::uint32_t row_id = blockIdx.x * kEmitWarps + warp;
  if (row_id >= num_rows) {
    return;
  }
  int y = 0;
  int z = 0;
  std::uint32_t valid_isovalues = 0;
  if (lane == 0) {
    y = static_cast<int>(row_id % static_cast<std::uint32_t>(cy));
    z = static_cast<int>(row_id / static_cast<std::uint32_t>(cy));
#pragma unroll
    for (int iso = 0; iso < NumIsovalues; ++iso) {
      warp_prefixes[warp][iso] =
          row_offsets[static_cast<std::size_t>(iso) * num_rows + row_id];
      valid_isovalues |= static_cast<std::uint32_t>(
                             output.triangle_counts[iso] <=
                             output.capacities[iso])
                         << iso;
    }
  }
  y = __shfl_sync(0xffffffffU, y, 0);
  z = __shfl_sync(0xffffffffU, z, 0);
  valid_isovalues = __shfl_sync(0xffffffffU, valid_isovalues, 0);
  const std::size_t num_cells =
      static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(cx);
  for (int base_x = 0; base_x < cx; base_x += 32) {
    const int x = base_x + lane;
    const std::size_t cell_id =
        static_cast<std::size_t>(row_id) * static_cast<std::size_t>(cx) +
        static_cast<std::size_t>(x);
    std::uint64_t packed_counts = 0;
#pragma unroll
    for (int iso = 0; iso < NumIsovalues; ++iso) {
      const std::uint64_t count =
          x < cx ? counts[static_cast<std::size_t>(iso) * num_cells + cell_id]
                 : 0;
      packed_counts |= count << (8 * iso);
    }

    float v0 = 0.0f;
    float v1 = 0.0f;
    float v2 = 0.0f;
    float v3 = 0.0f;
    float v4 = 0.0f;
    float v5 = 0.0f;
    float v6 = 0.0f;
    float v7 = 0.0f;
    if (packed_counts != 0) {
      const std::size_t unx = static_cast<std::size_t>(nx);
      const std::size_t volume_row = unx;
      const std::size_t volume_plane =
          unx * static_cast<std::size_t>(ny);
      const std::size_t base =
          (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny) +
           static_cast<std::size_t>(y)) *
              unx +
          static_cast<std::size_t>(x);
      v0 = volume[base];
      v1 = volume[base + 1];
      v2 = volume[base + volume_row + 1];
      v3 = volume[base + volume_row];
      v4 = volume[base + volume_plane];
      v5 = volume[base + volume_plane + 1];
      v6 = volume[base + volume_plane + volume_row + 1];
      v7 = volume[base + volume_plane + volume_row];
    }

#pragma unroll 1
    for (int iso = 0; iso < NumIsovalues; ++iso) {
      const std::uint32_t count =
          static_cast<std::uint32_t>((packed_counts >> (8 * iso)) & 0xffU);
      std::uint32_t inclusive = count;
#pragma unroll
      for (int delta = 1; delta < 32; delta <<= 1) {
        const std::uint32_t preceding =
            __shfl_up_sync(0xffffffffU, inclusive, delta);
        if (lane >= delta) {
          inclusive += preceding;
        }
      }
      std::uint32_t row_prefix = 0;
      if (lane == 0) {
        row_prefix = warp_prefixes[warp][iso];
      }
      row_prefix = __shfl_sync(0xffffffffU, row_prefix, 0);
      const std::uint32_t offset = row_prefix + inclusive - count;

      const std::uint64_t capacity = output.capacities[iso];
      if (count != 0 && ((valid_isovalues >> iso) & 1U) != 0 &&
          static_cast<std::uint64_t>(offset) + count <= capacity) {
        icecarver::Triangle* __restrict__ triangles = output.triangles[iso];
        const float isovalue = isovalues[iso];
        const std::uint8_t cube_case = static_cast<std::uint8_t>(
            (static_cast<unsigned int>(v0 < isovalue) << 0) |
            (static_cast<unsigned int>(v1 < isovalue) << 1) |
            (static_cast<unsigned int>(v2 < isovalue) << 2) |
            (static_cast<unsigned int>(v3 < isovalue) << 3) |
            (static_cast<unsigned int>(v4 < isovalue) << 4) |
            (static_cast<unsigned int>(v5 < isovalue) << 5) |
            (static_cast<unsigned int>(v6 < isovalue) << 6) |
            (static_cast<unsigned int>(v7 < isovalue) << 7));
        const std::int8_t* row = g_triangle_table[cube_case];
        const float fx = static_cast<float>(x);
        const float fy = static_cast<float>(y);
        const float fz = static_cast<float>(z);
        const auto interpolate_edge = [&](int edge, float& px, float& py,
                                          float& pz) {
          float t = 0.0f;
          switch (edge) {
            case 0:
              t = (isovalue - v0) / (v1 - v0);
              px = fx + t;
              py = fy;
              pz = fz;
              break;
            case 1:
              t = (isovalue - v1) / (v2 - v1);
              px = fx + 1.0f;
              py = fy + t;
              pz = fz;
              break;
            case 2:
              t = (isovalue - v2) / (v3 - v2);
              px = fx + 1.0f - t;
              py = fy + 1.0f;
              pz = fz;
              break;
            case 3:
              t = (isovalue - v3) / (v0 - v3);
              px = fx;
              py = fy + 1.0f - t;
              pz = fz;
              break;
            case 4:
              t = (isovalue - v4) / (v5 - v4);
              px = fx + t;
              py = fy;
              pz = fz + 1.0f;
              break;
            case 5:
              t = (isovalue - v5) / (v6 - v5);
              px = fx + 1.0f;
              py = fy + t;
              pz = fz + 1.0f;
              break;
            case 6:
              t = (isovalue - v6) / (v7 - v6);
              px = fx + 1.0f - t;
              py = fy + 1.0f;
              pz = fz + 1.0f;
              break;
            case 7:
              t = (isovalue - v7) / (v4 - v7);
              px = fx;
              py = fy + 1.0f - t;
              pz = fz + 1.0f;
              break;
            case 8:
              t = (isovalue - v0) / (v4 - v0);
              px = fx;
              py = fy;
              pz = fz + t;
              break;
            case 9:
              t = (isovalue - v1) / (v5 - v1);
              px = fx + 1.0f;
              py = fy;
              pz = fz + t;
              break;
            case 10:
              t = (isovalue - v2) / (v6 - v2);
              px = fx + 1.0f;
              py = fy + 1.0f;
              pz = fz + t;
              break;
            default:
              t = (isovalue - v3) / (v7 - v3);
              px = fx;
              py = fy + 1.0f;
              pz = fz + t;
              break;
          }
        };
        for (std::uint32_t triangle = 0; triangle < count; ++triangle) {
          icecarver::Triangle output_triangle;
          const int first = 3 * static_cast<int>(triangle);
          interpolate_edge(row[first], output_triangle.x0,
                           output_triangle.y0, output_triangle.z0);
          interpolate_edge(row[first + 1], output_triangle.x1,
                           output_triangle.y1, output_triangle.z1);
          interpolate_edge(row[first + 2], output_triangle.x2,
                           output_triangle.y2, output_triangle.z2);
          triangles[offset + triangle] = output_triangle;
        }
      }
      const std::uint32_t chunk_total =
          __shfl_sync(0xffffffffU, inclusive, 31);
      if (lane == 0) {
        warp_prefixes[warp][iso] = row_prefix + chunk_total;
      }
    }
  }
}

}  // namespace

extern "C" int icecarver_solve(const icecarver::Input* input,
                                icecarver::Output* output, void* workspace,
                                std::size_t workspace_bytes,
                                cudaStream_t stream) {
  std::uint32_t num_cells = 0;
  const int validation = validate_and_count_cells(input, output, &num_cells);
  if (validation != icecarver::kSuccess) {
    return validation;
  }
  const unsigned int rows =
      num_cells / static_cast<std::uint32_t>(input->nx - 1);
  const bool use_row_path =
      input->num_isovalues == 2 || input->num_isovalues == 4 ||
      input->num_isovalues == icecarver::kMaxIsovalues;

  std::size_t scan_temp_bytes = 0;
  cudaError_t status = cub::DeviceScan::ExclusiveSum(
      nullptr, scan_temp_bytes, static_cast<const std::uint8_t*>(nullptr),
      static_cast<std::uint32_t*>(nullptr), static_cast<int>(num_cells),
      stream);
  if (status != cudaSuccess) {
    return icecarver::kCudaFailure;
  }
  if (use_row_path) {
    std::size_t row_scan_temp_bytes = 0;
    status = cub::DeviceScan::ExclusiveSum(
        nullptr, row_scan_temp_bytes,
        static_cast<const std::uint32_t*>(nullptr),
        static_cast<std::uint32_t*>(nullptr), static_cast<int>(rows), stream);
    if (status != cudaSuccess) {
      return icecarver::kCudaFailure;
    }
    if (row_scan_temp_bytes > scan_temp_bytes) {
      scan_temp_bytes = row_scan_temp_bytes;
    }
  }

  std::size_t counts_bytes = 0;
  std::size_t offsets_bytes = 0;
  std::size_t offset_entries = static_cast<std::size_t>(num_cells);
  const std::size_t row_offset_entries =
      static_cast<std::size_t>(2 * input->num_isovalues) * rows;
  if (use_row_path && row_offset_entries > offset_entries) {
    offset_entries = row_offset_entries;
  }
  if (!checked_multiply(static_cast<std::size_t>(num_cells),
                        static_cast<std::size_t>(input->num_isovalues),
                        &counts_bytes) ||
      !checked_multiply(offset_entries, sizeof(std::uint32_t),
                        &offsets_bytes)) {
    return icecarver::kSizeOverflow;
  }

  std::size_t cursor = sizeof(icecarver::WorkspaceDescriptor);
  std::size_t counts_offset = 0;
  std::size_t offsets_offset = 0;
  std::size_t scan_temp_offset = 0;
  if (!append_region(counts_bytes, kWorkspaceAlignment, &cursor,
                     &counts_offset) ||
      !append_region(offsets_bytes, kWorkspaceAlignment, &cursor,
                     &offsets_offset) ||
      !append_region(scan_temp_bytes, kWorkspaceAlignment, &cursor,
                     &scan_temp_offset)) {
    return icecarver::kSizeOverflow;
  }
  const std::size_t required_bytes = cursor;
  if (workspace == nullptr || workspace_bytes < required_bytes) {
    return icecarver::kInsufficientWorkspace;
  }
  if ((reinterpret_cast<std::uintptr_t>(workspace) &
       (kWorkspaceAlignment - 1)) != 0) {
    return icecarver::kInvalidArgument;
  }

  auto* bytes = static_cast<unsigned char*>(workspace);
  auto* counts = reinterpret_cast<std::uint8_t*>(bytes + counts_offset);
  auto* offsets = reinterpret_cast<std::uint32_t*>(bytes + offsets_offset);
  void* scan_temp = bytes + scan_temp_offset;

  static std::once_flag table_init_once;
  static cudaError_t table_init_status = cudaSuccess;
  std::call_once(table_init_once, [] {
    table_init_status = cudaMemcpyToSymbol(
        g_triangle_table, icecarver::mc::kTriangleTable,
        sizeof(icecarver::mc::kTriangleTable));
    if (table_init_status == cudaSuccess) {
      table_init_status = cudaMemcpyToSymbol(
          g_triangle_count, icecarver::mc::kTriangleCount,
          sizeof(icecarver::mc::kTriangleCount));
    }
  });
  if (table_init_status != cudaSuccess) {
    return icecarver::kCudaFailure;
  }

  const icecarver::WorkspaceDescriptor descriptor{
      icecarver::kWorkspaceMagic,
      icecarver::kWorkspaceVersion,
      kImplementationId,
      static_cast<std::uint64_t>(required_bytes),
      num_cells,
      static_cast<std::uint64_t>(counts_offset),
      static_cast<std::uint64_t>(offsets_offset),
      static_cast<std::uint64_t>(scan_temp_offset),
      static_cast<std::uint64_t>(scan_temp_bytes),
      {0, 0, 0}};
  store_workspace_descriptor<<<1, 1, 0, stream>>>(
      static_cast<icecarver::WorkspaceDescriptor*>(workspace), descriptor);
  ICECARVER_RETURN_IF_LAUNCH_ERROR();

  ICECARVER_RETURN_IF_CUDA_ERROR(cudaMemsetAsync(
      output->triangle_counts, 0,
      static_cast<std::size_t>(input->num_isovalues) * sizeof(std::uint64_t),
      stream));

  const unsigned int blocks =
      (num_cells + static_cast<std::uint32_t>(kThreads) - 1U) /
      static_cast<std::uint32_t>(kThreads);
  if (input->num_isovalues == 2) {
    classify_cells_by_row<2><<<rows, kRowThreads, 0, stream>>>(
        input->volume, input->nx, input->ny, input->nx - 1, input->ny - 1,
        num_cells, input->isovalues, counts, offsets);
  } else if (input->num_isovalues == 4) {
    classify_cells_by_row<4><<<rows, kRowThreads, 0, stream>>>(
        input->volume, input->nx, input->ny, input->nx - 1, input->ny - 1,
        num_cells, input->isovalues, counts, offsets);
  } else if (input->num_isovalues == icecarver::kMaxIsovalues) {
    classify_cells_by_row<icecarver::kMaxIsovalues>
        <<<rows, kRowThreads, 0, stream>>>(
        input->volume, input->nx, input->ny, input->nx - 1, input->ny - 1,
        num_cells, input->isovalues, counts, offsets);
  } else {
    classify_cells<<<blocks, kThreads, 0, stream>>>(
        input->volume, input->nx, input->ny, input->nx - 1, input->ny - 1,
        num_cells, input->isovalues, input->num_isovalues, counts);
  }
  ICECARVER_RETURN_IF_LAUNCH_ERROR();

  for (int iso = 0; iso < input->num_isovalues; ++iso) {
    const std::uint8_t* iso_counts =
        counts + static_cast<std::size_t>(iso) * num_cells;

    if (use_row_path) {
      auto* row_counts = offsets + static_cast<std::size_t>(iso) * rows;
      auto* row_offsets =
          offsets + static_cast<std::size_t>(input->num_isovalues + iso) *
                        rows;
      ICECARVER_RETURN_IF_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(
          scan_temp, scan_temp_bytes, row_counts, row_offsets,
          static_cast<int>(rows), stream));
      publish_row_total<<<1, 1, 0, stream>>>(
          row_counts, row_offsets, rows, output->triangle_counts, iso);
      ICECARVER_RETURN_IF_LAUNCH_ERROR();
    } else {
      ICECARVER_RETURN_IF_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(
          scan_temp, scan_temp_bytes, iso_counts, offsets,
          static_cast<int>(num_cells), stream));
      publish_total<<<1, 1, 0, stream>>>(
          iso_counts, offsets, num_cells, output->triangle_counts, iso);
      ICECARVER_RETURN_IF_LAUNCH_ERROR();
      if (input->emit_triangles != 0) {
        generate_triangles<<<blocks, kThreads, 0, stream>>>(
            input->volume, input->nx, input->ny, input->nx - 1,
            input->ny - 1, num_cells, input->isovalues, iso, iso_counts,
            offsets, output->triangle_counts, output->triangles[iso],
            output->capacities[iso]);
        ICECARVER_RETURN_IF_LAUNCH_ERROR();
      }
    }
  }

  if (use_row_path && input->emit_triangles != 0) {
    const unsigned int row_blocks =
        (rows + static_cast<unsigned int>(kEmitWarps) - 1U) /
        static_cast<unsigned int>(kEmitWarps);
    if (input->num_isovalues == 2) {
      generate_triangles_by_row<2><<<row_blocks, kRowThreads, 0, stream>>>(
          input->volume, input->nx, input->ny, input->nx - 1, input->ny - 1,
          rows, input->isovalues, counts,
          offsets + static_cast<std::size_t>(input->num_isovalues) * rows,
          *output);
    } else if (input->num_isovalues == 4) {
      generate_triangles_by_row<4><<<row_blocks, kRowThreads, 0, stream>>>(
          input->volume, input->nx, input->ny, input->nx - 1, input->ny - 1,
          rows, input->isovalues, counts,
          offsets + static_cast<std::size_t>(input->num_isovalues) * rows,
          *output);
    } else {
      generate_triangles_by_row<icecarver::kMaxIsovalues>
          <<<row_blocks, kRowThreads, 0, stream>>>(
              input->volume, input->nx, input->ny, input->nx - 1,
              input->ny - 1, rows, input->isovalues, counts,
              offsets + static_cast<std::size_t>(input->num_isovalues) * rows,
              *output);
    }
    ICECARVER_RETURN_IF_LAUNCH_ERROR();
  }

  if (input->emit_triangles != 0) {
    std::uint64_t totals[icecarver::kMaxIsovalues]{};
    ICECARVER_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        totals, output->triangle_counts,
        static_cast<std::size_t>(input->num_isovalues) * sizeof(totals[0]),
        cudaMemcpyDeviceToHost, stream));
    ICECARVER_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (int iso = 0; iso < input->num_isovalues; ++iso) {
      if (totals[iso] > output->capacities[iso]) {
        return icecarver::kInsufficientOutput;
      }
    }
  }

  return icecarver::kSuccess;
}
