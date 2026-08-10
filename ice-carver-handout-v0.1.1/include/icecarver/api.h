#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace icecarver {

inline constexpr int kMaxIsovalues = 8;
inline constexpr std::uint64_t kWorkspaceMagic = 0x4943454341525645ULL;
inline constexpr std::uint32_t kWorkspaceVersion = 1;

// All pointer members passed to a CUDA solver point to device memory.  The
// descriptor objects themselves and capacities[] live in host memory.
struct Input {
  int nx;
  int ny;
  int nz;
  int num_isovalues;
  const float* volume;
  const float* isovalues;
  int emit_triangles;
};

struct Triangle {
  float x0;
  float y0;
  float z0;
  float x1;
  float y1;
  float z1;
  float x2;
  float y2;
  float z2;
};

struct Output {
  std::uint64_t capacities[kMaxIsovalues];
  Triangle* triangles[kMaxIsovalues];
  std::uint64_t* triangle_counts;
};

// The first bytes of a valid CUDA workspace contain this descriptor after a
// successful launch.  Offsets are measured from the beginning of workspace.
// Solvers may use the reserved fields for implementation-specific metadata.
struct WorkspaceDescriptor {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t implementation;
  std::uint64_t required_bytes;
  std::uint64_t num_cells;
  std::uint64_t counts_offset;
  std::uint64_t offsets_offset;
  std::uint64_t scan_temp_offset;
  std::uint64_t scan_temp_bytes;
  std::uint64_t reserved[3];
};

enum Status : int {
  kSuccess = 0,
  kInvalidArgument = 1,
  kInsufficientWorkspace = 2,
  kInsufficientOutput = 3,
  kSizeOverflow = 4,
  kCudaFailure = 5,
};

static_assert(sizeof(Triangle) == 9 * sizeof(float),
              "Triangle must contain exactly nine floats");
static_assert(sizeof(WorkspaceDescriptor) == 88,
              "WorkspaceDescriptor ABI unexpectedly changed");

}  // namespace icecarver

extern "C" int icecarver_solve(const icecarver::Input* input,
                                icecarver::Output* output, void* workspace,
                                std::size_t workspace_bytes,
                                cudaStream_t stream);
