#include "reference/cpu_reference.h"

#include "icecarver/mc_tables.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace icecarver::reference {
namespace {

bool valid_shape(const Input& input, std::size_t* num_cells) {
  if (input.nx < 2 || input.ny < 2 || input.nz < 2 ||
      input.num_isovalues < 1 || input.num_isovalues > kMaxIsovalues) {
    return false;
  }

  const std::size_t cx = static_cast<std::size_t>(input.nx - 1);
  const std::size_t cy = static_cast<std::size_t>(input.ny - 1);
  const std::size_t cz = static_cast<std::size_t>(input.nz - 1);
  if (cx > std::numeric_limits<std::size_t>::max() / cy) {
    return false;
  }
  const std::size_t xy = cx * cy;
  if (xy > std::numeric_limits<std::size_t>::max() / cz) {
    return false;
  }
  *num_cells = xy * cz;
  return true;
}

void decode_cell(std::size_t cell_id, int cx, int cy, int* x, int* y,
                 int* z) {
  const std::size_t ux = static_cast<std::size_t>(cx);
  const std::size_t uy = static_cast<std::size_t>(cy);
  *x = static_cast<int>(cell_id % ux);
  cell_id /= ux;
  *y = static_cast<int>(cell_id % uy);
  *z = static_cast<int>(cell_id / uy);
}

void load_corners(const Input& input, int x, int y, int z, float values[8]) {
  const std::size_t nx = static_cast<std::size_t>(input.nx);
  const std::size_t ny = static_cast<std::size_t>(input.ny);
  const std::size_t base =
      (static_cast<std::size_t>(z) * ny + static_cast<std::size_t>(y)) * nx +
      static_cast<std::size_t>(x);
  const std::size_t row = nx;
  const std::size_t plane = nx * ny;

  values[0] = input.volume[base];
  values[1] = input.volume[base + 1];
  values[2] = input.volume[base + row + 1];
  values[3] = input.volume[base + row];
  values[4] = input.volume[base + plane];
  values[5] = input.volume[base + plane + 1];
  values[6] = input.volume[base + plane + row + 1];
  values[7] = input.volume[base + plane + row];
}

std::uint8_t case_index(const float values[8], float isovalue) {
  std::uint8_t result = 0;
  for (int corner = 0; corner < 8; ++corner) {
    if (values[corner] < isovalue) {
      result = static_cast<std::uint8_t>(result | (1U << corner));
    }
  }
  return result;
}

struct Point {
  float x;
  float y;
  float z;
};

Point interpolate_edge(int edge, int x, int y, int z, const float values[8],
                       float isovalue) {
  const int c0 = mc::kEdgeCorners[edge][0];
  const int c1 = mc::kEdgeCorners[edge][1];
  const float t = (isovalue - values[c0]) / (values[c1] - values[c0]);

  const float x0 = static_cast<float>(x + mc::kCornerOffsets[c0][0]);
  const float y0 = static_cast<float>(y + mc::kCornerOffsets[c0][1]);
  const float z0 = static_cast<float>(z + mc::kCornerOffsets[c0][2]);
  const float x1 = static_cast<float>(x + mc::kCornerOffsets[c1][0]);
  const float y1 = static_cast<float>(y + mc::kCornerOffsets[c1][1]);
  const float z1 = static_cast<float>(z + mc::kCornerOffsets[c1][2]);
  return {x0 + t * (x1 - x0), y0 + t * (y1 - y0),
          z0 + t * (z1 - z0)};
}

Triangle make_triangle(std::uint8_t cube_case, int triangle_index, int x,
                       int y, int z, const float values[8], float isovalue) {
  const int table_index = 3 * triangle_index;
  const Point p0 = interpolate_edge(mc::kTriangleTable[cube_case][table_index],
                                    x, y, z, values, isovalue);
  const Point p1 = interpolate_edge(
      mc::kTriangleTable[cube_case][table_index + 1], x, y, z, values,
      isovalue);
  const Point p2 = interpolate_edge(
      mc::kTriangleTable[cube_case][table_index + 2], x, y, z, values,
      isovalue);
  return {p0.x, p0.y, p0.z, p1.x, p1.y,
          p1.z, p2.x, p2.y, p2.z};
}

}  // namespace

int cpu_reference_solve(const Input* input, Output* output) {
  if (input == nullptr || output == nullptr || input->volume == nullptr ||
      input->isovalues == nullptr || output->triangle_counts == nullptr) {
    return kInvalidArgument;
  }

  std::size_t num_cells = 0;
  if (!valid_shape(*input, &num_cells)) {
    return kInvalidArgument;
  }

  for (int iso = 0; iso < input->num_isovalues; ++iso) {
    output->triangle_counts[iso] = 0;
  }

  const int cx = input->nx - 1;
  const int cy = input->ny - 1;
  for (int iso = 0; iso < input->num_isovalues; ++iso) {
    const float isovalue = input->isovalues[iso];
    std::uint64_t total = 0;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id) {
      int x = 0;
      int y = 0;
      int z = 0;
      decode_cell(cell_id, cx, cy, &x, &y, &z);
      float values[8];
      load_corners(*input, x, y, z, values);
      total += mc::kTriangleCount[case_index(values, isovalue)];
    }
    output->triangle_counts[iso] = total;
  }

  if (input->emit_triangles != 0) {
    for (int iso = 0; iso < input->num_isovalues; ++iso) {
      const std::uint64_t count = output->triangle_counts[iso];
      if (output->capacities[iso] < count ||
          (count != 0 && output->triangles[iso] == nullptr)) {
        return kInsufficientOutput;
      }
    }
  } else {
    return kSuccess;
  }

  for (int iso = 0; iso < input->num_isovalues; ++iso) {
    const float isovalue = input->isovalues[iso];
    std::uint64_t write_offset = 0;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id) {
      int x = 0;
      int y = 0;
      int z = 0;
      decode_cell(cell_id, cx, cy, &x, &y, &z);
      float values[8];
      load_corners(*input, x, y, z, values);
      const std::uint8_t cube_case = case_index(values, isovalue);
      const int count = mc::kTriangleCount[cube_case];
      for (int triangle = 0; triangle < count; ++triangle) {
        output->triangles[iso][write_offset++] = make_triangle(
            cube_case, triangle, x, y, z, values, isovalue);
      }
    }
  }

  return kSuccess;
}

}  // namespace icecarver::reference
