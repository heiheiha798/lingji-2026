#include "icecarver/mc_tables.cuh"

#include <array>
#include <cstdint>
#include <iostream>

int main() {
  std::array<bool, 8> seen_corners{};
  for (int corner = 0; corner < 8; ++corner) {
    const int x = icecarver::mc::kCornerOffsets[corner][0];
    const int y = icecarver::mc::kCornerOffsets[corner][1];
    const int z = icecarver::mc::kCornerOffsets[corner][2];
    if ((x != 0 && x != 1) || (y != 0 && y != 1) ||
        (z != 0 && z != 1)) {
      std::cerr << "corner offset is not binary at " << corner << '\n';
      return 1;
    }
    const int encoded = x | (y << 1) | (z << 2);
    if (seen_corners[encoded]) {
      std::cerr << "duplicate corner offset at " << corner << '\n';
      return 1;
    }
    seen_corners[encoded] = true;
  }

  for (int edge = 0; edge < 12; ++edge) {
    const int a = icecarver::mc::kEdgeCorners[edge][0];
    const int b = icecarver::mc::kEdgeCorners[edge][1];
    if (a < 0 || a >= 8 || b < 0 || b >= 8 || a == b) {
      std::cerr << "invalid edge endpoints at " << edge << '\n';
      return 1;
    }
    int changed_axes = 0;
    for (int axis = 0; axis < 3; ++axis) {
      changed_axes +=
          icecarver::mc::kCornerOffsets[a][axis] !=
          icecarver::mc::kCornerOffsets[b][axis];
    }
    if (changed_axes != 1) {
      std::cerr << "edge does not join adjacent corners at " << edge << '\n';
      return 1;
    }
  }

  for (int cube_case = 0; cube_case < 256; ++cube_case) {
    int entries = 0;
    bool saw_sentinel = false;
    for (int column = 0; column < 16; ++column) {
      const int edge = icecarver::mc::kTriangleTable[cube_case][column];
      if (edge == -1) {
        saw_sentinel = true;
      } else {
        if (saw_sentinel || edge < 0 || edge >= 12) {
          std::cerr << "invalid triangle row at case " << cube_case << '\n';
          return 1;
        }
        ++entries;
      }
    }
    if (!saw_sentinel || entries % 3 != 0 || entries > 15 ||
        icecarver::mc::kTriangleCount[cube_case] != entries / 3) {
      std::cerr << "triangle-count mismatch at case " << cube_case << '\n';
      return 1;
    }
  }

  std::cout << "Marching Cubes table invariants: PASS\n";
  return 0;
}
