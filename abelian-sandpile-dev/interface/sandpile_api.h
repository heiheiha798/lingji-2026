#pragma once

#include <cstdint>

// Stabilize an open-boundary Abelian sandpile as one complete host task.
//
// initial    host, row-major uint32 input, rows * cols elements, read-only
// stable     host, row-major uint8 output, final values must be in [0, 3]
// odometer   host, row-major uint64 output, number of topplings at each cell
//
// The function owns device allocation, transfers, computation and cleanup.
// It returns only after both host outputs are complete. Zero means success.
extern "C" int sandpile_run(const std::uint32_t *initial,
                             std::uint8_t *stable,
                             std::uint64_t *odometer, int rows, int cols);
