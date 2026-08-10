#pragma once

#include <cstdint>

// Compute the reflexive transitive closure as one complete host task.
// Input and output are host, row-major bit-packed matrices. The function owns
// device allocation, transfers, computation and cleanup, and returns only
// after reachability is complete. Zero means success.
extern "C" int closure_run(const std::uint64_t *adjacency,
                            std::uint64_t *reachability, int vertices,
                            int words_per_row);
