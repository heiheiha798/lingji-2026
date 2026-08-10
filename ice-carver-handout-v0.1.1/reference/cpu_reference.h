#pragma once

#include "icecarver/api.h"

namespace icecarver::reference {

// Host-only trusted implementation.  Unlike the CUDA ABI, every pointer in
// input and output must refer to host memory.
int cpu_reference_solve(const Input* input, Output* output);

}  // namespace icecarver::reference
