#pragma once

#include "icecarver/api.h"

extern "C" int icecarver_reference_solve(const icecarver::Input* input,
                                          icecarver::Output* output,
                                          void* workspace,
                                          std::size_t workspace_bytes,
                                          cudaStream_t stream);

extern "C" int icecarver_target_solve(const icecarver::Input* input,
                                       icecarver::Output* output,
                                       void* workspace,
                                       std::size_t workspace_bytes,
                                       cudaStream_t stream);
