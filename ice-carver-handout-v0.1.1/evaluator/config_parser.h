#pragma once

#include <string>

#include "icecarver/case_config.h"

namespace icecarver {

// Parses the intentionally small, dependency-free YAML subset used by the
// checked-in case files. Supported values are scalars and one-line arrays.
bool LoadCaseConfig(const std::string& path, CaseConfig* config,
                    std::string* error);

}  // namespace icecarver
