#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace icecarver {

enum class FieldKind {
  kSphere,
  kMetaball,
  kGyroid,
  kMixed,
  kMultiscale,
  kDense,
};

struct CaseConfig {
  std::string id;
  FieldKind field = FieldKind::kSphere;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  std::uint64_t seed = 0;
  std::vector<float> isovalues;
  std::uint64_t per_iso_capacity = 0;
  int warmup_runs = 2;
  int measure_runs = 5;
  double abs_tolerance = 1.0e-5;
  double rel_tolerance = 1.0e-5;
  double timeout_seconds = 600.0;
  double correctness_weight = 0.0;
  double performance_weight = 0.0;
};

inline const char* FieldKindName(FieldKind field) {
  switch (field) {
    case FieldKind::kSphere:
      return "sphere";
    case FieldKind::kMetaball:
      return "metaball";
    case FieldKind::kGyroid:
      return "gyroid";
    case FieldKind::kMixed:
      return "mixed";
    case FieldKind::kMultiscale:
      return "multiscale";
    case FieldKind::kDense:
      return "dense";
  }
  return "unknown";
}

}  // namespace icecarver
