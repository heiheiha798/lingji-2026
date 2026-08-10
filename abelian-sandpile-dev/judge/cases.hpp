#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

enum class Pattern : int {
  kCriticalPatch = 1,
  kMultiSource = 2,
  kGlobalRain = 3,
  kMixed = 4,
  kEdgeSources = 5,
};

struct CaseSpec {
  const char *name;
  int rows;
  int cols;
  Pattern pattern;
  std::uint32_t magnitude;
  int param_a;
  int param_b;
  std::uint64_t seed;
};

class Pcg32 {
 public:
  explicit Pcg32(std::uint64_t seed) : state_(0), inc_((seed << 1U) | 1U) {
    Next();
    state_ += seed ^ 0x9e3779b97f4a7c15ULL;
    Next();
  }

  std::uint32_t Next() {
    const std::uint64_t old = state_;
    state_ = old * 6364136223846793005ULL + inc_;
    const std::uint32_t x =
        static_cast<std::uint32_t>(((old >> 18U) ^ old) >> 27U);
    const std::uint32_t rot = static_cast<std::uint32_t>(old >> 59U);
    return (x >> rot) | (x << ((-rot) & 31));
  }

 private:
  std::uint64_t state_;
  std::uint64_t inc_;
};

inline const std::vector<CaseSpec> &PublicCases() {
  static const std::vector<CaseSpec> cases = {
      {"tiny-correctness", 127, 131, Pattern::kMixed, 250, 16, 8, 101},
      {"global-rain", 4096, 4093, Pattern::kGlobalRain, 5500, 0, 0, 202},
      {"local-avalanche", 4096, 4093, Pattern::kCriticalPatch, 40000, 96, 0,
       303},
      {"many-sources", 4096, 4093, Pattern::kMultiSource, 20000, 256, 0, 404},
      {"near-critical", 2048, 2039, Pattern::kGlobalRain, 6500, 0, 0, 505},
      {"large-mixed", 8192, 8179, Pattern::kMixed, 10000, 96, 32, 606},
  };
  return cases;
}

inline int RandomCaseCount() { return 5; }

inline CaseSpec RandomCase(std::uint64_t suite_seed, int index) {
  static const char *names[] = {
      "random-critical-patch", "random-multi-source", "random-global-rain",
      "random-mixed", "random-edge-sources"};
  if (index < 0 || index >= RandomCaseCount())
    throw std::invalid_argument("random case index out of range");

  const std::uint64_t seed =
      suite_seed ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(index + 1));
  Pcg32 rng(seed);
  const int rows = 193 + static_cast<int>(rng.Next() % 320U);
  const int cols = 191 + static_cast<int>(rng.Next() % 322U);
  switch (index) {
    case 0:
      return {names[index], rows, cols, Pattern::kCriticalPatch,
              3000U + rng.Next() % 12000U,
              16 + static_cast<int>(rng.Next() % 48U), 0, seed};
    case 1:
      return {names[index], rows, cols, Pattern::kMultiSource,
              1000U + rng.Next() % 7000U,
              8 + static_cast<int>(rng.Next() % 40U), 0, seed};
    case 2:
      return {names[index], rows, cols, Pattern::kGlobalRain,
              2000U + rng.Next() % 3500U, 0, 0, seed};
    case 3:
      return {names[index], rows, cols, Pattern::kMixed,
              1500U + rng.Next() % 6000U,
              16 + static_cast<int>(rng.Next() % 48U),
              4 + static_cast<int>(rng.Next() % 20U), seed};
    default:
      return {names[index], rows, cols, Pattern::kEdgeSources,
              1000U + rng.Next() % 6000U,
              8 + static_cast<int>(rng.Next() % 40U),
              1 + static_cast<int>(rng.Next() % 8U), seed};
  }
}

inline void FillCriticalPatch(std::vector<std::uint32_t> &data,
                              const CaseSpec &spec, Pcg32 &rng) {
  const std::size_t n = data.size();
  for (std::size_t i = 0; i < n; ++i) data[i] = rng.Next() & 3U;
  const int patch = std::min({spec.param_a, spec.rows, spec.cols});
  const int y0 = (spec.rows - patch) / 2;
  const int x0 = (spec.cols - patch) / 2;
  for (int y = y0; y < y0 + patch; ++y)
    for (int x = x0; x < x0 + patch; ++x)
      data[static_cast<std::size_t>(y) * spec.cols + x] = 3;
  data[static_cast<std::size_t>(spec.rows / 2) * spec.cols + spec.cols / 2] +=
      spec.magnitude;
}

inline void FillMultiSource(std::vector<std::uint32_t> &data,
                            const CaseSpec &spec, Pcg32 &rng) {
  const std::size_t n = data.size();
  for (std::size_t i = 0; i < n; ++i) data[i] = rng.Next() & 1U;
  for (int k = 0; k < spec.param_a; ++k) {
    const int y = 1 + static_cast<int>(rng.Next() % (spec.rows - 2));
    const int x = 1 + static_cast<int>(rng.Next() % (spec.cols - 2));
    data[static_cast<std::size_t>(y) * spec.cols + x] +=
        spec.magnitude + rng.Next() % spec.magnitude;
  }
}

inline void FillGlobalRain(std::vector<std::uint32_t> &data,
                           const CaseSpec &spec, Pcg32 &rng) {
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = rng.Next() & 3U;
    if (rng.Next() % 10000U < spec.magnitude) ++data[i];
  }
}

inline void FillMixed(std::vector<std::uint32_t> &data, const CaseSpec &spec,
                      Pcg32 &rng) {
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = rng.Next() & 1U;
  const int patch = std::min({spec.param_a, spec.rows / 3, spec.cols / 3});
  const int centers[4][2] = {{spec.rows / 3, spec.cols / 3},
                             {spec.rows / 3, 2 * spec.cols / 3},
                             {2 * spec.rows / 3, spec.cols / 3},
                             {2 * spec.rows / 3, 2 * spec.cols / 3}};
  for (const auto &center : centers) {
    const int y0 = center[0] - patch / 2;
    const int x0 = center[1] - patch / 2;
    for (int y = y0; y < y0 + patch; ++y)
      for (int x = x0; x < x0 + patch; ++x)
        data[static_cast<std::size_t>(y) * spec.cols + x] = 3;
  }
  for (int k = 0; k < spec.param_b; ++k) {
    const int c = k & 3;
    const int radius = std::max(1, patch / 3);
    const int y = centers[c][0] +
                  static_cast<int>(rng.Next() % (2 * radius + 1)) - radius;
    const int x = centers[c][1] +
                  static_cast<int>(rng.Next() % (2 * radius + 1)) - radius;
    data[static_cast<std::size_t>(y) * spec.cols + x] +=
        spec.magnitude + rng.Next() % spec.magnitude;
  }
}

inline void FillEdgeSources(std::vector<std::uint32_t> &data,
                            const CaseSpec &spec, Pcg32 &rng) {
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = rng.Next() & 1U;
  const int margin = std::max(1, spec.param_b);
  for (int k = 0; k < spec.param_a; ++k) {
    int y = 0;
    int x = 0;
    switch (k & 3) {
      case 0: y = margin; x = static_cast<int>(rng.Next() % spec.cols); break;
      case 1: y = spec.rows - 1 - margin; x = static_cast<int>(rng.Next() % spec.cols); break;
      case 2: y = static_cast<int>(rng.Next() % spec.rows); x = margin; break;
      default: y = static_cast<int>(rng.Next() % spec.rows); x = spec.cols - 1 - margin;
    }
    data[static_cast<std::size_t>(y) * spec.cols + x] +=
        spec.magnitude + rng.Next() % spec.magnitude;
  }
}

inline std::vector<std::uint32_t> GenerateCase(const CaseSpec &spec) {
  if (spec.rows < 3 || spec.cols < 3)
    throw std::invalid_argument("rows and cols must be at least 3");
  const std::size_t n = static_cast<std::size_t>(spec.rows) * spec.cols;
  if (n >= (1ULL << 32))
    throw std::invalid_argument("case has too many cells for the public API");
  std::vector<std::uint32_t> data(n);
  Pcg32 rng(spec.seed);
  switch (spec.pattern) {
    case Pattern::kCriticalPatch: FillCriticalPatch(data, spec, rng); break;
    case Pattern::kMultiSource: FillMultiSource(data, spec, rng); break;
    case Pattern::kGlobalRain: FillGlobalRain(data, spec, rng); break;
    case Pattern::kMixed: FillMixed(data, spec, rng); break;
    case Pattern::kEdgeSources: FillEdgeSources(data, spec, rng); break;
    default: throw std::invalid_argument("unknown case pattern");
  }
  return data;
}
