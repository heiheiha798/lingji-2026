#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

enum class GraphPattern : int {
  kLayeredDag = 1,
  kBlockScc = 2,
  kRandomSparse = 3,
  kGridDag = 4,
  kMixed = 5,
};

struct CaseSpec {
  const char *name;
  int vertices;
  GraphPattern pattern;
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

inline int WordsPerRow(int vertices) { return (vertices + 63) / 64; }

inline void AddEdge(std::vector<std::uint64_t> &graph, int words, int from,
                    int to) {
  graph[static_cast<std::size_t>(from) * words + to / 64] |=
      1ULL << (to & 63);
}

inline const std::vector<CaseSpec> &PublicCases() {
  static const std::vector<CaseSpec> cases = {
      {"tiny-correctness", 257, GraphPattern::kMixed, 16, 3, 101},
      {"layered-dag", 16384, GraphPattern::kLayeredDag, 128, 5, 202},
      {"block-scc", 32768, GraphPattern::kBlockScc, 64, 3, 303},
      {"random-sparse", 32768, GraphPattern::kRandomSparse, 8, 0, 404},
      {"grid-dag", 16384, GraphPattern::kGridDag, 128, 0, 505},
      {"large-mixed", 65536, GraphPattern::kMixed, 64, 5, 606},
  };
  return cases;
}

inline int RandomCaseCount() { return 5; }

inline CaseSpec RandomCase(std::uint64_t suite_seed, int index) {
  static const char *names[] = {
      "random-layered-dag", "random-block-scc", "random-sparse",
      "random-grid-dag", "random-mixed"};
  if (index < 0 || index >= RandomCaseCount())
    throw std::invalid_argument("random case index out of range");
  const std::uint64_t seed =
      suite_seed ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(index + 1));
  Pcg32 rng(seed);
  int vertices = 97 + static_cast<int>(rng.Next() % 160U);
  switch (index) {
    case 0:
      return {names[index], vertices, GraphPattern::kLayeredDag,
              8 + static_cast<int>(rng.Next() % 16U),
              2 + static_cast<int>(rng.Next() % 5U), seed};
    case 1:
      return {names[index], vertices, GraphPattern::kBlockScc,
              4 + static_cast<int>(rng.Next() % 12U),
              1 + static_cast<int>(rng.Next() % 4U), seed};
    case 2:
      return {names[index], vertices, GraphPattern::kRandomSparse,
              2 + static_cast<int>(rng.Next() % 8U), 0, seed};
    case 3: {
      const int width = 11 + static_cast<int>(rng.Next() % 8U);
      vertices = width * (9 + static_cast<int>(rng.Next() % 8U));
      return {names[index], vertices, GraphPattern::kGridDag, width, 0, seed};
    }
    default:
      return {names[index], vertices, GraphPattern::kMixed,
              4 + static_cast<int>(rng.Next() % 12U),
              2 + static_cast<int>(rng.Next() % 5U), seed};
  }
}

inline void FillLayeredDag(std::vector<std::uint64_t> &graph,
                           const CaseSpec &spec, Pcg32 &rng) {
  const int words = WordsPerRow(spec.vertices);
  const int layers = std::max(2, spec.param_a);
  const int layer_size = (spec.vertices + layers - 1) / layers;
  for (int u = 0; u < spec.vertices; ++u) {
    const int layer = u / layer_size;
    if (layer + 1 >= layers) continue;
    for (int e = 0; e < spec.param_b; ++e) {
      const int jump = 1 + static_cast<int>(rng.Next() % 3U);
      const int target_layer = std::min(layers - 1, layer + jump);
      const int begin = target_layer * layer_size;
      const int end = std::min(spec.vertices, begin + layer_size);
      if (begin < end)
        AddEdge(graph, words, u,
                begin + static_cast<int>(rng.Next() % (end - begin)));
    }
  }
}

inline void FillBlockScc(std::vector<std::uint64_t> &graph,
                         const CaseSpec &spec, Pcg32 &rng) {
  const int words = WordsPerRow(spec.vertices);
  const int block = std::max(2, spec.param_a);
  const int blocks = (spec.vertices + block - 1) / block;
  for (int b = 0; b < blocks; ++b) {
    const int begin = b * block;
    const int end = std::min(spec.vertices, begin + block);
    for (int u = begin; u < end; ++u) {
      AddEdge(graph, words, u, u + 1 < end ? u + 1 : begin);
      for (int e = 0; e < spec.param_b; ++e)
        AddEdge(graph, words, u,
                begin + static_cast<int>(rng.Next() % (end - begin)));
    }
    if (b + 1 < blocks) {
      AddEdge(graph, words, begin,
              (b + 1) * block + static_cast<int>(rng.Next() %
                                                  std::min(block, spec.vertices - (b + 1) * block)));
      if (b + 2 < blocks && (rng.Next() & 1U))
        AddEdge(graph, words, begin, (b + 2) * block);
    }
  }
}

inline void FillRandomSparse(std::vector<std::uint64_t> &graph,
                             const CaseSpec &spec, Pcg32 &rng) {
  const int words = WordsPerRow(spec.vertices);
  for (int u = 0; u < spec.vertices; ++u)
    for (int e = 0; e < spec.param_a; ++e)
      AddEdge(graph, words, u, static_cast<int>(rng.Next() % spec.vertices));
}

inline void FillGridDag(std::vector<std::uint64_t> &graph,
                        const CaseSpec &spec) {
  const int words = WordsPerRow(spec.vertices);
  const int width = std::max(2, spec.param_a);
  for (int u = 0; u < spec.vertices; ++u) {
    const int x = u % width;
    if (x + 1 < width && u + 1 < spec.vertices) AddEdge(graph, words, u, u + 1);
    if (u + width < spec.vertices) AddEdge(graph, words, u, u + width);
    if (x + 1 < width && u + width + 1 < spec.vertices)
      AddEdge(graph, words, u, u + width + 1);
  }
}

inline void FillMixed(std::vector<std::uint64_t> &graph, const CaseSpec &spec,
                      Pcg32 &rng) {
  const int words = WordsPerRow(spec.vertices);
  const int block = std::max(2, spec.param_a);
  for (int begin = 0; begin < spec.vertices; begin += block) {
    const int end = std::min(spec.vertices, begin + block);
    for (int u = begin; u < end; ++u) {
      if (((begin / block) & 1) == 0)
        AddEdge(graph, words, u, u + 1 < end ? u + 1 : begin);
      for (int e = 0; e < spec.param_b; ++e) {
        const int target = static_cast<int>(rng.Next() % spec.vertices);
        if (target >= begin) AddEdge(graph, words, u, target);
      }
    }
  }
}

inline std::vector<std::uint64_t> GenerateCase(const CaseSpec &spec) {
  if (spec.vertices <= 0)
    throw std::invalid_argument("vertices must be positive");
  const int words = WordsPerRow(spec.vertices);
  std::vector<std::uint64_t> graph(
      static_cast<std::size_t>(spec.vertices) * words, 0);
  Pcg32 rng(spec.seed);
  switch (spec.pattern) {
    case GraphPattern::kLayeredDag: FillLayeredDag(graph, spec, rng); break;
    case GraphPattern::kBlockScc: FillBlockScc(graph, spec, rng); break;
    case GraphPattern::kRandomSparse: FillRandomSparse(graph, spec, rng); break;
    case GraphPattern::kGridDag: FillGridDag(graph, spec); break;
    case GraphPattern::kMixed: FillMixed(graph, spec, rng); break;
    default: throw std::invalid_argument("unknown graph pattern");
  }
  return graph;
}
