#include "config_parser.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "icecarver/api.h"

namespace icecarver {
namespace {

std::string Trim(const std::string& value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string StripComment(const std::string& line) {
  bool single_quote = false;
  bool double_quote = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '\'' && !double_quote) {
      single_quote = !single_quote;
    } else if (line[i] == '"' && !single_quote &&
               (i == 0 || line[i - 1] != '\\')) {
      double_quote = !double_quote;
    } else if (line[i] == '#' && !single_quote && !double_quote) {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string Unquote(const std::string& value) {
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

bool ParseUnsigned(const std::string& text, std::uint64_t* value) {
  if (text.empty() || text.front() == '-') {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 0);
  if (errno != 0 || end == text.c_str() || *end != '\0') {
    return false;
  }
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}

bool ParseInt(const std::string& text, int* value) {
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseDouble(const std::string& text, double* value) {
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      !std::isfinite(parsed)) {
    return false;
  }
  *value = parsed;
  return true;
}

bool ParseArray(const std::string& text, std::vector<std::string>* values) {
  if (text.size() < 2 || text.front() != '[' || text.back() != ']') {
    return false;
  }
  const std::string body = Trim(text.substr(1, text.size() - 2));
  values->clear();
  if (body.empty()) {
    return true;
  }
  std::size_t begin = 0;
  while (begin <= body.size()) {
    const std::size_t comma = body.find(',', begin);
    const std::string item = Trim(body.substr(
        begin, comma == std::string::npos ? std::string::npos : comma - begin));
    if (item.empty()) {
      return false;
    }
    values->push_back(Unquote(item));
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }
  return true;
}

bool ParseField(const std::string& text, FieldKind* field) {
  if (text == "sphere") {
    *field = FieldKind::kSphere;
  } else if (text == "metaball") {
    *field = FieldKind::kMetaball;
  } else if (text == "gyroid") {
    *field = FieldKind::kGyroid;
  } else if (text == "mixed") {
    *field = FieldKind::kMixed;
  } else if (text == "multiscale") {
    *field = FieldKind::kMultiscale;
  } else if (text == "dense") {
    *field = FieldKind::kDense;
  } else {
    return false;
  }
  return true;
}

bool Fail(std::size_t line, const std::string& message, std::string* error) {
  if (error != nullptr) {
    std::ostringstream stream;
    if (line != 0) {
      stream << "line " << line << ": ";
    }
    stream << message;
    *error = stream.str();
  }
  return false;
}

}  // namespace

bool LoadCaseConfig(const std::string& path, CaseConfig* config,
                    std::string* error) {
  if (config == nullptr) {
    return Fail(0, "null CaseConfig output", error);
  }
  std::ifstream input(path);
  if (!input) {
    return Fail(0, "cannot open config: " + path, error);
  }

  CaseConfig parsed;
  std::set<std::string> seen;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = Trim(StripComment(line));
    if (line.empty() || line == "---") {
      continue;
    }
    if (!line.empty() && (line.front() == '-' || line.front() == ' ')) {
      return Fail(line_number,
                  "nested YAML and sequence entries are not supported", error);
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      return Fail(line_number, "expected 'key: value'", error);
    }
    const std::string key = Trim(line.substr(0, colon));
    const std::string value = Trim(line.substr(colon + 1));
    if (key.empty() || value.empty()) {
      return Fail(line_number, "empty key or value", error);
    }
    if (!seen.insert(key).second) {
      return Fail(line_number, "duplicate key: " + key, error);
    }

    if (key == "id") {
      parsed.id = Unquote(value);
    } else if (key == "field") {
      if (!ParseField(Unquote(value), &parsed.field)) {
        return Fail(line_number, "unsupported field: " + value, error);
      }
    } else if (key == "shape") {
      std::vector<std::string> items;
      if (!ParseArray(value, &items) || items.size() != 3 ||
          !ParseInt(items[0], &parsed.nx) ||
          !ParseInt(items[1], &parsed.ny) ||
          !ParseInt(items[2], &parsed.nz)) {
        return Fail(line_number, "shape must be [nx, ny, nz]", error);
      }
    } else if (key == "seed") {
      if (!ParseUnsigned(value, &parsed.seed)) {
        return Fail(line_number, "seed must be an unsigned integer", error);
      }
    } else if (key == "isovalues") {
      std::vector<std::string> items;
      if (!ParseArray(value, &items) || items.empty()) {
        return Fail(line_number, "isovalues must be a non-empty array", error);
      }
      parsed.isovalues.clear();
      for (const std::string& item : items) {
        double number = 0.0;
        if (!ParseDouble(item, &number) ||
            number < -std::numeric_limits<float>::max() ||
            number > std::numeric_limits<float>::max()) {
          return Fail(line_number, "invalid float in isovalues: " + item,
                      error);
        }
        parsed.isovalues.push_back(static_cast<float>(number));
      }
    } else if (key == "per_iso_capacity") {
      if (!ParseUnsigned(value, &parsed.per_iso_capacity)) {
        return Fail(line_number, "invalid per_iso_capacity", error);
      }
    } else if (key == "warmup_runs") {
      if (!ParseInt(value, &parsed.warmup_runs)) {
        return Fail(line_number, "invalid warmup_runs", error);
      }
    } else if (key == "measure_runs") {
      if (!ParseInt(value, &parsed.measure_runs)) {
        return Fail(line_number, "invalid measure_runs", error);
      }
    } else if (key == "abs_tolerance") {
      if (!ParseDouble(value, &parsed.abs_tolerance)) {
        return Fail(line_number, "invalid abs_tolerance", error);
      }
    } else if (key == "rel_tolerance") {
      if (!ParseDouble(value, &parsed.rel_tolerance)) {
        return Fail(line_number, "invalid rel_tolerance", error);
      }
    } else if (key == "timeout_seconds") {
      if (!ParseDouble(value, &parsed.timeout_seconds)) {
        return Fail(line_number, "invalid timeout_seconds", error);
      }
    } else if (key == "correctness_weight") {
      if (!ParseDouble(value, &parsed.correctness_weight)) {
        return Fail(line_number, "invalid correctness_weight", error);
      }
    } else if (key == "performance_weight") {
      if (!ParseDouble(value, &parsed.performance_weight)) {
        return Fail(line_number, "invalid performance_weight", error);
      }
    } else {
      return Fail(line_number, "unsupported key: " + key, error);
    }
  }

  const char* required[] = {"id",          "field",     "shape",
                            "seed",        "isovalues", "per_iso_capacity",
                            "warmup_runs", "measure_runs"};
  for (const char* key : required) {
    if (seen.find(key) == seen.end()) {
      return Fail(0, std::string("missing required key: ") + key, error);
    }
  }
  if (parsed.id.empty()) {
    return Fail(0, "id cannot be empty", error);
  }
  if (parsed.nx < 2 || parsed.ny < 2 || parsed.nz < 2) {
    return Fail(0, "every shape dimension must be at least 2", error);
  }
  if (parsed.isovalues.size() > static_cast<std::size_t>(kMaxIsovalues)) {
    return Fail(0, "too many isovalues", error);
  }
  if (parsed.per_iso_capacity == 0) {
    return Fail(0, "per_iso_capacity must be positive", error);
  }
  if (parsed.warmup_runs != 2 || parsed.measure_runs != 5) {
    return Fail(0, "contest cases require warmup_runs=2 and measure_runs=5",
                error);
  }
  constexpr double kMaximumTimeoutSeconds = 3600.0;
  if (parsed.abs_tolerance < 0.0 || parsed.rel_tolerance < 0.0 ||
      parsed.timeout_seconds <= 0.0 ||
      parsed.timeout_seconds > kMaximumTimeoutSeconds ||
      parsed.correctness_weight < 0.0 ||
      parsed.performance_weight < 0.0) {
    return Fail(0,
                "tolerances and weights must be non-negative; timeout must "
                "be in (0, 3600] seconds",
                error);
  }

  const std::uint64_t nx = static_cast<std::uint64_t>(parsed.nx);
  const std::uint64_t ny = static_cast<std::uint64_t>(parsed.ny);
  const std::uint64_t nz = static_cast<std::uint64_t>(parsed.nz);
  if (nx > std::numeric_limits<std::uint64_t>::max() / ny ||
      nx * ny > std::numeric_limits<std::uint64_t>::max() / nz ||
      nx * ny * nz >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return Fail(0, "shape is too large", error);
  }

  *config = std::move(parsed);
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

}  // namespace icecarver
