#ifndef NUMERICAL_TABULATE_HPP
#define NUMERICAL_TABULATE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <bit>

template<typename T>
class TabulatedFunction
{
  static_assert(std::is_floating_point_v<T>,
                "TabulatedFunction requires a floating-point scalar type");

public:
  using Function = std::function<T(T)>;
  struct Point
  {
    T first;
    T second;
  };

private:
  static constexpr std::size_t max_depth_ = 32;
  static constexpr std::size_t max_points_ = 1'000'000;
  // The payload is native-endian Point records; the header rejects incompatible types.
  static constexpr std::array<char, 8> file_magic_{
    'N', 'T', 'B', 'L', 'v', '0', '0', '1'};
  static constexpr std::uint32_t file_version_ = 1;
  static_assert(std::is_trivially_copyable_v<Point>);
  static_assert(sizeof(Point) == 2 * sizeof(T),
                "TabulatedFunction point records must not contain padding");

  std::vector<Point> points_;

  template<typename Value>
  static void ReadValue(std::ifstream& file, Value& value)
  {
    file.read(reinterpret_cast<char*>(&value), sizeof(Value));
    if (!file) {
      throw std::runtime_error("TabulatedFunction file is truncated");
    }
  }

  template<typename Value>
  static void WriteValue(std::ofstream& file, const Value& value)
  {
    file.write(reinterpret_cast<const char*>(&value), sizeof(Value));
    if (!file) {
      throw std::runtime_error("Failed to write TabulatedFunction file");
    }
  }

  static std::vector<Point> ReadFile(const std::filesystem::path& path)
  {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Could not open TabulatedFunction file");
    }

    std::array<char, file_magic_.size()> magic{};
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!file || magic != file_magic_) {
      throw std::runtime_error("Invalid TabulatedFunction file header");
    }

    std::uint32_t version = 0;
    std::uint32_t scalar_size = 0;
    std::uint32_t digits = 0;
    std::uint32_t max_exponent = 0;
    std::uint8_t byte_order = 0;
    std::uint64_t count = 0;
    ReadValue(file, version);
    ReadValue(file, scalar_size);
    ReadValue(file, digits);
    ReadValue(file, max_exponent);
    ReadValue(file, byte_order);
    ReadValue(file, count);

    const std::uint8_t native_byte_order =
      std::endian::native == std::endian::little ? 1 :
      std::endian::native == std::endian::big ? 2 : 0;
    if (version != file_version_ || scalar_size != sizeof(T) ||
        digits != std::numeric_limits<T>::digits ||
        max_exponent != std::numeric_limits<T>::max_exponent ||
        byte_order != native_byte_order) {
      throw std::runtime_error(
        "TabulatedFunction file format or scalar type is incompatible");
    }
    if (count < 2 || count > max_points_) {
      throw std::runtime_error("Invalid TabulatedFunction point count");
    }

    constexpr std::uint64_t header_size =
      file_magic_.size() + sizeof(version) + sizeof(scalar_size) +
      sizeof(digits) + sizeof(max_exponent) + sizeof(byte_order) +
      sizeof(count);
    const std::uint64_t expected_size =
      header_size + count * static_cast<std::uint64_t>(sizeof(Point));
    std::error_code error;
    const std::uintmax_t actual_size = std::filesystem::file_size(path, error);
    if (error || actual_size != expected_size) {
      throw std::runtime_error(
        "TabulatedFunction file has an invalid size");
    }

    std::vector<Point> points(static_cast<std::size_t>(count));
    file.read(reinterpret_cast<char*>(points.data()),
              static_cast<std::streamsize>(points.size() * sizeof(Point)));
    if (!file) {
      throw std::runtime_error("TabulatedFunction file is truncated");
    }

    for (std::size_t i = 0; i < points.size(); ++i) {
      if (!std::isfinite(points[i].first) ||
          !std::isfinite(points[i].second) ||
          (i != 0 && !(points[i - 1].first < points[i].first))) {
        throw std::runtime_error("Invalid points in TabulatedFunction file");
      }
    }
    return points;
  }

  static T Interpolate(const Point& left, const Point& right, T x)
  {
    const T fraction = (x - left.first) / (right.first - left.first);
    return left.second + fraction * (right.second - left.second);
  }

  static bool WithinTolerance(T actual, T estimate, T tolerance)
  {
    if (!std::isfinite(actual) || !std::isfinite(estimate)) {
      return false;
    }

    const T error = std::abs(actual - estimate);
    if (actual == T(0)) {
      return error == T(0);
    }
    return error / std::abs(actual) <= tolerance;
  }

  void BuildInterval(const Function& function,
                     const Point& left,
                     const Point& right,
                     T tolerance,
                     std::size_t depth)
  {
    const T width = right.first - left.first;
    const T midpoint_x = left.first + width / T(2);
    if (midpoint_x == left.first || midpoint_x == right.first) {
      throw std::runtime_error(
        "TabulatedFunction cannot refine an interval at this precision");
    }

    const Point midpoint{midpoint_x, function(midpoint_x)};
    if (!std::isfinite(midpoint.second)) {
      throw std::domain_error("TabulatedFunction produced a non-finite value");
    }

    bool acceptable = true;
    for (const T fraction : {T(0.25), T(0.75)}) {
      const T x = left.first + width * fraction;
      const T actual = function(x);
      if (!std::isfinite(actual)) {
        throw std::domain_error(
          "TabulatedFunction produced a non-finite value");
      }
      acceptable = acceptable &&
        WithinTolerance(actual, Interpolate(left, right, x), tolerance);
    }
    acceptable = acceptable &&
      WithinTolerance(midpoint.second,
                      Interpolate(left, right, midpoint.first),
                      tolerance);

    if (acceptable) {
      if (points_.size() >= max_points_) {
        throw std::runtime_error("TabulatedFunction exceeded its point limit");
      }
      points_.push_back(right);
      return;
    }

    if (depth >= max_depth_) {
      throw std::runtime_error(
        "TabulatedFunction could not meet the requested relative tolerance");
    }

    BuildInterval(function, left, midpoint, tolerance, depth + 1);
    BuildInterval(function, midpoint, right, tolerance, depth + 1);
  }

public:
  explicit TabulatedFunction(const std::filesystem::path& path)
    : points_(ReadFile(path))
  {}

  TabulatedFunction(Function function, T lower, T upper, T relative_tolerance)
    : TabulatedFunction(std::move(function), Point{lower, T(0)},
                       Point{upper, T(0)}, relative_tolerance)
  {}

  TabulatedFunction(Function function,
                    std::pair<T, T> range,
                    T relative_tolerance)
    : TabulatedFunction(std::move(function),
                       Point{range.first, T(0)},
                       Point{range.second, T(0)},
                       relative_tolerance)
  {}

private:
  TabulatedFunction(Function function,
                    Point left,
                    Point right,
                    T relative_tolerance)
  {
    if (!function) {
      throw std::invalid_argument("TabulatedFunction requires a function");
    }
    if (!std::isfinite(left.first) || !std::isfinite(right.first) ||
        !(left.first < right.first)) {
      throw std::invalid_argument(
        "TabulatedFunction range must be finite and increasing");
    }
    if (!std::isfinite(relative_tolerance) ||
        !(relative_tolerance > T(0))) {
      throw std::invalid_argument(
        "TabulatedFunction relative tolerance must be finite and positive");
    }

    left.second = function(left.first);
    right.second = function(right.first);
    if (!std::isfinite(left.second) || !std::isfinite(right.second)) {
      throw std::domain_error("TabulatedFunction produced a non-finite value");
    }

    points_.reserve(64);
    points_.push_back(left);
    BuildInterval(function, left, right, relative_tolerance, 0);
  }

public:
  T operator()(T x) const
  {
    if (!std::isfinite(x)) {
      throw std::out_of_range("TabulatedFunction input must be finite");
    }
    if (x < points_.front().first || x > points_.back().first) {
      throw std::out_of_range("TabulatedFunction input is outside its range");
    }

    const auto right = std::lower_bound(
      points_.begin(), points_.end(), x,
      [](const Point& point, T value) { return point.first < value; });
    if (right == points_.begin() || right->first == x) {
      return right->second;
    }

    return Interpolate(*(right - 1), *right, x);
  }

  std::size_t PointCount() const noexcept
  {
    return points_.size();
  }

  const std::vector<Point>& Points() const noexcept
  {
    return points_;
  }

  void Save(const std::filesystem::path& path) const
  {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
      throw std::runtime_error("Could not open TabulatedFunction file for writing");
    }

    file.write(file_magic_.data(),
               static_cast<std::streamsize>(file_magic_.size()));
    const std::uint32_t scalar_size = sizeof(T);
    const std::uint32_t digits = std::numeric_limits<T>::digits;
    const std::uint32_t max_exponent = std::numeric_limits<T>::max_exponent;
    const std::uint8_t byte_order =
      std::endian::native == std::endian::little ? 1 :
      std::endian::native == std::endian::big ? 2 : 0;
    const std::uint64_t count = points_.size();
    WriteValue(file, file_version_);
    WriteValue(file, scalar_size);
    WriteValue(file, digits);
    WriteValue(file, max_exponent);
    WriteValue(file, byte_order);
    WriteValue(file, count);
    file.write(reinterpret_cast<const char*>(points_.data()),
               static_cast<std::streamsize>(points_.size() * sizeof(Point)));
    if (!file) {
      throw std::runtime_error("Failed to write TabulatedFunction file");
    }
  }
};

#endif