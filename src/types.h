#pragma once

#include <cstdint>
#include <cmath>

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;

  Vec2() = default;
  Vec2(float x_, float y_) : x(x_), y(y_) {}

  Vec2 operator+(const Vec2 &o) const { return {x + o.x, y + o.y}; }
  Vec2 operator-(const Vec2 &o) const { return {x - o.x, y - o.y}; }
  Vec2 operator*(float s) const { return {x * s, y * s}; }
  Vec2 operator/(float s) const { return {x / s, y / s}; }

  Vec2 &operator+=(const Vec2 &o) {
    x += o.x;
    y += o.y;
    return *this;
  }

  Vec2 &operator-=(const Vec2 &o) {
    x -= o.x;
    y -= o.y;
    return *this;
  }
};

inline float dot(const Vec2 &a, const Vec2 &b) { return a.x * b.x + a.y * b.y; }

inline float cross(const Vec2 &a, const Vec2 &b) { return a.x * b.y - a.y * b.x; }

inline float length(const Vec2 &v) { return std::sqrt(dot(v, v)); }

inline Vec2 normalize(const Vec2 &v) {
  float len = length(v);
  if (len < 1e-8f) {
    return {0.0f, 0.0f};
  }
  return v / len;
}

inline Vec2 perp(const Vec2 &v) { return {-v.y, v.x}; }
