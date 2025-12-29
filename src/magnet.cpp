#include "magnet.h"

#include <cmath>

namespace {
int polarity(uint8_t mode) {
  if (mode == 1) {
    return 1;
  }
  if (mode == 2) {
    return -1;
  }
  return 0;
}

Vec2 side_center(const HexBody &body, float apothem, int side_index) {
  Vec2 n = side_normal_world(body.theta, side_index);
  return body.pos + n * apothem;
}

std::pair<int, int> facing_sides_consistent(const HexBody &a, const HexBody &b,
                                            float apothem) {
  Vec2 d = b.pos - a.pos;
  float dist = length(d);
  if (dist < 1e-6f) {
    return {0, 0};
  }
  Vec2 dir = d / dist;
  int side_a = facing_side_index(a.theta, dir);
  Vec2 side_a_center = side_center(a, apothem, side_a);
  Vec2 dir_b = normalize(side_a_center - b.pos);
  int side_b = facing_side_index(b.theta, dir_b);
  Vec2 side_b_center = side_center(b, apothem, side_b);
  Vec2 dir_a_refine = normalize(side_b_center - a.pos);
  side_a = facing_side_index(a.theta, dir_a_refine);
  return {side_a, side_b};
}
}  // namespace

Vec2 side_normal_world(float theta, int side_index) {
  float ang = theta + static_cast<float>(side_index) * static_cast<float>(M_PI) / 3.0f;
  return {std::cos(ang), std::sin(ang)};
}

int facing_side_index(float theta, const Vec2 &dir_world) {
  float phi = std::atan2(dir_world.y, dir_world.x);
  float phi_local = std::fmod(phi - theta + 2.0f * static_cast<float>(M_PI),
                              2.0f * static_cast<float>(M_PI));
  float sector = phi_local / (static_cast<float>(M_PI) / 3.0f);
  int idx = static_cast<int>(std::round(sector)) % 6;
  if (idx < 0) {
    idx += 6;
  }
  return idx;
}

void compute_magnetic_forces(const std::vector<HexBody> &bodies,
                             const MagnetParams &params,
                             std::vector<Vec2> &forces,
                             std::vector<float> &torques) {
  size_t n = bodies.size();
  for (size_t i = 0; i < n; ++i) {
    forces[i] = {0.0f, 0.0f};
    torques[i] = 0.0f;
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const HexBody &a = bodies[i];
      const HexBody &b = bodies[j];
      Vec2 d = b.pos - a.pos;
      float dist = length(d);
      if (dist < 1e-6f) {
        continue;
      }
      Vec2 dir = d / dist;

      auto [side_a, side_b] = facing_sides_consistent(a, b, params.apothem);
      int pol_a = polarity(a.modes[side_a]);
      int pol_b = polarity(b.modes[side_b]);
      if (pol_a == 0 || pol_b == 0) {
        continue;
      }

      Vec2 normal_a = side_normal_world(a.theta, side_a);
      Vec2 normal_b = side_normal_world(b.theta, side_b);

      float align_a = std::max(0.0f, dot(normal_a, dir));
      float align_b = std::max(0.0f, dot(normal_b, dir * -1.0f));
      float alignment = align_a * align_b;
      if (alignment <= 0.0f) {
        continue;
      }

      float magnitude = params.strength * alignment / (dist * dist + 1e-4f);
      int pol_prod = pol_a * pol_b;
      Vec2 force = dir * (-static_cast<float>(pol_prod) * magnitude);

      Vec2 point_a = side_center(a, params.apothem, side_a);
      Vec2 point_b = side_center(b, params.apothem, side_b);

      forces[i] += force;
      forces[j] -= force;

      Vec2 ra = point_a - a.pos;
      Vec2 rb = point_b - b.pos;
      torques[i] += cross(ra, force);
      torques[j] += cross(rb, force * -1.0f);
    }
  }
}
