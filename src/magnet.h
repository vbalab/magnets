#pragma once

#include <cstdint>
#include <vector>

#include "env.h"
#include "types.h"

struct MagnetParams {
  float strength = 2.5f;
  float apothem = 0.025f;
};

void compute_magnetic_forces(const std::vector<HexBody> &bodies,
                             const MagnetParams &params,
                             std::vector<Vec2> &forces,
                             std::vector<float> &torques);

int facing_side_index(float theta, const Vec2 &dir_world);
Vec2 side_normal_world(float theta, int side_index);
