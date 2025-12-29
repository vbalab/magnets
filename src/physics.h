#pragma once

#include <cstdint>
#include <vector>

#include "types.h"
#include "env.h"

struct CollisionInfo {
  bool hit = false;
  Vec2 normal;
  float penetration = 0.0f;
  Vec2 contact_point;
};

struct PhysicsParams {
  float dt = 0.02f;
  float hex_radius = 0.03f;
  float apothem = 0.025f;
  float mass = 1.0f;
  float inv_mass = 1.0f;
  float inertia = 1.0f;
  float inv_inertia = 1.0f;
  float restitution = 0.2f;
  float friction = 0.6f;
  float linear_damping = 0.995f;
  float angular_damping = 0.995f;
};

void integrate(std::vector<HexBody> &bodies, const std::vector<Vec2> &forces,
               const std::vector<float> &torques, const PhysicsParams &params);

CollisionInfo sat_collision(const std::vector<Vec2> &a, const std::vector<Vec2> &b,
                            const Vec2 &center_a, const Vec2 &center_b);

void resolve_collision(HexBody &a, HexBody &b, const CollisionInfo &info,
                       const PhysicsParams &params);

uint32_t resolve_wall_collisions(HexBody &body, const std::vector<Vec2> &verts,
                                 const PhysicsParams &params);

std::vector<Vec2> hex_vertices(const HexBody &body, float radius);
