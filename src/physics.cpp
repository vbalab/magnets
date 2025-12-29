#include "physics.h"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<Vec2> hex_vertices(const HexBody &body, float radius) {
  std::vector<Vec2> verts;
  verts.reserve(6);
  float angle_step = static_cast<float>(M_PI) / 3.0f;
  for (int i = 0; i < 6; ++i) {
    float ang = body.theta + angle_step * static_cast<float>(i);
    verts.emplace_back(body.pos.x + radius * std::cos(ang),
                       body.pos.y + radius * std::sin(ang));
  }
  return verts;
}

static void project_polygon(const std::vector<Vec2> &verts, const Vec2 &axis,
                            float &min_proj, float &max_proj) {
  min_proj = std::numeric_limits<float>::infinity();
  max_proj = -std::numeric_limits<float>::infinity();
  for (const auto &v : verts) {
    float p = dot(v, axis);
    min_proj = std::min(min_proj, p);
    max_proj = std::max(max_proj, p);
  }
}

CollisionInfo sat_collision(const std::vector<Vec2> &a, const std::vector<Vec2> &b,
                            const Vec2 &center_a, const Vec2 &center_b) {
  CollisionInfo info;
  float min_overlap = std::numeric_limits<float>::infinity();
  Vec2 best_axis{0.0f, 0.0f};

  auto check_axes = [&](const std::vector<Vec2> &poly) {
    for (size_t i = 0; i < poly.size(); ++i) {
      Vec2 p1 = poly[i];
      Vec2 p2 = poly[(i + 1) % poly.size()];
      Vec2 edge = p2 - p1;
      Vec2 axis = normalize(perp(edge));
      float min_a, max_a, min_b, max_b;
      project_polygon(a, axis, min_a, max_a);
      project_polygon(b, axis, min_b, max_b);
      float overlap = std::min(max_a, max_b) - std::max(min_a, min_b);
      if (overlap <= 0.0f) {
        return false;
      }
      if (overlap < min_overlap) {
        min_overlap = overlap;
        best_axis = axis;
        if (dot(center_b - center_a, best_axis) < 0.0f) {
          best_axis = best_axis * -1.0f;
        }
      }
    }
    return true;
  };

  if (!check_axes(a)) {
    return info;
  }
  if (!check_axes(b)) {
    return info;
  }

  info.hit = true;
  info.normal = best_axis;
  info.penetration = min_overlap;

  Vec2 support_a = a[0];
  float best_a = dot(support_a, info.normal);
  for (const auto &v : a) {
    float d = dot(v, info.normal);
    if (d > best_a) {
      best_a = d;
      support_a = v;
    }
  }
  Vec2 support_b = b[0];
  float best_b = dot(support_b, info.normal * -1.0f);
  for (const auto &v : b) {
    float d = dot(v, info.normal * -1.0f);
    if (d > best_b) {
      best_b = d;
      support_b = v;
    }
  }
  info.contact_point = (support_a + support_b) * 0.5f;
  return info;
}

void integrate(std::vector<HexBody> &bodies, const std::vector<Vec2> &forces,
               const std::vector<float> &torques, const PhysicsParams &params) {
  float dt = params.dt;
  for (size_t i = 0; i < bodies.size(); ++i) {
    HexBody &body = bodies[i];
    Vec2 acc = forces[i] * params.inv_mass;
    body.vel += acc * dt;
    body.vel = body.vel * params.linear_damping;
    body.pos += body.vel * dt;

    float ang_acc = torques[i] * params.inv_inertia;
    body.omega += ang_acc * dt;
    body.omega *= params.angular_damping;
    body.theta += body.omega * dt;
  }
}

void resolve_collision(HexBody &a, HexBody &b, const CollisionInfo &info,
                       const PhysicsParams &params) {
  if (!info.hit) {
    return;
  }

  Vec2 ra = info.contact_point - a.pos;
  Vec2 rb = info.contact_point - b.pos;

  Vec2 rv = (b.vel + perp(rb) * b.omega) - (a.vel + perp(ra) * a.omega);
  float vel_along_normal = dot(rv, info.normal);
  if (vel_along_normal > 0.0f) {
    return;
  }

  float restitution = params.restitution;
  float ra_cn = cross(ra, info.normal);
  float rb_cn = cross(rb, info.normal);
  float inv_mass_sum = params.inv_mass + params.inv_mass +
                       (ra_cn * ra_cn) * params.inv_inertia +
                       (rb_cn * rb_cn) * params.inv_inertia;

  float j = -(1.0f + restitution) * vel_along_normal;
  j /= inv_mass_sum;

  Vec2 impulse = info.normal * j;
  a.vel -= impulse * params.inv_mass;
  b.vel += impulse * params.inv_mass;
  a.omega -= cross(ra, impulse) * params.inv_inertia;
  b.omega += cross(rb, impulse) * params.inv_inertia;

  Vec2 tangent = rv - info.normal * vel_along_normal;
  tangent = normalize(tangent);
  float jt = -dot(rv, tangent);
  jt /= inv_mass_sum;

  float mu = params.friction;
  Vec2 friction_impulse;
  if (std::fabs(jt) < j * mu) {
    friction_impulse = tangent * jt;
  } else {
    friction_impulse = tangent * -j * mu;
  }

  a.vel -= friction_impulse * params.inv_mass;
  b.vel += friction_impulse * params.inv_mass;
  a.omega -= cross(ra, friction_impulse) * params.inv_inertia;
  b.omega += cross(rb, friction_impulse) * params.inv_inertia;

  float percent = 0.8f;
  float slop = 1e-4f;
  Vec2 correction = info.normal *
                    (std::max(info.penetration - slop, 0.0f) /
                     (params.inv_mass + params.inv_mass) * percent);
  a.pos -= correction * params.inv_mass;
  b.pos += correction * params.inv_mass;
}

uint32_t resolve_wall_collisions(HexBody &body, const std::vector<Vec2> &verts,
                                 const PhysicsParams &params) {
  uint32_t collisions = 0;
  float min_x = 1.0f;
  float max_x = 0.0f;
  float min_y = 1.0f;
  float max_y = 0.0f;
  for (const auto &v : verts) {
    min_x = std::min(min_x, v.x);
    max_x = std::max(max_x, v.x);
    min_y = std::min(min_y, v.y);
    max_y = std::max(max_y, v.y);
  }

  if (min_x < 0.0f) {
    body.pos.x += -min_x;
    body.vel.x = -body.vel.x * params.restitution;
    collisions++;
  } else if (max_x > 1.0f) {
    body.pos.x -= (max_x - 1.0f);
    body.vel.x = -body.vel.x * params.restitution;
    collisions++;
  }

  if (min_y < 0.0f) {
    body.pos.y += -min_y;
    body.vel.y = -body.vel.y * params.restitution;
    collisions++;
  } else if (max_y > 1.0f) {
    body.pos.y -= (max_y - 1.0f);
    body.vel.y = -body.vel.y * params.restitution;
    collisions++;
  }

  return collisions;
}
