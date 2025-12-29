#include "env.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "magnet.h"
#include "physics.h"

MagnetEnv::MagnetEnv(const EnvConfig &config) : config_(config) {
  apothem_ = config_.hex_radius * std::sqrt(3.0f) / 2.0f;
  mass_ = 1.0f;
  inv_mass_ = 1.0f / mass_;
  inertia_ = 0.5f * mass_ * config_.hex_radius * config_.hex_radius;
  inv_inertia_ = 1.0f / inertia_;

  bodies_.resize(config_.num_bodies);
  frame_stride_bytes_ = config_.num_bodies * (3 * sizeof(float) + 6 * sizeof(uint8_t));
  history_.resize(frame_stride_bytes_ * config_.history_len);

  reset(config_.seed);
}

void MagnetEnv::reset(uint32_t seed) {
  if (seed == 0) {
    std::random_device rd;
    rng_.seed(rd());
  } else {
    rng_.seed(seed);
  }

  episode_.step_count = 0;
  episode_.episode_return = 0.0f;
  episode_.episode_id += 1;
  collision_count_ = 0;
  last_reward_ = 0.0f;
  last_done_ = false;

  initialize_bodies();
  fill_history_with_current();
}

void MagnetEnv::initialize_bodies() {
  std::uniform_real_distribution<float> dist_pos(config_.hex_radius, 1.0f - config_.hex_radius);
  std::uniform_real_distribution<float> dist_theta(-static_cast<float>(M_PI),
                                                   static_cast<float>(M_PI));

  for (uint32_t i = 0; i < config_.num_bodies; ++i) {
    bodies_[i].pos = {dist_pos(rng_), dist_pos(rng_)};
    bodies_[i].vel = {0.0f, 0.0f};
    bodies_[i].theta = dist_theta(rng_);
    bodies_[i].omega = 0.0f;
    std::fill(std::begin(bodies_[i].modes), std::end(bodies_[i].modes), 0);
  }
}

RewardDone MagnetEnv::step(const std::vector<uint8_t> &modes) {
  if (modes.size() != bodies_.size() * 6) {
    return {0.0f, false};
  }
  for (size_t i = 0; i < bodies_.size(); ++i) {
    for (size_t j = 0; j < 6; ++j) {
      bodies_[i].modes[j] = modes[i * 6 + j];
    }
  }

  std::vector<Vec2> forces(bodies_.size());
  std::vector<float> torques(bodies_.size());

  MagnetParams magnet_params{config_.magnetic_strength, apothem_};
  compute_magnetic_forces(bodies_, magnet_params, forces, torques);

  PhysicsParams physics_params{config_.dt,
                               config_.hex_radius,
                               apothem_,
                               mass_,
                               inv_mass_,
                               inertia_,
                               inv_inertia_,
                               config_.restitution,
                               config_.friction,
                               config_.linear_damping,
                               config_.angular_damping};

  integrate(bodies_, forces, torques, physics_params);

  collision_count_ = 0;
  for (size_t i = 0; i < bodies_.size(); ++i) {
    auto verts_i = hex_vertices(bodies_[i], config_.hex_radius);
    collision_count_ += resolve_wall_collisions(bodies_[i], verts_i, physics_params);
    for (size_t j = i + 1; j < bodies_.size(); ++j) {
      auto verts_j = hex_vertices(bodies_[j], config_.hex_radius);
      CollisionInfo info = sat_collision(verts_i, verts_j, bodies_[i].pos, bodies_[j].pos);
      if (info.hit) {
        collision_count_ += 1;
      }
      resolve_collision(bodies_[i], bodies_[j], info, physics_params);
    }
  }

  RewardDone rd = compute_reward_done();
  episode_.episode_return += rd.reward;
  episode_.step_count += 1;
  if (episode_.step_count >= config_.max_steps) {
    rd.done = true;
  }
  last_reward_ = rd.reward;
  last_done_ = rd.done;

  push_history_frame();
  return rd;
}

RewardDone MagnetEnv::compute_reward_done() {
  float reward = 0.0f;
  Vec2 target{0.5f, 0.5f};
  for (const auto &body : bodies_) {
    Vec2 d = body.pos - target;
    reward -= length(d);
  }
  return {reward, false};
}

void MagnetEnv::build_frame(std::vector<uint8_t> &frame) const {
  frame.resize(frame_stride_bytes_);
  uint8_t *ptr = frame.data();
  for (const auto &body : bodies_) {
    std::memcpy(ptr, &body.pos.x, sizeof(float));
    ptr += sizeof(float);
    std::memcpy(ptr, &body.pos.y, sizeof(float));
    ptr += sizeof(float);
    std::memcpy(ptr, &body.theta, sizeof(float));
    ptr += sizeof(float);
    std::memcpy(ptr, body.modes, sizeof(uint8_t) * 6);
    ptr += sizeof(uint8_t) * 6;
  }
}

void MagnetEnv::fill_history_with_current() {
  history_head_ = 0;
  std::vector<uint8_t> frame;
  build_frame(frame);
  for (uint32_t i = 0; i < config_.history_len; ++i) {
    std::memcpy(history_.data() + i * frame_stride_bytes_, frame.data(),
                frame_stride_bytes_);
  }
}

void MagnetEnv::push_history_frame() {
  std::vector<uint8_t> frame;
  build_frame(frame);
  std::memcpy(history_.data() + history_head_ * frame_stride_bytes_, frame.data(),
              frame_stride_bytes_);
  history_head_ = (history_head_ + 1) % config_.history_len;
}

void MagnetEnv::write_history(std::vector<uint8_t> &out) const {
  out.resize(frame_stride_bytes_ * config_.history_len);
  for (uint32_t i = 0; i < config_.history_len; ++i) {
    uint32_t idx = (history_head_ + i) % config_.history_len;
    std::memcpy(out.data() + i * frame_stride_bytes_,
                history_.data() + idx * frame_stride_bytes_, frame_stride_bytes_);
  }
}
