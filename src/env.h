#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "types.h"

struct HexBody {
  Vec2 pos;
  Vec2 vel;
  float theta = 0.0f;
  float omega = 0.0f;
  uint8_t modes[6] = {0, 0, 0, 0, 0, 0};
};

struct EnvConfig {
  uint32_t num_bodies = 3;
  uint32_t history_len = 4;
  float dt = 0.02f;
  uint32_t max_steps = 200;
  float hex_radius = 0.03f;
  uint32_t seed = 0;
  float magnetic_strength = 2.5f;
  float restitution = 0.2f;
  float friction = 0.6f;
  float linear_damping = 0.995f;
  float angular_damping = 0.995f;
};

struct EpisodeContext {
  uint32_t step_count = 0;
  uint32_t episode_id = 0;
  float episode_return = 0.0f;
};

struct RewardDone {
  float reward = 0.0f;
  bool done = false;
};

class MagnetEnv {
 public:
  explicit MagnetEnv(const EnvConfig &config);

  void reset(uint32_t seed);
  RewardDone step(const std::vector<uint8_t> &modes);

  const std::vector<HexBody> &bodies() const { return bodies_; }
  const EpisodeContext &episode() const { return episode_; }
  uint32_t history_len() const { return config_.history_len; }
  uint32_t num_bodies() const { return config_.num_bodies; }
  float hex_radius() const { return config_.hex_radius; }

  uint32_t frame_stride_bytes() const { return frame_stride_bytes_; }
  void write_history(std::vector<uint8_t> &out) const;

  void set_render_enabled(bool enabled) { render_enabled_ = enabled; }
  bool render_enabled() const { return render_enabled_; }

  uint32_t collision_count() const { return collision_count_; }
  float last_reward() const { return last_reward_; }
  bool last_done() const { return last_done_; }

 private:
  void initialize_bodies();
  void fill_history_with_current();
  void push_history_frame();
  void build_frame(std::vector<uint8_t> &frame) const;

  RewardDone compute_reward_done();

  EnvConfig config_{};
  std::vector<HexBody> bodies_;
  EpisodeContext episode_{};
  std::mt19937 rng_;

  float apothem_ = 0.0f;
  float mass_ = 1.0f;
  float inv_mass_ = 1.0f;
  float inertia_ = 1.0f;
  float inv_inertia_ = 1.0f;

  uint32_t collision_count_ = 0;
  float last_reward_ = 0.0f;
  bool last_done_ = false;

  std::vector<uint8_t> history_;
  uint32_t frame_stride_bytes_ = 0;
  uint32_t history_head_ = 0;

  bool render_enabled_ = false;
};
