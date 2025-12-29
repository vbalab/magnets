#include "server.h"

#include <cstring>
#include <iostream>
#include <memory>

#include <zmq.h>

namespace {
constexpr uint32_t MSG_HELLO = 1;
constexpr uint32_t MSG_INIT = 2;
constexpr uint32_t MSG_RESET = 3;
constexpr uint32_t MSG_STEP = 4;
constexpr uint32_t MSG_SET_RENDER = 5;
constexpr uint32_t MSG_CLOSE = 6;

constexpr uint32_t MSG_OK = 100;
constexpr uint32_t MSG_ERR = 101;

void append_u32(std::vector<uint8_t> &buf, uint32_t value) {
  uint8_t bytes[4];
  std::memcpy(bytes, &value, sizeof(value));
  buf.insert(buf.end(), bytes, bytes + 4);
}

void append_f32(std::vector<uint8_t> &buf, float value) {
  uint8_t bytes[4];
  std::memcpy(bytes, &value, sizeof(value));
  buf.insert(buf.end(), bytes, bytes + 4);
}

bool read_u32(const std::vector<uint8_t> &buf, size_t &offset, uint32_t &out) {
  if (offset + 4 > buf.size()) {
    return false;
  }
  std::memcpy(&out, buf.data() + offset, 4);
  offset += 4;
  return true;
}

bool read_f32(const std::vector<uint8_t> &buf, size_t &offset, float &out) {
  if (offset + 4 > buf.size()) {
    return false;
  }
  std::memcpy(&out, buf.data() + offset, 4);
  offset += 4;
  return true;
}

std::vector<uint8_t> build_observation_payload(const MagnetEnv &env) {
  std::vector<uint8_t> payload;
  append_u32(payload, env.num_bodies());
  append_u32(payload, env.history_len());
  append_u32(payload, env.frame_stride_bytes());

  std::vector<uint8_t> history;
  env.write_history(history);
  payload.insert(payload.end(), history.begin(), history.end());

  append_f32(payload, env.last_reward());
  append_u32(payload, env.last_done() ? 1u : 0u);
  append_u32(payload, env.episode().step_count);
  append_f32(payload, env.episode().episode_return);
  append_u32(payload, env.episode().episode_id);
  append_u32(payload, env.collision_count());
  return payload;
}
}  // namespace

EnvServer::EnvServer(const std::string &endpoint) : endpoint_(endpoint) {
  context_ = zmq_ctx_new();
  socket_ = zmq_socket(context_, ZMQ_REP);
  if (zmq_bind(socket_, endpoint_.c_str()) != 0) {
    throw std::runtime_error("Failed to bind ZeroMQ socket");
  }
}

void EnvServer::run() {
  while (running_) {
    Message msg;
    if (!receive_message(msg)) {
      continue;
    }
    switch (msg.type) {
      case MSG_HELLO:
        handle_hello();
        break;
      case MSG_INIT:
        handle_init(msg.payload);
        break;
      case MSG_RESET:
        handle_reset(msg.payload);
        break;
      case MSG_STEP:
        handle_step(msg.payload);
        break;
      case MSG_SET_RENDER:
        handle_set_render(msg.payload);
        break;
      case MSG_CLOSE:
        handle_close();
        break;
      default:
        send_error(1, "Unknown message type");
        break;
    }
  }

  if (socket_) {
    zmq_close(socket_);
    socket_ = nullptr;
  }
  if (context_) {
    zmq_ctx_term(context_);
    context_ = nullptr;
  }
}

bool EnvServer::receive_message(Message &msg) {
  zmq_msg_t zmsg;
  zmq_msg_init(&zmsg);
  int rc = zmq_msg_recv(&zmsg, socket_, 0);
  if (rc < 0) {
    zmq_msg_close(&zmsg);
    return false;
  }
  size_t size = zmq_msg_size(&zmsg);
  if (size < 8) {
    zmq_msg_close(&zmsg);
    send_error(2, "Invalid message size");
    return false;
  }
  const uint8_t *data = static_cast<const uint8_t *>(zmq_msg_data(&zmsg));
  uint32_t type = 0;
  uint32_t len = 0;
  std::memcpy(&type, data, 4);
  std::memcpy(&len, data + 4, 4);
  if (size != static_cast<size_t>(8 + len)) {
    zmq_msg_close(&zmsg);
    send_error(3, "Payload length mismatch");
    return false;
  }
  msg.type = type;
  msg.payload.assign(data + 8, data + size);
  zmq_msg_close(&zmsg);
  return true;
}

void EnvServer::send_ok(const std::vector<uint8_t> &payload) {
  std::vector<uint8_t> response;
  append_u32(response, MSG_OK);
  append_u32(response, static_cast<uint32_t>(payload.size()));
  response.insert(response.end(), payload.begin(), payload.end());
  zmq_send(socket_, response.data(), response.size(), 0);
}

void EnvServer::send_error(uint32_t code, const std::string &message) {
  std::vector<uint8_t> payload;
  append_u32(payload, code);
  payload.insert(payload.end(), message.begin(), message.end());
  std::vector<uint8_t> response;
  append_u32(response, MSG_ERR);
  append_u32(response, static_cast<uint32_t>(payload.size()));
  response.insert(response.end(), payload.begin(), payload.end());
  zmq_send(socket_, response.data(), response.size(), 0);
}

void EnvServer::handle_hello() { send_ok({}); }

void EnvServer::handle_init(const std::vector<uint8_t> &payload) {
  size_t offset = 0;
  uint32_t n = 0;
  uint32_t k = 0;
  uint32_t max_steps = 0;
  float dt = 0.0f;
  float hex_radius = 0.0f;
  uint32_t seed = 0;
  uint32_t render_enabled = 0;
  if (!read_u32(payload, offset, n) || !read_u32(payload, offset, k) ||
      !read_f32(payload, offset, dt) || !read_u32(payload, offset, max_steps) ||
      !read_f32(payload, offset, hex_radius) || !read_u32(payload, offset, seed) ||
      !read_u32(payload, offset, render_enabled)) {
    send_error(4, "Invalid INIT payload");
    return;
  }

  config_.num_bodies = n;
  config_.history_len = k;
  config_.dt = dt;
  config_.max_steps = max_steps;
  config_.hex_radius = hex_radius;
  config_.seed = seed;
  env_ = std::make_unique<MagnetEnv>(config_);
  initialized_ = true;

  bool enable_render = render_enabled != 0;
  env_->set_render_enabled(enable_render);
  if (enable_render) {
    renderer_.init(640, 640, "Magnet Env");
  }

  send_ok({});
}

void EnvServer::handle_reset(const std::vector<uint8_t> &payload) {
  if (!initialized_ || !env_) {
    send_error(5, "Env not initialized");
    return;
  }
  size_t offset = 0;
  uint32_t seed = 0;
  if (!read_u32(payload, offset, seed)) {
    send_error(6, "Invalid RESET payload");
    return;
  }
  env_->reset(seed);
  std::vector<uint8_t> obs = build_observation_payload(*env_);
  if (env_->render_enabled() && renderer_.is_ready()) {
    renderer_.draw(env_->bodies(), env_->hex_radius());
  }
  send_ok(obs);
}

void EnvServer::handle_step(const std::vector<uint8_t> &payload) {
  if (!initialized_ || !env_) {
    send_error(5, "Env not initialized");
    return;
  }
  if (payload.size() != env_->num_bodies() * 6) {
    send_error(8, "Invalid STEP payload");
    return;
  }
  RewardDone rd = env_->step(payload);
  std::vector<uint8_t> obs = build_observation_payload(*env_);
  if (env_->render_enabled() && renderer_.is_ready()) {
    renderer_.draw(env_->bodies(), env_->hex_radius());
  }
  send_ok(obs);
}

void EnvServer::handle_set_render(const std::vector<uint8_t> &payload) {
  if (!initialized_ || !env_) {
    send_error(5, "Env not initialized");
    return;
  }
  if (payload.empty()) {
    send_error(7, "Invalid SET_RENDER payload");
    return;
  }
  bool enable = payload[0] != 0;
  env_->set_render_enabled(enable);
  if (enable && !renderer_.is_ready()) {
    renderer_.init(640, 640, "Magnet Env");
  }
  if (!enable && renderer_.is_ready()) {
    renderer_.shutdown();
  }
  send_ok({});
}

void EnvServer::handle_close() {
  send_ok({});
  running_ = false;
}
