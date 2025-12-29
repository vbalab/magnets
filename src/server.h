#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "env.h"
#include "render.h"

class EnvServer {
 public:
  explicit EnvServer(const std::string &endpoint);
  void run();

 private:
  struct Message {
    uint32_t type = 0;
    std::vector<uint8_t> payload;
  };

  bool receive_message(Message &msg);
  void send_ok(const std::vector<uint8_t> &payload);
  void send_error(uint32_t code, const std::string &message);

  void handle_hello();
  void handle_init(const std::vector<uint8_t> &payload);
  void handle_reset(const std::vector<uint8_t> &payload);
  void handle_step(const std::vector<uint8_t> &payload);
  void handle_set_render(const std::vector<uint8_t> &payload);
  void handle_close();

  std::string endpoint_;
  bool running_ = true;

  void *context_ = nullptr;
  void *socket_ = nullptr;

  bool initialized_ = false;
  EnvConfig config_{};
  std::unique_ptr<MagnetEnv> env_;
  Renderer renderer_;
};
