#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "env.h"
#include "types.h"

class Renderer {
 public:
  Renderer();
  ~Renderer();

  bool init(int width, int height, const std::string &title);
  void shutdown();
  void draw(const std::vector<HexBody> &bodies, float radius);
  bool is_ready() const { return ready_; }

 private:
  bool ready_ = false;

#ifdef MAGNET_USE_SDL
  void *window_ = nullptr;
  void *renderer_ = nullptr;
#endif
};
