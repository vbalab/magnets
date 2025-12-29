#include "render.h"

#include <cmath>

#ifdef MAGNET_USE_SDL
#include <SDL2/SDL.h>
#endif

Renderer::Renderer() = default;

Renderer::~Renderer() { shutdown(); }

bool Renderer::init(int width, int height, const std::string &title) {
#ifdef MAGNET_USE_SDL
  if (ready_) {
    return true;
  }
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    return false;
  }
  SDL_Window *window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, width, height,
                                        SDL_WINDOW_SHOWN);
  if (!window) {
    SDL_Quit();
    return false;
  }
  SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (!renderer) {
    SDL_DestroyWindow(window);
    SDL_Quit();
    return false;
  }
  window_ = window;
  renderer_ = renderer;
  ready_ = true;
  return true;
#else
  (void)width;
  (void)height;
  (void)title;
  return false;
#endif
}

void Renderer::shutdown() {
#ifdef MAGNET_USE_SDL
  if (!ready_) {
    return;
  }
  SDL_Renderer *renderer = static_cast<SDL_Renderer *>(renderer_);
  SDL_Window *window = static_cast<SDL_Window *>(window_);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
#endif
  ready_ = false;
  window_ = nullptr;
  renderer_ = nullptr;
}

void Renderer::draw(const std::vector<HexBody> &bodies, float radius) {
#ifdef MAGNET_USE_SDL
  if (!ready_) {
    return;
  }
  SDL_Renderer *renderer = static_cast<SDL_Renderer *>(renderer_);
  SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
  SDL_RenderClear(renderer);

  int width = 0;
  int height = 0;
  SDL_GetRendererOutputSize(renderer, &width, &height);

  auto to_screen = [&](const Vec2 &p) {
    return SDL_Point{static_cast<int>(p.x * width),
                     static_cast<int>((1.0f - p.y) * height)};
  };

  SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
  SDL_Rect bounds{0, 0, width, height};
  SDL_RenderDrawRect(renderer, &bounds);

  float angle_step = static_cast<float>(M_PI) / 3.0f;
  for (const auto &body : bodies) {
    SDL_Point points[7];
    for (int i = 0; i < 6; ++i) {
      float ang = body.theta + angle_step * static_cast<float>(i);
      Vec2 v{body.pos.x + radius * std::cos(ang), body.pos.y + radius * std::sin(ang)};
      points[i] = to_screen(v);
    }
    points[6] = points[0];
    SDL_SetRenderDrawColor(renderer, 180, 180, 180, 255);
    SDL_RenderDrawLines(renderer, points, 7);

    for (int i = 0; i < 6; ++i) {
      uint8_t mode = body.modes[i];
      if (mode == 0) {
        SDL_SetRenderDrawColor(renderer, 80, 80, 80, 255);
      } else if (mode == 1) {
        SDL_SetRenderDrawColor(renderer, 220, 40, 40, 255);
      } else {
        SDL_SetRenderDrawColor(renderer, 40, 80, 220, 255);
      }
      SDL_RenderDrawLine(renderer, points[i].x, points[i].y,
                         points[(i + 1) % 6].x, points[(i + 1) % 6].y);
    }
  }

  SDL_RenderPresent(renderer);

  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      ready_ = false;
    }
  }
#else
  (void)bodies;
  (void)radius;
#endif
}
