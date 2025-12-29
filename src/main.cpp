#include "server.h"

#include <iostream>

int main(int argc, char **argv) {
  std::string endpoint = "tcp://127.0.0.1:5555";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--endpoint" && i + 1 < argc) {
      endpoint = argv[++i];
    } else if (arg == "--help") {
      std::cout << "Usage: magnet_env_server [--endpoint tcp://127.0.0.1:5555]\n";
      return 0;
    }
  }

  try {
    EnvServer server(endpoint);
    std::cout << "Magnet env server listening on " << endpoint << "\n";
    server.run();
  } catch (const std::exception &ex) {
    std::cerr << "Server error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
