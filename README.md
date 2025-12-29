# Magnet Env Server

Standalone C++ magnet-physics environment server with a thin Python client. The C++ process owns
physics, reward/done logic, and observation history. Python sends actions and receives observations
via ZeroMQ.

## Build

```bash
cmake -S . -B build
cmake --build build
```

Rendering uses SDL2 if available; disable with `-DMAGNET_USE_SDL=OFF`.

## Run

Start the server:

```bash
./build/magnet_env_server --endpoint tcp://127.0.0.1:5555
```

Test client:

```bash
python client.py --spawn --steps 5
```

## IPC Protocol

Transport: ZeroMQ REQ/REP. All messages are little-endian with header:

```
u32 msg_type
u32 payload_len
payload bytes
```

### Client → Server

- `HELLO (1)` no payload
- `INIT (2)` payload:
  - u32 N
  - u32 K
  - f32 dt
  - u32 max_steps
  - f32 hex_radius
  - u32 seed
  - u32 render_enabled
- `RESET (3)` payload: u32 seed (0 = random)
- `STEP (4)` payload: `N*6` bytes (uint8) side modes
- `SET_RENDER (5)` payload: u8 enabled
- `CLOSE (6)` no payload

### Server → Client

- `OK (100)` payload varies
- `ERR (101)` payload: u32 error_code + utf-8 string

### Observation Payload (RESET / STEP)

```
u32 N
u32 K
u32 frame_stride_bytes
K frames (back-to-back)
float32 reward
u32 done (0/1)
u32 step_count
float32 episode_return
u32 episode_id
u32 collision_count
```

Each frame has `N` hex records:

```
float32 x
float32 y
float32 theta
uint8 m0..m5
```

## Project Layout

```
src/
  main.cpp
  server.cpp / server.h
  env.cpp / env.h
  physics.cpp / physics.h
  magnet.cpp / magnet.h
  render.cpp / render.h
  types.h
```
