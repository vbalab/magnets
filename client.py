import argparse
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import zmq

MSG_HELLO = 1
MSG_INIT = 2
MSG_RESET = 3
MSG_STEP = 4
MSG_SET_RENDER = 5
MSG_CLOSE = 6

MSG_OK = 100
MSG_ERR = 101


@dataclass
class StepResult:
    history: bytes
    reward: float
    done: bool
    step_count: int
    episode_return: float
    episode_id: int
    collision_count: int


class MagnetEnvClient:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555") -> None:
        self.endpoint = endpoint
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(endpoint)

    def close(self) -> None:
        self._send(MSG_CLOSE, b"")
        self.socket.close()

    def hello(self) -> None:
        self._send(MSG_HELLO, b"")

    def init(
        self,
        n: int,
        k: int,
        dt: float,
        max_steps: int,
        hex_radius: float,
        seed: int,
        render_enabled: bool,
    ) -> None:
        payload = struct.pack(
            "<II f I f I I",
            n,
            k,
            dt,
            max_steps,
            hex_radius,
            seed,
            1 if render_enabled else 0,
        )
        self._send(MSG_INIT, payload)

    def reset(self, seed: int = 0) -> StepResult:
        payload = struct.pack("<I", seed)
        return self._send(MSG_RESET, payload)

    def step(self, modes: np.ndarray) -> StepResult:
        if modes.dtype != np.uint8:
            modes = modes.astype(np.uint8)
        payload = modes.tobytes(order="C")
        return self._send(MSG_STEP, payload)

    def set_render(self, enabled: bool) -> None:
        payload = bytes([1 if enabled else 0])
        self._send(MSG_SET_RENDER, payload)

    def _send(self, msg_type: int, payload: bytes) -> StepResult:
        header = struct.pack("<II", msg_type, len(payload))
        self.socket.send(header + payload)
        reply = self.socket.recv()
        if len(reply) < 8:
            raise RuntimeError("Invalid reply")
        resp_type, length = struct.unpack_from("<II", reply, 0)
        payload = reply[8:]
        if len(payload) != length:
            raise RuntimeError("Payload length mismatch")
        if resp_type == MSG_ERR:
            code = struct.unpack_from("<I", payload, 0)[0]
            msg = payload[4:].decode("utf-8", errors="replace")
            raise RuntimeError(f"Server error {code}: {msg}")
        if resp_type != MSG_OK:
            raise RuntimeError(f"Unexpected response type {resp_type}")
        if msg_type in (MSG_RESET, MSG_STEP):
            return self._decode_observation(payload)
        return StepResult(b"", 0.0, False, 0, 0.0, 0, 0)

    @staticmethod
    def _decode_observation(payload: bytes) -> StepResult:
        offset = 0
        n, k, stride = struct.unpack_from("<III", payload, offset)
        offset += 12
        history_size = k * stride
        history = payload[offset : offset + history_size]
        offset += history_size
        reward, = struct.unpack_from("<f", payload, offset)
        offset += 4
        done_flag, = struct.unpack_from("<I", payload, offset)
        offset += 4
        step_count, = struct.unpack_from("<I", payload, offset)
        offset += 4
        episode_return, = struct.unpack_from("<f", payload, offset)
        offset += 4
        episode_id, = struct.unpack_from("<I", payload, offset)
        offset += 4
        collision_count, = struct.unpack_from("<I", payload, offset)
        return StepResult(
            history=history,
            reward=reward,
            done=bool(done_flag),
            step_count=step_count,
            episode_return=episode_return,
            episode_id=episode_id,
            collision_count=collision_count,
        )

    @staticmethod
    def decode_history(history: bytes, n: int, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        frame_stride = n * (3 * 4 + 6)
        frames = []
        for i in range(k):
            frame = history[i * frame_stride : (i + 1) * frame_stride]
            floats = np.frombuffer(frame, dtype=np.float32, count=n * 3)
            float_data = floats.reshape(n, 3)
            offset = n * 3 * 4
            modes = np.frombuffer(frame[offset:], dtype=np.uint8).reshape(n, 6)
            frames.append((float_data, modes))
        return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    proc: Optional[subprocess.Popen[str]] = None
    if args.spawn:
        proc = subprocess.Popen(["./magnet_env_server", "--endpoint", args.endpoint])
        time.sleep(0.3)

    client = MagnetEnvClient(args.endpoint)
    client.hello()
    client.init(n=3, k=4, dt=0.02, max_steps=200, hex_radius=0.03, seed=42, render_enabled=False)
    obs = client.reset(seed=123)
    print("reset", obs.step_count, obs.reward, obs.done)

    rng = np.random.default_rng(0)
    for _ in range(args.steps):
        action = rng.integers(0, 3, size=(3, 6), dtype=np.uint8)
        obs = client.step(action)
        print("step", obs.step_count, obs.reward, obs.done)

    client.close()
    if proc:
        proc.terminate()
        proc.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
