import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import spaces
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D


class PhysicsMagnetsEnv(gym.Env):
    """
    2D square [0,1]x[0,1] with N rigid hexagons (free rotation + collisions).

    Each hexagon has 6 sides; each side has a discrete polarity:
      0 = off
      1 = red
      2 = blue

    Action:
      - controls ALL sides of ALL hexagons: MultiDiscrete([3]*(n*6)), reshaped to (n,6)

    Magnet interaction (side-to-side, geometry-aware, produces force + torque):
      - For each pair (i,j), determine which side of i faces j and which side of j faces i
        in a *self-consistent* way using side-center geometry.
      - If either facing side is OFF => no magnetic force
      - If both ON:
          same polarity => repel
          opposite polarity => attract
      - Force is applied at each facing side center => creates torque (rotation)

    Rigid body physics:
      - state per hex: position, velocity, orientation theta, angular velocity omega
      - integrates with semi-implicit Euler
      - collisions: convex polygon SAT + impulse (restitution + friction) + positional correction
      - walls: keep inside [0,1]^2 with impulse-like reflection based on polygon AABB

    Reward:
      - negative sum of distances from each hexagon center to the target point
    Termination:
      - all centers within goal_radius of target
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    STATE_OFF = 0
    STATE_RED = 1
    STATE_BLUE = 2

    def __init__(
        self,
        n_points=3,
        render_mode=None,
        max_steps=200,
        goal_radius=0.03,
        hex_radius=0.03,
        interaction_radius=0.35,
        interaction_strength=2.5,     # force scale (physics)
        dt=0.02,
        mass=1.0,
        linear_damping=0.995,
        angular_damping=0.995,
        restitution=0.2,
        friction=0.6,
        collision_pos_correction=0.8,
        collision_slop=1e-4,
        seed=None,
    ):
        super().__init__()

        self.n = int(n_points)
        assert self.n >= 1

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.r = float(interaction_radius)
        self.k = float(interaction_strength)
        self.goal_radius = float(goal_radius)

        self.R = float(hex_radius)              # circumradius
        self.apothem = self.R * np.sqrt(3) / 2  # center to side midpoint

        self.dt = float(dt)
        self.m = float(mass)
        self.inv_m = 1.0 / self.m if self.m > 0 else 0.0

        # Moment of inertia (approx)
        self.I = 0.5 * self.m * (self.R ** 2)
        self.inv_I = 1.0 / self.I if self.I > 0 else 0.0

        self.lin_damp = float(linear_damping)
        self.ang_damp = float(angular_damping)

        self.e = float(restitution)
        self.mu = float(friction)

        self.pos_correction = float(collision_pos_correction)
        self.collision_slop = float(collision_slop)

        self.action_space = spaces.MultiDiscrete([3] * (self.n * 6))

        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(0.0, 1.0, shape=(self.n, 2), dtype=np.float32),
                "vel": spaces.Box(-np.inf, np.inf, shape=(self.n, 2), dtype=np.float32),
                "theta": spaces.Box(-np.pi, np.pi, shape=(self.n,), dtype=np.float32),
                "omega": spaces.Box(-np.inf, np.inf, shape=(self.n,), dtype=np.float32),
                "sides": spaces.MultiDiscrete([[3] * 6] * self.n),
                "target": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
                "t": spaces.Box(0, self.max_steps, shape=(1,), dtype=np.int32),
            }
        )

        self.t = 0
        self.pos = None
        self.vel = None
        self.theta = None
        self.omega = None
        self.sides = None
        self.target = None

        self._fig = None
        self._ax = None
        self._goal_scat = None
        self._text = None
        self._canvas = None
        self._hex_patches = None
        self._edge_lines = None

        if seed is not None:
            self.reset(seed=seed)

    def _get_obs(self):
        return {
            "pos": self.pos.astype(np.float32, copy=False),
            "vel": self.vel.astype(np.float32, copy=False),
            "theta": self.theta.astype(np.float32, copy=False),
            "omega": self.omega.astype(np.float32, copy=False),
            "sides": self.sides.astype(np.int64, copy=False),
            "target": self.target.astype(np.float32, copy=False),
            "t": np.array([self.t], dtype=np.int32),
        }

    @staticmethod
    def _wrap_pi(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _hex_vertices(self, center: np.ndarray, theta: float) -> np.ndarray:
        angles = theta + np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
        return np.stack(
            [center[0] + self.R * np.cos(angles), center[1] + self.R * np.sin(angles)],
            axis=1,
        )

    def _state_to_color(self, s: int) -> str:
        if s == self.STATE_OFF:
            return "black"
        if s == self.STATE_RED:
            return "red"
        return "blue"

    def _side_normal_world(self, theta: float, k: int) -> np.ndarray:
        ang = theta + k * (np.pi / 3.0)
        return np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)

    def _facing_side_index(self, theta: float, dir_world: np.ndarray) -> int:
        phi = float(np.arctan2(dir_world[1], dir_world[0]))
        phi_local = (phi - theta) % (2.0 * np.pi)
        return int(np.round(phi_local / (np.pi / 3.0))) % 6

    def _side_center(self, i: int, side_idx: int) -> np.ndarray:
        n = self._side_normal_world(float(self.theta[i]), int(side_idx))
        return self.pos[i].astype(np.float64) + self.apothem * n

    def _pick_facing_sides_consistent(self, i: int, j: int):
        """
        Compute (si, sj, pi, pj, dirc, cd) such that:
          - si is the side of i that faces j *based on side-center geometry*
          - sj is the side of j that faces i *based on side-center geometry*
        We do a short fixed-point refinement:
          1) si from center-to-center direction
          2) pi from si
          3) sj from (pi - center_j)
          4) pj from sj
          5) refine si from (pj - center_i)
          6) recompute pi
        This removes the major "wrong edge chosen" artifacts when rotated.
        """
        ci = self.pos[i].astype(np.float64)
        cj = self.pos[j].astype(np.float64)
        dvec = cj - ci
        d = float(np.linalg.norm(dvec))
        if d < 1e-12:
            return None

        u = dvec / d

        # initial guess
        si = self._facing_side_index(float(self.theta[i]), u)
        pi = ci + self.apothem * self._side_normal_world(float(self.theta[i]), si)

        vj = pi - cj
        vj_norm = float(np.linalg.norm(vj))
        if vj_norm < 1e-12:
            return None
        sj = self._facing_side_index(float(self.theta[j]), vj / vj_norm)
        pj = cj + self.apothem * self._side_normal_world(float(self.theta[j]), sj)

        # refine si using pj
        vi = pj - ci
        vi_norm = float(np.linalg.norm(vi))
        if vi_norm < 1e-12:
            return None
        si = self._facing_side_index(float(self.theta[i]), vi / vi_norm)
        pi = ci + self.apothem * self._side_normal_world(float(self.theta[i]), si)

        cvec = pj - pi
        cd = float(np.linalg.norm(cvec))
        if cd < 1e-12:
            return None
        dirc = cvec / cd

        return int(si), int(sj), pi, pj, dirc, cd

    # ----------------------- reset -----------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        self.pos = self.np_random.random((self.n, 2), dtype=np.float32)
        margin = self.R + 1e-3
        self.pos[:, 0] = np.clip(self.pos[:, 0], margin, 1.0 - margin)
        self.pos[:, 1] = np.clip(self.pos[:, 1], margin, 1.0 - margin)

        self.vel = np.zeros((self.n, 2), dtype=np.float32)
        self.theta = self.np_random.uniform(-np.pi, np.pi, size=(self.n,)).astype(np.float32)
        self.omega = np.zeros((self.n,), dtype=np.float32)

        self.sides = self.np_random.integers(0, 3, size=(self.n, 6), dtype=np.int64)
        self.target = self.np_random.random(2, dtype=np.float32)

        for _ in range(6):
            if not self._solve_all_collisions():
                break
            self._solve_walls()

        return self._get_obs(), {}

    # ----------------------- SAT collision utilities -----------------------

    @staticmethod
    def _poly_axes(verts: np.ndarray) -> np.ndarray:
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        n = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        return normals / n

    @staticmethod
    def _project(verts: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        dots = verts @ axis
        return float(dots.min()), float(dots.max())

    def _sat_mtv(self, A: np.ndarray, B: np.ndarray, center_dir: np.ndarray):
        axes = np.vstack([self._poly_axes(A), self._poly_axes(B)])
        min_overlap = np.inf
        best_axis = None

        for axis in axes:
            a_min, a_max = self._project(A, axis)
            b_min, b_max = self._project(B, axis)
            overlap = min(a_max, b_max) - max(a_min, b_min)
            if overlap <= 0.0:
                return None, None
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = axis.copy()

        if np.dot(best_axis, center_dir) < 0:
            best_axis = -best_axis
        return float(min_overlap), best_axis

    @staticmethod
    def _support_points(verts: np.ndarray, axis: np.ndarray, take_max: bool):
        dots = verts @ axis
        extreme = dots.max() if take_max else dots.min()
        mask = np.isclose(dots, extreme, rtol=0.0, atol=1e-6)
        pts = verts[mask]
        if pts.shape[0] == 0:
            idx = int(np.argmax(dots) if take_max else np.argmin(dots))
            return verts[idx]
        return pts.mean(axis=0)

    def _resolve_pair_collision(self, i: int, j: int) -> bool:
        Ai = self._hex_vertices(self.pos[i], float(self.theta[i]))
        Aj = self._hex_vertices(self.pos[j], float(self.theta[j]))

        center_dir = (self.pos[j] - self.pos[i]).astype(np.float64)
        overlap, n = self._sat_mtv(Ai, Aj, center_dir)
        if overlap is None:
            return False

        pa = self._support_points(Ai, n, take_max=True)
        pb = self._support_points(Aj, n, take_max=False)
        contact = 0.5 * (pa + pb)

        pen = max(0.0, overlap - self.collision_slop)
        if pen > 0:
            corr = self.pos_correction * pen * n
            self.pos[i] -= (self.inv_m / (self.inv_m + self.inv_m)) * corr
            self.pos[j] += (self.inv_m / (self.inv_m + self.inv_m)) * corr

        ri = contact - self.pos[i].astype(np.float64)
        rj = contact - self.pos[j].astype(np.float64)

        vi = self.vel[i].astype(np.float64) + np.array([-self.omega[i] * ri[1], self.omega[i] * ri[0]], dtype=np.float64)
        vj = self.vel[j].astype(np.float64) + np.array([-self.omega[j] * rj[1], self.omega[j] * rj[0]], dtype=np.float64)
        rv = vj - vi

        vn = float(np.dot(rv, n))
        if vn > 0:
            return True

        rn_i = self._cross2(ri, n)
        rn_j = self._cross2(rj, n)
        inv_mass_sum = self.inv_m + self.inv_m + (rn_i * rn_i) * self.inv_I + (rn_j * rn_j) * self.inv_I

        jn = -(1.0 + self.e) * vn
        jn /= (inv_mass_sum + 1e-12)

        impulse_n = jn * n

        self.vel[i] -= (self.inv_m * impulse_n).astype(np.float32)
        self.vel[j] += (self.inv_m * impulse_n).astype(np.float32)

        self.omega[i] -= np.float32(self.inv_I * self._cross2(ri, impulse_n))
        self.omega[j] += np.float32(self.inv_I * self._cross2(rj, impulse_n))

        rv2 = (self.vel[j].astype(np.float64) + np.array([-self.omega[j] * rj[1], self.omega[j] * rj[0]], dtype=np.float64)) - \
              (self.vel[i].astype(np.float64) + np.array([-self.omega[i] * ri[1], self.omega[i] * ri[0]], dtype=np.float64))
        t = rv2 - np.dot(rv2, n) * n
        t_norm = float(np.linalg.norm(t))
        if t_norm > 1e-12:
            t /= t_norm
            vt = float(np.dot(rv2, t))

            rt_i = self._cross2(ri, t)
            rt_j = self._cross2(rj, t)
            inv_mass_t = self.inv_m + self.inv_m + (rt_i * rt_i) * self.inv_I + (rt_j * rt_j) * self.inv_I

            jt = -vt / (inv_mass_t + 1e-12)
            jt = float(np.clip(jt, -self.mu * jn, self.mu * jn))

            impulse_t = jt * t

            self.vel[i] -= (self.inv_m * impulse_t).astype(np.float32)
            self.vel[j] += (self.inv_m * impulse_t).astype(np.float32)

            self.omega[i] -= np.float32(self.inv_I * self._cross2(ri, impulse_t))
            self.omega[j] += np.float32(self.inv_I * self._cross2(rj, impulse_t))

        return True

    def _solve_all_collisions(self) -> bool:
        collided_any = False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self._resolve_pair_collision(i, j):
                    collided_any = True
        return collided_any

    def _solve_walls(self):
        for i in range(self.n):
            verts = self._hex_vertices(self.pos[i], float(self.theta[i]))
            minx, miny = verts.min(axis=0)
            maxx, maxy = verts.max(axis=0)

            if minx < 0.0:
                self.pos[i, 0] += np.float32(-minx)
                self.vel[i, 0] = np.float32(-self.e * self.vel[i, 0])
                self.vel[i, 1] = np.float32((1.0 - self.mu * 0.2) * self.vel[i, 1])
                self.omega[i] = np.float32((1.0 - self.mu * 0.15) * self.omega[i])

            if maxx > 1.0:
                self.pos[i, 0] -= np.float32(maxx - 1.0)
                self.vel[i, 0] = np.float32(-self.e * self.vel[i, 0])
                self.vel[i, 1] = np.float32((1.0 - self.mu * 0.2) * self.vel[i, 1])
                self.omega[i] = np.float32((1.0 - self.mu * 0.15) * self.omega[i])

            if miny < 0.0:
                self.pos[i, 1] += np.float32(-miny)
                self.vel[i, 1] = np.float32(-self.e * self.vel[i, 1])
                self.vel[i, 0] = np.float32((1.0 - self.mu * 0.2) * self.vel[i, 0])
                self.omega[i] = np.float32((1.0 - self.mu * 0.15) * self.omega[i])

            if maxy > 1.0:
                self.pos[i, 1] -= np.float32(maxy - 1.0)
                self.vel[i, 1] = np.float32(-self.e * self.vel[i, 1])
                self.vel[i, 0] = np.float32((1.0 - self.mu * 0.2) * self.vel[i, 0])
                self.omega[i] = np.float32((1.0 - self.mu * 0.15) * self.omega[i])

    # ----------------------- Dynamics step -----------------------

    def step(self, action):
        self.t += 1

        a = np.asarray(action, dtype=np.int64).reshape(self.n, 6)
        self.sides[:, :] = a

        F = np.zeros((self.n, 2), dtype=np.float64)
        Tau = np.zeros((self.n,), dtype=np.float64)

        # Magnetic interactions (force + torque at side centers)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dvec = (self.pos[j] - self.pos[i]).astype(np.float64)
                d = float(np.linalg.norm(dvec))
                if d < 1e-8 or d > self.r:
                    continue

                picked = self._pick_facing_sides_consistent(i, j)
                if picked is None:
                    continue
                si, sj, pi, pj, dirc, cd = picked

                a_i = int(self.sides[i, si])
                a_j = int(self.sides[j, sj])
                if a_i == self.STATE_OFF or a_j == self.STATE_OFF:
                    continue

                # same polarity repel; opposite attract
                same = (a_i == a_j)
                sign = -1.0 if same else 1.0  # with dirc = (pj - pi)/||...||, this is correct

                # use side-center distance for falloff (not center distance)
                w = 1.0 - (cd / self.r)
                if w <= 0.0:
                    continue

                mag = sign * self.k * w
                fij = mag * dirc  # force on i at pi

                F[i] += fij
                F[j] -= fij

                ri = (pi - self.pos[i].astype(np.float64))
                rj = (pj - self.pos[j].astype(np.float64))
                Tau[i] += self._cross2(ri, fij)
                Tau[j] += self._cross2(rj, -fij)

        dt = self.dt
        self.vel = (self.vel.astype(np.float64) + (F * self.inv_m) * dt).astype(np.float32)
        self.omega = (self.omega.astype(np.float64) + (Tau * self.inv_I) * dt).astype(np.float32)

        self.vel *= np.float32(self.lin_damp)
        self.omega *= np.float32(self.ang_damp)

        self.pos = (self.pos.astype(np.float64) + self.vel.astype(np.float64) * dt).astype(np.float32)
        self.theta = self._wrap_pi(self.theta.astype(np.float64) + self.omega.astype(np.float64) * dt).astype(np.float32)

        for _ in range(6):
            any_col = self._solve_all_collisions()
            self._solve_walls()
            if not any_col:
                break

        dists = np.linalg.norm(self.pos - self.target[None, :], axis=1)
        reward = -float(np.sum(dists))

        terminated = bool(np.all(dists <= self.goal_radius))
        truncated = self.t >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {"dists": dists}

    # ----------------------- Render -----------------------

    def render(self, size: float | None = 6):
        if self.render_mode is None:
            return None

        if self._fig is None:
            if self.render_mode == "rgb_array":
                self._fig = Figure(figsize=(size, size), dpi=140)
                self._canvas = FigureCanvas(self._fig)
                self._ax = self._fig.add_subplot(111)
            else:
                self._fig, self._ax = plt.subplots(figsize=(size, size), dpi=140)

            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
            self._ax.set_aspect("equal")
            self._ax.grid(alpha=0.25)

            self._hex_patches = []
            self._edge_lines = []

            for i in range(self.n):
                verts = self._hex_vertices(self.pos[i], float(self.theta[i]))
                poly = Polygon(verts, closed=True, facecolor="none", edgecolor="none")
                self._ax.add_patch(poly)
                self._hex_patches.append(poly)

                lines_i = []
                for k in range(6):
                    p0 = verts[k]
                    p1 = verts[(k + 1) % 6]
                    line = Line2D(
                        [p0[0], p1[0]],
                        [p0[1], p1[1]],
                        linewidth=2.5,
                        color=self._state_to_color(int(self.sides[i, k])),
                    )
                    self._ax.add_line(line)
                    lines_i.append(line)
                self._edge_lines.append(lines_i)

            self._goal_scat = self._ax.scatter(
                [self.target[0]],
                [self.target[1]],
                s=220,
                marker="X",
                edgecolors="k",
                linewidths=2,
            )
            self._text = self._ax.text(0.02, 1.02, "", transform=self._ax.transAxes)

        for i in range(self.n):
            verts = self._hex_vertices(self.pos[i], float(self.theta[i]))
            self._hex_patches[i].set_xy(verts)

            for k in range(6):
                p0 = verts[k]
                p1 = verts[(k + 1) % 6]
                self._edge_lines[i][k].set_data([p0[0], p1[0]], [p0[1], p1[1]])
                self._edge_lines[i][k].set_color(self._state_to_color(int(self.sides[i, k])))

        self._goal_scat.set_offsets(self.target[None, :])
        self._text.set_text(f"t={self.t}")

        self._fig.canvas.draw()

        if self.render_mode == "human":
            plt.pause(0.001)
            return None

        buf = np.asarray(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf[:, :, :3]

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._goal_scat = None
            self._text = None
            self._canvas = None
            self._hex_patches = None
            self._edge_lines = None


env = PhysicsMagnetsEnv(render_mode="rgb_array", seed=42, n_points=20)
obs, _ = env.reset()

plt.imshow(env.render())
plt.axis("off")
plt.show()
