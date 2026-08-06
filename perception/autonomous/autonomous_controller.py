import time


class PIDController:
    """
    PID (Proportional-Integral-Derivative) controller.
    Used for lane-keeping steering control.

    PID formula:
        output = Kp*error + Ki*integral + Kd*derivative

    Kp — Proportional: main correction strength
    Ki — Integral: eliminates steady-state offset
    Kd — Derivative: dampens oscillation
    """

    def __init__(self, Kp=0.8, Ki=0.01, Kd=0.15):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.prev_error    = 0.0
        self.integral      = 0.0
        self.last_time     = time.time()
        self.integral_limit = 1.0

    def compute(self, error):
        """Compute PID output for a given error. Returns [-1, 1]."""
        now = time.time()
        dt  = max(0.001, min(now - self.last_time, 0.5))
        self.last_time = now

        P = self.Kp * error

        self.integral += error * dt
        self.integral  = max(-self.integral_limit,
                             min(self.integral_limit,
                                 self.integral))
        I = self.Ki * self.integral

        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative

        self.prev_error = error

        return max(-1.0, min(1.0, P + I + D))

    def reset(self):
        """Reset controller state."""
        self.prev_error = 0.0
        self.integral   = 0.0
        self.last_time  = time.time()


class AutonomousController:
    """
    Rule-based autonomous driving controller.

    Reads fused scene state and outputs:
        steering:  [-1, 1] left to right
        throttle:  [0, 1]
        brake:     [0, 1]

    Modes:
        MODE_MANUAL     = 0  Human controls
        MODE_ASSISTED   = 1  AI brakes only
        MODE_AUTONOMOUS = 2  AI controls everything
    """

    MODE_MANUAL     = 0
    MODE_ASSISTED   = 1
    MODE_AUTONOMOUS = 2

    def __init__(self):
        self.mode = self.MODE_AUTONOMOUS

        # PID for lane keeping
        self.lane_pid = PIDController(
            Kp=0.8, Ki=0.01, Kd=0.15)

        # Adaptive cruise control
        self.acc_target_distance = 15.0  # metres
        self.acc_min_distance    = 5.0   # emergency brake
        self.acc_max_speed       = 15.0  # m/s (~54 km/h)
        self.acc_max_throttle    = 0.6

        # Current outputs
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.current_brake    = 0.0

        # Stats
        self.frames_processed = 0
        self.emergency_brakes = 0

        print("[AutonomousController] Initialised")
        print(f"  Lane PID: Kp={self.lane_pid.Kp}, "
              f"Ki={self.lane_pid.Ki}, "
              f"Kd={self.lane_pid.Kd}")
        print(f"  ACC: target={self.acc_target_distance}m, "
              f"emergency={self.acc_min_distance}m")

    def set_mode(self, mode):
        """Switch drive mode."""
        if mode != self.mode:
            self.mode = mode
            self.lane_pid.reset()
            names = {0: "MANUAL", 1: "ASSISTED",
                     2: "AUTONOMOUS"}
            print(f"[AutonomousController] Mode → "
                  f"{names.get(mode, '?')}")

    def compute(self, scene_state, lane_offset, lane_detected):
        """
        Compute control signals from scene state.
        Returns dict with steering, throttle, brake, mode.
        """
        self.frames_processed += 1

        if self.mode == self.MODE_MANUAL:
            return self._manual_output()

        # Steering — lane keeping PID
        steering = self._compute_steering(
            lane_offset, lane_detected)

        # Throttle — adaptive cruise control
        throttle = self._compute_throttle(scene_state)

        # Brake — collision avoidance
        brake = self._compute_brake(scene_state)

        # Emergency brake overrides everything
        emergency = scene_state.get("emergency_brake", False)
        if emergency:
            brake    = 1.0
            throttle = 0.0
            self.emergency_brakes += 1

        # ASSISTED: AI brakes only, no steering override
        if self.mode == self.MODE_ASSISTED:
            steering = 0.0

        self.current_steering = max(-1.0, min(1.0, steering))
        self.current_throttle = max(0.0,  min(1.0, throttle))
        self.current_brake    = max(0.0,  min(1.0, brake))

        return {
            "steering":      round(self.current_steering, 4),
            "throttle":      round(self.current_throttle, 4),
            "brake":         round(self.current_brake, 4),
            "mode":          self.mode,
            "emergency_brake": bool(emergency),
        }

    def _manual_output(self):
        return {
            "steering": 0.0, "throttle": 0.0,
            "brake": 0.0, "mode": self.MODE_MANUAL,
            "emergency_brake": False,
        }

    def _compute_steering(self, lane_offset, lane_detected):
        """Lane-keeping PID. Negate offset because
        positive offset = car right of centre = steer left."""
        if not lane_detected:
            self.lane_pid.integral *= 0.95
            return 0.0
        return self.lane_pid.compute(-lane_offset)

    def _compute_throttle(self, scene_state):
        """Adaptive cruise control throttle."""
        closest   = scene_state.get(
            "closest_vehicle_distance", 999.0)
        ego_speed = scene_state.get("ego_speed", 0.0)

        # Speed limit
        if ego_speed >= self.acc_max_speed:
            return 0.0

        speed_ratio   = ego_speed / self.acc_max_speed
        base_throttle = self.acc_max_throttle * (
            1.0 - speed_ratio * 0.3)

        if closest >= self.acc_target_distance:
            return base_throttle
        elif closest >= self.acc_min_distance:
            ratio = ((closest - self.acc_min_distance) /
                     (self.acc_target_distance -
                      self.acc_min_distance))
            return base_throttle * ratio
        else:
            return 0.0

    def _compute_brake(self, scene_state):
        """Collision avoidance braking."""
        closest     = scene_state.get(
            "closest_vehicle_distance", 999.0)
        brake_start = self.acc_target_distance * 0.6
        brake_full  = self.acc_min_distance

        if closest >= brake_start:
            return 0.0
        elif closest >= brake_full:
            ratio = 1.0 - ((closest - brake_full) /
                           (brake_start - brake_full))
            return ratio * 0.8
        else:
            return 1.0