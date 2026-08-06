import time
import json
import socket


class PIDController:
    """
    PID (Proportional-Integral-Derivative) controller.

    Used for lane-keeping steering control.
    Converts lane offset error into a steering correction.

    PID formula:
        output = Kp*error + Ki*integral + Kd*derivative

    Where:
        error      = current measurement error
        integral   = accumulated error over time
        derivative = rate of change of error
    """

    def __init__(self, Kp=0.8, Ki=0.01, Kd=0.15):
        """
        Args:
            Kp: Proportional gain — main correction strength
            Ki: Integral gain — eliminates steady-state offset
            Kd: Derivative gain — dampens oscillation
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.prev_error  = 0.0
        self.integral    = 0.0
        self.last_time   = time.time()

        # Integral windup limit
        # Prevents integral from growing unbounded
        # if the car is stuck or can't reach centre
        self.integral_limit = 1.0

    def compute(self, error):
        """
        Compute PID output for a given error.

        Args:
            error: current error value
                   (lane offset: negative=left, positive=right)

        Returns:
            output: control signal in range [-1, 1]
        """
        now = time.time()
        dt  = now - self.last_time
        self.last_time = now

        # Clamp dt to avoid huge jumps after pauses
        dt = max(0.001, min(dt, 0.5))

        # Proportional term
        P = self.Kp * error

        # Integral term — accumulate error over time
        self.integral += error * dt
        # Anti-windup: clamp integral
        self.integral = max(-self.integral_limit,
                            min(self.integral_limit,
                                self.integral))
        I = self.Ki * self.integral

        # Derivative term — rate of change
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative

        self.prev_error = error

        # Sum all three terms
        output = P + I + D

        # Clamp output to [-1, 1]
        return max(-1.0, min(1.0, output))

    def reset(self):
        """Reset controller state. Call when switching modes."""
        self.prev_error = 0.0
        self.integral   = 0.0
        self.last_time  = time.time()


class AutonomousController:
    """
    Rule-based autonomous driving controller.

    Reads fused scene state (detections + lane + telemetry)
    and outputs vehicle control signals:
        - steering:  [-1, 1] (left to right)
        - throttle:  [0, 1]
        - brake:     [0, 1]

    Three driving modes:
        MANUAL:     Human controls the car
        ASSISTED:   Human steers, AI assists with braking
        AUTONOMOUS: AI controls everything
    """

    # Drive modes
    MODE_MANUAL     = 0
    MODE_ASSISTED   = 1
    MODE_AUTONOMOUS = 2

    def __init__(self, send_socket, unity_ip, unity_port):
        self.send_socket = send_socket
        self.unity_ip    = unity_ip
        self.unity_port  = unity_port

        # Current drive mode
        self.mode = self.MODE_MANUAL

        # PID controller for lane keeping
        self.lane_pid = PIDController(
            Kp=0.8,   # Strong proportional response
            Ki=0.01,  # Small integral to remove drift
            Kd=0.15   # Moderate derivative damping
        )

        # Adaptive cruise control settings
        self.acc_target_distance = 15.0  # metres
        self.acc_min_distance    = 5.0   # emergency brake
        self.acc_max_speed       = 15.0  # m/s (~54 km/h)
        self.acc_comfort_decel   = 3.0   # m/s² comfortable
        self.acc_max_throttle    = 0.6   # max throttle

        # Control state
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.current_brake    = 0.0

        # Statistics
        self.frames_processed = 0
        self.emergency_brakes = 0

        print("[AutonomousController] Initialised")
        print(f"  Lane PID: Kp={self.lane_pid.Kp}, "
              f"Ki={self.lane_pid.Ki}, "
              f"Kd={self.lane_pid.Kd}")
        print(f"  ACC: target={self.acc_target_distance}m, "
              f"emergency={self.acc_min_distance}m")

    def set_mode(self, mode):
        """Switch drive mode. Resets PID on mode change."""
        if mode != self.mode:
            self.mode = mode
            self.lane_pid.reset()
            mode_names = {
                0: "MANUAL",
                1: "ASSISTED",
                2: "AUTONOMOUS"
            }
            print(f"[AutonomousController] Mode → "
                  f"{mode_names.get(mode, '?')}")

    def compute(self, scene_state, lane_offset,
                lane_detected):
        """
        Compute control signals from scene state.

        Args:
            scene_state: dict from KalmanFusion.update()
            lane_offset: float from LaneDetector
            lane_detected: bool from LaneDetector

        Returns:
            dict with steering, throttle, brake, mode
        """
        self.frames_processed += 1

        if self.mode == self.MODE_MANUAL:
            return self._manual_output()

        # ── STEERING (lane keeping PID) ───────────────────
        steering = self._compute_steering(
            lane_offset, lane_detected)

        # ── THROTTLE (adaptive cruise control) ───────────
        throttle = self._compute_throttle(scene_state)

        # ── BRAKE (collision avoidance) ───────────────────
        brake = self._compute_brake(scene_state)

        # Emergency brake overrides everything
        if scene_state.get("emergency_brake", False):
            brake    = 1.0
            throttle = 0.0
            self.emergency_brakes += 1

        # In ASSISTED mode only AI brakes, human steers
        if self.mode == self.MODE_ASSISTED:
            steering = 0.0  # Don't override human steering

        # Clamp all outputs
        self.current_steering = max(-1.0, min(1.0, steering))
        self.current_throttle = max(0.0,  min(1.0, throttle))
        self.current_brake    = max(0.0,  min(1.0, brake))

        return {
            "steering": round(self.current_steering, 4),
            "throttle": round(self.current_throttle, 4),
            "brake":    round(self.current_brake, 4),
            "mode":     self.mode,
            "emergency_brake": bool(
                scene_state.get("emergency_brake", False)),
        }

    def _manual_output(self):
        """Return zero control — human is driving."""
        return {
            "steering": 0.0,
            "throttle": 0.0,
            "brake":    0.0,
            "mode":     self.MODE_MANUAL,
            "emergency_brake": False,
        }

    def _compute_steering(self, lane_offset, lane_detected):
        """
        Lane-keeping PID controller.

        The lane_offset from LaneDetector is:
            negative = car is left of centre (steer right)
            positive = car is right of centre (steer left)

        We negate it because:
            steer right = positive steering in RCC
            steer left  = negative steering in RCC
        """
        if not lane_detected:
            # No lane detected — gentle return to centre
            # using last known error, decayed
            self.lane_pid.integral *= 0.95
            return 0.0

        # PID input is the error (how far from centre)
        # Negate because offset and steering have opposite signs
        error = -lane_offset

        steering = self.lane_pid.compute(error)
        return steering

    def _compute_throttle(self, scene_state):
        """
        Adaptive Cruise Control (ACC).

        Target: maintain acc_target_distance from vehicle ahead.
        Speed: limit to acc_max_speed.

        Logic:
        - If no vehicle ahead: maintain target speed
        - If vehicle ahead > target distance: maintain speed
        - If vehicle ahead < target distance: reduce throttle
        - If vehicle ahead < emergency distance: full brake
        """
        closest = scene_state.get(
            "closest_vehicle_distance", 999.0)
        ego_speed = scene_state.get("ego_speed", 0.0)

        # Speed limit — don't exceed max speed
        if ego_speed >= self.acc_max_speed:
            return 0.0

        # Speed factor: ramp up throttle as speed increases
        # (prevents too aggressive acceleration from stop)
        speed_ratio = ego_speed / self.acc_max_speed
        base_throttle = self.acc_max_throttle * (
            1.0 - speed_ratio * 0.3)

        # Distance factor
        if closest >= self.acc_target_distance:
            # Vehicle far away or no vehicle — full throttle
            return base_throttle

        elif closest >= self.acc_min_distance:
            # Vehicle in ACC range — proportional reduction
            # At target_distance: full throttle
            # At min_distance: zero throttle
            distance_ratio = (
                (closest - self.acc_min_distance) /
                (self.acc_target_distance - self.acc_min_distance)
            )
            return base_throttle * distance_ratio

        else:
            # Too close — no throttle, brake takes over
            return 0.0

    def _compute_brake(self, scene_state):
        """
        Collision avoidance braking.

        Smooth braking profile:
        - Beyond brake_start_distance: no braking
        - Between brake_start and emergency: proportional brake
        - Within emergency distance: full brake
        """
        closest = scene_state.get(
            "closest_vehicle_distance", 999.0)

        brake_start = self.acc_target_distance * 0.6  # ~15m
        brake_full  = self.acc_min_distance           # 8m

        if closest >= brake_start:
            return 0.0
        elif closest >= brake_full:
            # Proportional braking
            ratio = 1.0 - (
                (closest - brake_full) /
                (brake_start - brake_full))
            return ratio * 0.8  # Max 80% braking (smooth)
        else:
            return 1.0  # Full emergency brake

    def send_control(self, control_signals):
        """
        Send control signals to Unity over UDP.
        Unity PerceptionBridge receives these on port 5007.
        """
        try:
            data = json.dumps(control_signals).encode('utf-8')
            self.send_socket.sendto(
                data, (self.unity_ip, self.unity_port))
        except Exception as e:
            print(f"[AutonomousController] Send error: {e}")

    def get_stats(self):
        """Return controller statistics."""
        return {
            "frames_processed": self.frames_processed,
            "emergency_brakes": self.emergency_brakes,
            "current_mode":     self.mode,
        }