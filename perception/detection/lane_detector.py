import cv2
import numpy as np
import math


class LaneDetector:
    """
    Classical OpenCV lane detection pipeline.
    Detects left and right lane boundaries and computes
    lateral offset from lane centre.

    Pipeline:
    1. HSV colour masking (white + yellow lanes)
    2. Canny edge detection
    3. Region of interest crop
    4. Hough line transform
    5. Line averaging and lane centre calculation
    """

    def __init__(self):
        # HSV colour ranges for lane markings
        # White lanes
        self.white_lower = np.array([0,   0,   180])
        self.white_upper = np.array([180, 30,  255])

        # Yellow lanes
        self.yellow_lower = np.array([15,  80,  80])
        self.yellow_upper = np.array([35, 255, 255])

        # Canny thresholds
        self.canny_low  = 50
        self.canny_high = 150

        # Hough line parameters
        self.hough_rho         = 1
        self.hough_theta       = np.pi / 180
        self.hough_threshold   = 20
        self.hough_min_length  = 20
        self.hough_max_gap     = 10

        # Lane state
        self.left_lane  = None  # (slope, intercept)
        self.right_lane = None
        self.lane_centre_offset = 0.0  # -1.0 (left) to +1.0 (right)
        self.lane_detected = False

    def detect(self, frame):
        """
        Run full lane detection pipeline on a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            result_frame: frame with lane overlay drawn
            offset: lateral offset from lane centre (-1 to +1)
            detected: whether lanes were successfully detected
        """
        h, w = frame.shape[:2]

        # ── STEP 1: HSV COLOUR MASK ───────────────────────
        # Convert BGR to HSV
        # HSV separates Hue (colour), Saturation (intensity),
        # Value (brightness). Much more robust than RGB for
        # detecting specific colours under varying lighting.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for white and yellow lane markings
        white_mask  = cv2.inRange(hsv, self.white_lower,
                                       self.white_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower,
                                       self.yellow_upper)

        # Combine both masks
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Apply mask to original frame
        masked = cv2.bitwise_and(frame, frame, mask=lane_mask)

        # ── STEP 2: GREYSCALE + BLUR ──────────────────────
        grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # Gaussian blur reduces noise before edge detection.
        # Kernel size (5,5) — larger = more smoothing.
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)

        # ── STEP 3: CANNY EDGE DETECTION ─────────────────
        # Canny finds pixels where intensity changes sharply.
        # low threshold = 50: weak edges included if connected
        # high threshold = 150: strong edges always included
        # The ratio 1:3 (50:150) is standard for lane detection.
        edges = cv2.Canny(blurred,
                          self.canny_low,
                          self.canny_high)

        # ── STEP 4: REGION OF INTEREST ───────────────────
        # Mask out everything except the road area.
        # A trapezoid that covers the lower portion of the frame
        # where lane markings appear.
        roi = self._apply_roi(edges, w, h)

        # ── STEP 5: HOUGH LINE TRANSFORM ─────────────────
        # Finds straight lines in the edge image.
        # Works in polar coordinates (rho, theta) to find
        # lines of any angle, not just horizontal/vertical.
        lines = cv2.HoughLinesP(
            roi,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap
        )

        # ── STEP 6: SEPARATE AND AVERAGE LINES ───────────
        left_lines, right_lines = self._classify_lines(lines, w, h)

        self.left_lane  = self._average_line(left_lines)
        self.right_lane = self._average_line(right_lines)

        # ── STEP 7: CALCULATE LANE CENTRE OFFSET ─────────
        self.lane_centre_offset, self.lane_detected = \
            self._calculate_offset(w, h)

        # ── STEP 8: DRAW OVERLAY ──────────────────────────
        result = self._draw_lanes(frame.copy(), w, h)

        return result, self.lane_centre_offset, self.lane_detected

    def _apply_roi(self, edges, w, h):
        """
        Apply a trapezoidal region of interest mask.
        Only keep the road area — discard sky and buildings.
        """
        # Define trapezoid vertices
        # Bottom-left, top-left, top-right, bottom-right
        roi_vertices = np.array([[
            (int(w * 0.05), h),              # bottom-left
            (int(w * 0.40), int(h * 0.58)),  # top-left
            (int(w * 0.60), int(h * 0.58)),  # top-right
            (int(w * 0.95), h),              # bottom-right
        ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        return cv2.bitwise_and(edges, mask)

    def _classify_lines(self, lines, w, h):
        """
        Classify Hough lines into left and right lane lines
        based on their slope.

        Positive slope = right lane (goes up-right)
        Negative slope = left lane (goes up-left)

        Lines with near-zero slope are horizontal noise —
        discard them.
        """
        left_lines  = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Skip vertical lines (division by zero)
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Discard near-horizontal lines (slope < 0.3)
            # These are usually road markings, not lane edges
            if abs(slope) < 0.3:
                continue

            # Calculate intercept (y = mx + b, so b = y - mx)
            intercept = y1 - slope * x1

            if slope < 0:
                # Negative slope = left lane
                left_lines.append((slope, intercept))
            else:
                # Positive slope = right lane
                right_lines.append((slope, intercept))

        return left_lines, right_lines

    def _average_line(self, lines):
        """
        Average a list of (slope, intercept) pairs
        into a single representative line.
        Returns None if no lines provided.
        """
        if not lines:
            return None

        slopes     = [l[0] for l in lines]
        intercepts = [l[1] for l in lines]

        avg_slope     = np.mean(slopes)
        avg_intercept = np.mean(intercepts)

        return (avg_slope, avg_intercept)

    def _line_points(self, line, h):
        """
        Convert a (slope, intercept) line to two pixel points
        for drawing. Points span from bottom of frame to
        middle of frame.
        """
        if line is None:
            return None

        slope, intercept = line

        # Avoid division by very small slopes
        if abs(slope) < 0.001:
            return None

        # y1 = bottom of frame, y2 = middle of frame
        y1 = h
        y2 = int(h * 0.58)

        # x = (y - intercept) / slope
        try:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
        except (ZeroDivisionError, OverflowError, ValueError):
            return None

        return (x1, y1, x2, y2)

    def _calculate_offset(self, w, h):
        """
        Calculate lateral offset from lane centre.

        If both lanes detected:
            Find where each lane crosses the bottom of frame,
            compute the midpoint, compare to image centre.

        Returns:
            offset: -1.0 (far left) to +1.0 (far right)
            detected: True if at least one lane found
        """
        if self.left_lane is None and self.right_lane is None:
            return 0.0, False

        image_centre = w / 2.0

        # Get bottom x coordinates of each lane
        left_x  = None
        right_x = None

        if self.left_lane is not None:
            slope, intercept = self.left_lane
            if abs(slope) > 0.001:
                left_x = (h - intercept) / slope

        if self.right_lane is not None:
            slope, intercept = self.right_lane
            if abs(slope) > 0.001:
                right_x = (h - intercept) / slope

        # Calculate lane centre
        if left_x is not None and right_x is not None:
            lane_centre = (left_x + right_x) / 2.0
        elif left_x is not None:
            # Only left lane — estimate right from left
            lane_centre = left_x + (w * 0.25)
        elif right_x is not None:
            # Only right lane — estimate left from right
            lane_centre = right_x - (w * 0.25)
        else:
            return 0.0, False

        # Offset: positive = car is right of centre (steer left)
        # Normalised to [-1, 1]
        offset = (lane_centre - image_centre) / (w / 2.0)
        offset = max(-1.0, min(1.0, offset))

        return offset, True

    def _draw_lanes(self, frame, w, h):
        """
        Draw detected lane lines and centre offset on the frame.
        """
        overlay = frame.copy()

        # Draw left lane line
        left_pts = self._line_points(self.left_lane, h)
        if left_pts is not None:
            x1, y1, x2, y2 = left_pts
            cv2.line(overlay, (x1, y1), (x2, y2),
                     (255, 0, 0), 4)  # Blue

        # Draw right lane line
        right_pts = self._line_points(self.right_lane, h)
        if right_pts is not None:
            x1, y1, x2, y2 = right_pts
            cv2.line(overlay, (x1, y1), (x2, y2),
                     (0, 0, 255), 4)  # Red

        # Draw lane fill if both lanes detected
        if left_pts is not None and right_pts is not None:
            lx1, ly1, lx2, ly2 = left_pts
            rx1, ry1, rx2, ry2 = right_pts

            pts = np.array([
                [lx1, ly1], [lx2, ly2],
                [rx2, ry2], [rx1, ry1]
            ], dtype=np.int32)

            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        # Draw lane centre indicator
        centre_x = int(w / 2 + self.lane_centre_offset * w / 2)
        cv2.line(frame,
                 (centre_x, h - 20),
                 (centre_x, h - 60),
                 (0, 255, 255), 3)  # Yellow

        # Draw image centre reference
        cv2.line(frame,
                 (w // 2, h - 20),
                 (w // 2, h - 60),
                 (255, 255, 255), 1)  # White

        # Draw offset text
        status = "DETECTED" if self.lane_detected else "NOT DETECTED"
        color  = (0, 255, 0) if self.lane_detected else (0, 0, 255)
        cv2.putText(frame,
            f"Lane: {status} | Offset: {self.lane_centre_offset:.3f}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw left/right lane lines on top
        if left_pts is not None:
            x1, y1, x2, y2 = left_pts
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
        if right_pts is not None:
            x1, y1, x2, y2 = right_pts
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

        return frame

    def get_offset(self):
        """Return the latest lane centre offset."""
        return self.lane_centre_offset

    def is_detected(self):
        """Return True if lanes were detected in last frame."""
        return self.lane_detected