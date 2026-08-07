import cv2
import numpy as np


class StereoDepth:
    """
    Stereo vision depth estimation using OpenCV StereoBM.

    Takes left (RGB) and right (Stereo) camera frames,
    computes a disparity map, converts to metric depth.

    Stereo geometry:
        depth = (focal_length_px × baseline_m) / disparity_px

    Camera setup:
        baseline = 0.06m (6cm — matches Unity StereoCamera offset)
        FOV = 60 degrees
        Resolution = 640 × 360
        focal_length = width / (2 × tan(FOV/2))
                     = 640 / (2 × tan(30°))
                     = 640 / 1.1547
                     ≈ 554 pixels
    """

    def __init__(self, baseline=0.06, fov_degrees=60,
                 image_width=640):
        self.baseline = baseline  # metres

        # Compute focal length from FOV and image width
        # This matches the Unity camera configuration exactly
        fov_rad = np.radians(fov_degrees)
        self.focal_length = image_width / (
            2 * np.tan(fov_rad / 2))

        print(f"[StereoDepth] Initialised")
        print(f"  Baseline: {baseline*100:.0f}cm")
        print(f"  Focal length: {self.focal_length:.1f}px")

        # ── STEREO BM PARAMETERS ─────────────────────────
        # StereoBM = Block Matching stereo algorithm
        # Finds matching blocks between left and right images
        # to compute disparity at each pixel

        # numDisparities: range of disparity values to search
        # Must be divisible by 16
        # Higher = handles closer objects but slower
        num_disparities = 64

        # blockSize: size of matching block (must be odd)
        # Larger = smoother but less detail
        block_size = 15

        self.stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )

        # Fine-tune for our Unity scene
        self.stereo.setPreFilterCap(31)
        self.stereo.setMinDisparity(0)
        self.stereo.setSpeckleRange(32)
        self.stereo.setSpeckleWindowSize(100)
        self.stereo.setUniquenessRatio(10)

        # Depth range for visualisation
        self.min_depth = 1.0   # metres
        self.max_depth = 50.0  # metres

    def compute(self, left_frame, right_frame):
        """
        Compute disparity and depth map from stereo pair.

        Args:
            left_frame:  BGR image from RGBCamera
            right_frame: BGR image from StereoCamera

        Returns:
            depth_map:     float32 array of depth in metres
            depth_vis:     BGR colour visualisation
            valid_mask:    boolean mask of valid depth pixels
        """
        # ── STEP 1: CONVERT TO GREYSCALE ─────────────────
        # StereoBM works on greyscale images
        left_grey  = cv2.cvtColor(
            left_frame, cv2.COLOR_BGR2GRAY)
        right_grey = cv2.cvtColor(
            right_frame, cv2.COLOR_BGR2GRAY)

        # ── STEP 2: COMPUTE DISPARITY ─────────────────────
        # Returns disparity map scaled by 16 (fixed point)
        # Divide by 16 to get actual disparity in pixels
        disparity_raw = self.stereo.compute(
            left_grey, right_grey)
        disparity = disparity_raw.astype(np.float32) / 16.0

        # ── STEP 3: FILTER INVALID DISPARITIES ───────────
        # StereoBM returns -1 for pixels where matching failed
        valid_mask = disparity > 0

        # ── STEP 4: CONVERT DISPARITY TO DEPTH ───────────
        # depth = (focal_length × baseline) / disparity
        # Where disparity and focal_length are in pixels,
        # baseline is in metres → depth is in metres
        depth_map = np.zeros_like(disparity)
        depth_map[valid_mask] = (
            self.focal_length * self.baseline /
            disparity[valid_mask])

        # Clamp to realistic range
        depth_map = np.clip(
            depth_map, self.min_depth, self.max_depth)
        depth_map[~valid_mask] = 0

        # ── STEP 5: COLOUR VISUALISATION ─────────────────
        # Normalise depth to 0-255 for display
        # Close = red, Far = blue (COLORMAP_JET)
        depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
        valid_depth = depth_map[valid_mask]

        if len(valid_depth) > 0:
            # Invert so close objects are bright
            depth_inverted = self.max_depth - depth_map
            depth_norm = cv2.normalize(
                depth_inverted, None,
                0, 255, cv2.NORM_MINMAX,
                dtype=cv2.CV_8U)

        depth_vis = cv2.applyColorMap(
            depth_norm, cv2.COLORMAP_JET)

        # Mask invalid pixels to black
        depth_vis[~valid_mask] = 0

        return depth_map, depth_vis, valid_mask

    def get_point_depth(self, depth_map, x, y,
                        window=5):
        """
        Get depth at a specific pixel location.
        Uses a small window average for robustness.

        Args:
            depth_map: float32 depth array
            x, y: pixel coordinates
            window: averaging window size

        Returns:
            depth in metres, or None if invalid
        """
        h, w = depth_map.shape
        x1 = max(0, x - window // 2)
        x2 = min(w, x + window // 2)
        y1 = max(0, y - window // 2)
        y2 = min(h, y + window // 2)

        region = depth_map[y1:y2, x1:x2]
        valid  = region[region > 0]

        if len(valid) == 0:
            return None

        return float(np.median(valid))

    def get_detection_depths(self, depth_map, detections):
        """
        Get depth for each YOLOv8 detection using
        the stereo depth map rather than bbox height estimation.

        This replaces the empirical distance formula in
        kalman_fusion.py with geometrically accurate depth.

        Args:
            depth_map:  float32 depth array (H x W)
            detections: list of detection dicts from YOLOv8

        Returns:
            list of (detection, depth_metres) tuples
        """
        h, w = depth_map.shape
        results = []

        for det in detections:
            bbox = det["bbox"]

            # Convert normalised bbox centre to pixel coords
            cx_px = int(bbox["cx"] * w)
            cy_px = int(bbox["cy"] * h)

            depth = self.get_point_depth(
                depth_map, cx_px, cy_px, window=10)

            results.append((det, depth))

        return results