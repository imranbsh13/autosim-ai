import cv2
import numpy as np


class BEVProjection:
    """
    Bird's Eye View (BEV) projection using homography.

    Transforms forward-facing camera view to top-down view.
    Projects detected objects into BEV space for a minimap.

    Homography:
        A 3x3 matrix that maps points from one plane
        (perspective camera view) to another plane
        (top-down bird's eye view).

        Computed from 4 known point correspondences using
        cv2.getPerspectiveTransform().
        Applied using cv2.warpPerspective().
    """

    def __init__(self, image_width=640, image_height=360,
                 bev_width=300, bev_height=400):
        self.img_w = image_width
        self.img_h = image_height
        self.bev_w = bev_width
        self.bev_h = bev_height

        # ── HOMOGRAPHY MATRIX ─────────────────────────────
        # Define 4 source points in camera perspective view
        # These are road points visible in our Unity scene
        # Tuned for our camera FOV and road width

        # Source points (in camera image, normalised)
        # These define the trapezoid of visible road
        src = np.float32([
            [0.40 * image_width,  0.60 * image_height],  # top-left
            [0.60 * image_width,  0.60 * image_height],  # top-right
            [0.90 * image_width,  0.95 * image_height],  # bottom-right
            [0.10 * image_width,  0.95 * image_height],  # bottom-left
        ])

        # Destination points (in BEV output image)
        # Rectangle covering full BEV canvas
        dst = np.float32([
            [0.25 * bev_width,  0.0 * bev_height],  # top-left
            [0.75 * bev_width,  0.0 * bev_height],  # top-right
            [0.75 * bev_width,  1.0 * bev_height],  # bottom-right
            [0.25 * bev_width,  1.0 * bev_height],  # bottom-left
        ])

        # Compute homography matrix
        self.H = cv2.getPerspectiveTransform(src, dst)
        # Inverse homography (BEV → perspective)
        self.H_inv = cv2.getPerspectiveTransform(dst, src)

        # Colours for BEV objects
        self.colors = {
            0: (0, 255, 0),    # vehicle — green
            1: (0, 0, 255),    # pedestrian — red
            2: (0, 255, 255),  # traffic_sign — yellow
            3: (255, 0, 0),    # traffic_light — blue
        }

        print("[BEVProjection] Initialised")
        print(f"  BEV canvas: {bev_width}x{bev_height}px")

    def warp_frame(self, frame):
        """
        Apply perspective warp to get BEV of road surface.

        Args:
            frame: BGR camera image

        Returns:
            bev_frame: warped top-down view
        """
        return cv2.warpPerspective(
            frame, self.H,
            (self.bev_w, self.bev_h))

    def project_point(self, x_norm, y_norm):
        """
        Project a normalised image point to BEV coordinates.

        Args:
            x_norm: normalised x (0-1) in camera image
            y_norm: normalised y (0-1) in camera image

        Returns:
            (bev_x, bev_y) in BEV pixel coordinates
            or None if projection fails
        """
        # Convert to pixel coordinates
        pt = np.float32([[[
            x_norm * self.img_w,
            y_norm * self.img_h
        ]]])

        # Apply homography
        bev_pt = cv2.perspectiveTransform(pt, self.H)

        bev_x = int(bev_pt[0][0][0])
        bev_y = int(bev_pt[0][0][1])

        # Check if within BEV canvas
        if (0 <= bev_x < self.bev_w and
                0 <= bev_y < self.bev_h):
            return (bev_x, bev_y)

        return None

    def create_minimap(self, detections, tracks,
                       frame=None):
        """
        Create a BEV minimap showing detected and
        tracked objects in top-down space.

        Args:
            detections: raw YOLOv8 detections
            tracks:     Kalman-filtered confirmed tracks
            frame:      optional camera frame for BEV warp

        Returns:
            minimap: BGR minimap image (bev_w x bev_h)
        """
        # Start with dark background or warped road
        if frame is not None:
            minimap = self.warp_frame(frame)
            # Darken for overlay visibility
            minimap = (minimap * 0.4).astype(np.uint8)
        else:
            minimap = np.zeros(
                (self.bev_h, self.bev_w, 3),
                dtype=np.uint8)

        # Draw road centre line
        centre_x = self.bev_w // 2
        cv2.line(minimap,
                 (centre_x, 0),
                 (centre_x, self.bev_h),
                 (40, 40, 40), 1)

        # Draw ego vehicle (your car) at bottom centre
        ego_x = self.bev_w // 2
        ego_y = self.bev_h - 20
        cv2.rectangle(minimap,
                      (ego_x - 8, ego_y - 14),
                      (ego_x + 8, ego_y + 14),
                      (255, 255, 255), -1)
        cv2.putText(minimap, "EGO",
                    (ego_x - 12, ego_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1)

        # Draw distance rings (10m, 20m, 30m)
        for dist_m, label in [(10, "10m"),
                               (20, "20m"),
                               (30, "30m")]:
            # Approximate pixel distance
            # Tuned for our camera setup
            pixel_dist = int(dist_m * 5)
            y_pos = ego_y - pixel_dist
            if 0 <= y_pos < self.bev_h:
                cv2.line(minimap,
                         (0, y_pos),
                         (self.bev_w, y_pos),
                         (30, 30, 30), 1)
                cv2.putText(minimap, label,
                            (2, y_pos - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (60, 60, 60), 1)

        # Draw Kalman-tracked objects (confirmed tracks)
        for track in tracks:
            # Project detection bbox centre to BEV
            bev_pt = self.project_point(
                track["cx"], track["cy"])

            if bev_pt is None:
                continue

            bx, by = bev_pt
            class_id = track["class_id"]
            color = self.colors.get(class_id, (255, 255, 255))

            # Draw object dot
            radius = 8 if class_id == 0 else 5
            cv2.circle(minimap, (bx, by), radius,
                       color, -1)

            # Draw track ID and distance
            label = (f"#{track['track_id']} "
                     f"{track['distance_m']:.0f}m")
            cv2.putText(minimap, label,
                        (bx + 6, by),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, color, 1)

            # Draw velocity vector
            if track["velocity"] > 0.01:
                vx = int(track["vx"] * 50)
                vy = int(track["vy"] * 50)
                cv2.arrowedLine(minimap,
                                (bx, by),
                                (bx + vx, by + vy),
                                color, 1,
                                tipLength=0.3)

        # Border
        cv2.rectangle(minimap, (0, 0),
                      (self.bev_w - 1, self.bev_h - 1),
                      (80, 80, 80), 1)

        # Title
        cv2.putText(minimap, "BEV",
                    (self.bev_w - 30, 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1)

        return minimap