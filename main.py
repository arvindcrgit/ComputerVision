import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import sys

# --- 1. CONFIGURATION ---
# SCALE: 1.0 = Width of poster matches width of marker.
# 2.0 = Poster is 2x wider than the marker.
SCALE = 5.0

# OFFSET: (0,0) is centered on the marker.
# Positive Y moves down, Negative Y moves up.
OFFSET_X = 0.0
OFFSET_Y = 0.0


def select_image_file(prompt):
    """Opens a system window to select an image file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    print(f"Waiting for user to select: {prompt}...")
    file_path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()
    return file_path


def get_perspective_corners(marker_corners, scale, off_x, off_y, aspect_ratio):
    """
    Calculates the 4 corners of the poster on the wall, respecting the
    poster's aspect ratio (height/width).
    """
    # 1. Define the marker's logical coordinate system (0,0 to 1,1)
    # The marker is always a 1x1 square in logical space
    unit_marker = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

    # 2. Get the matrix that maps logical square -> actual wall pixels
    # This defines the "plane" of the wall
    perspective_matrix = cv2.getPerspectiveTransform(unit_marker, marker_corners)

    # 3. Calculate the size of the poster in logical space
    # Width is determined by scale. Height is determined by Aspect Ratio.
    half_width = scale / 2.0
    half_height = (scale * aspect_ratio) / 2.0

    center_x = 0.5 + off_x
    center_y = 0.5 + off_y

    # Define the 4 corners of the poster in logical space relative to the marker center
    poster_logical_corners = np.float32([
        [center_x - half_width, center_y - half_height],  # Top-Left
        [center_x + half_width, center_y - half_height],  # Top-Right
        [center_x + half_width, center_y + half_height],  # Bottom-Right
        [center_x - half_width, center_y + half_height]  # Bottom-Left
    ]).reshape(-1, 1, 2)

    # 4. Project these logical corners onto the actual wall image
    expanded_pixels = cv2.perspectiveTransform(poster_logical_corners, perspective_matrix)
    return expanded_pixels


def resize_for_display(img, max_width=1000):
    """Resizes image for display ONLY if it's too big."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        dim = (int(w * scale), int(h * scale))
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Select Images
    wall_path = select_image_file("Step 1: Select WALL Image")
    if not wall_path: sys.exit()

    poster_path = select_image_file("Step 2: Select POSTER Image")
    if not poster_path: sys.exit()

    # 2. Load Images
    wall_img = cv2.imread(wall_path)
    poster_img = cv2.imread(poster_path)

    if wall_img is None or poster_img is None:
        print("❌ Error: Could not read image files.")
        sys.exit()

    # 3. Detect Marker
    gray = cv2.cvtColor(wall_img, cv2.COLOR_BGR2GRAY)

    # Try different dictionaries if detection fails
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        print(f"✅ Found {len(ids)} marker(s). Using ID: {ids[0][0]}")

        # Get the first marker found
        marker_corners = corners[0][0]

        # 4. Calculate Poster Aspect Ratio (Height / Width)
        poster_h, poster_w = poster_img.shape[:2]
        poster_ar = poster_h / poster_w

        # 5. Calculate where the poster corners should go on the wall
        dst_corners = get_perspective_corners(marker_corners, SCALE, OFFSET_X, OFFSET_Y, poster_ar)

        # 6. Prepare source corners (the whole poster image)
        src_corners = np.float32([[0, 0], [poster_w, 0], [poster_w, poster_h], [0, poster_h]])

        # 7. Warp and Blend
        # Calculate the transformation matrix
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)

        # Warp the poster to fit the wall perspective
        warped_poster = cv2.warpPerspective(poster_img, M, (wall_img.shape[1], wall_img.shape[0]))

        # Create a mask to cut out the area on the wall
        mask = np.zeros_like(wall_img, dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_corners.astype(int), (255, 255, 255))
        mask_inv = cv2.bitwise_not(mask)

        # Black out the area on the wall
        wall_bg = cv2.bitwise_and(wall_img, mask_inv)

        # Combine
        final_result = cv2.add(wall_bg, warped_poster)

        # 8. Show Result
        print("Displaying result... Press any key to close the window.")

        # Resize for display so it fits on screen
        display_img = resize_for_display(final_result, max_width=1200)

        cv2.imshow(f"Augmented Reality (Scale {SCALE})", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optional: Save the full resolution result
        # cv2.imwrite("output_ar.jpg", final_result)

    else:
        print("❌ No ArUco markers found. Try an image with better lighting or a clearer marker.")