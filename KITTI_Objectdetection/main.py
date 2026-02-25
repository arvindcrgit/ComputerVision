import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_PATH = r"D:\Arvind Masters\Git\ComputerVision\KITTI_Objectdetection\KITTI_Selection"
IMAGE_DIR = os.path.join(BASE_PATH, "images")
LABEL_DIR = os.path.join(BASE_PATH, "labels")

# Note: Ensure this path points to where your model file actually is
MODEL_PATH = os.path.join(BASE_PATH, "yolo11x.pt")
# If the model is in the current directory, you can just use "yolo11x.pt"

H_CAMERA = 1.65
K = np.array([[721.5377, 0.0, 609.5593],
              [0.0, 721.5377, 172.8540],
              [0.0, 0.0, 1.0]])

K_INV = np.linalg.inv(K)


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if (areaA + areaB - inter) > 0:
        return inter / float(areaA + areaB - inter)
    else:
        return 0


def estimate_distance(box):
    # u = midpoint of width, v = bottom edge (y_max)
    # Scale to ground plane and return Euclidean norm
    ray = K_INV @ np.array([(box[0] + box[2]) / 2, box[3], 1.0])
    return np.linalg.norm(ray * (H_CAMERA / ray[1]))


def main():
    # Load model (make sure path is correct or just use string name if in current dir)
    # Using 'yolo11x.pt' directly if path issues occur is often safer
    model = YOLO("yolo11x.pt")

    img_list = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])

    # Initialize lists for global distance tracking
    all_est, all_gt = [], []

    print(f"{'Scene':<15} {'Prec':<6} {'Rec':<6} {'Avg IoU':<8}")

    for img_name in img_list:
        img_path = os.path.join(IMAGE_DIR, img_name)
        # Assuming label files have same name but .txt extension
        lab_path = os.path.join(LABEL_DIR, img_name.replace('.png', '.txt'))

        # Inference
        results = model(img_path, verbose=False)[0]
        # Filter for class 2 (Car) if using standard COCO model
        # Note: box.cls is a tensor, we convert to numpy
        preds = results.boxes.xyxy.cpu().numpy()[results.boxes.cls.cpu().numpy() == 2]

        # Ground Truth Loading
        gt_boxes, gt_dists = [], []
        if os.path.exists(lab_path):
            with open(lab_path, 'r') as f:
                for line in f:
                    d = line.strip().split()
                    # Parse based on expected format.
                    # Warning: Make sure indices match your label format!
                    # Usually KITTI labels are: type ... bbox(4) ... distance
                    # Your code used d[1]-d[4] for box and d[5] for distance.
                    # Standard KITTI often has box at indices 4,5,6,7 and dist at 13.
                    # Adjust indices below if your custom labels differ.
                    gt_boxes.append([float(d[1]), float(d[2]), float(d[3]), float(d[4])])
                    gt_dists.append(float(d[5]))

        canvas = cv2.imread(img_path)
        tp = 0
        current_ious = []
        matched_gt = set()

        # Match Predictions to GT
        for p in preds:
            dist_c = estimate_distance(p)
            best_iou, best_idx = 0, -1

            for i, g in enumerate(gt_boxes):
                if i not in matched_gt:
                    iou = compute_iou(p, g)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

            current_ious.append(best_iou)

            if best_iou >= 0.5:
                tp += 1
                matched_gt.add(best_idx)
                all_est.append(dist_c)
                all_gt.append(gt_dists[best_idx])

            # Draw YOLO (Red)
            x1, y1, x2, y2 = map(int, p)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(canvas, f"{dist_c:.1f}m", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw GT (Green)
        for i, g in enumerate(gt_boxes):
            gx1, gy1, gx2, gy2 = map(int, g)
            cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(canvas, f"GT:{gt_dists[i]:.1f}m", (gx1, gy1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        prec = tp / len(preds) if len(preds) > 0 else 0
        rec = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0

        avg_iou = np.mean(current_ious) if current_ious else 0
        print(f"{img_name:<15} {prec:<6.2f} {rec:<6.2f} {avg_iou:<8.2f}")

        cv2.imshow("Distance Estimated", canvas)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # --- Plotting ---
    if all_est:
        plt.figure(figsize=(8, 8))
        plt.scatter(all_gt, all_est, alpha=0.6, color='tab:blue', label='Detections')

        # Reference Line
        limit = max(max(all_est), max(all_gt)) if all_gt else max(all_est)
        plt.plot([0, limit], [0, limit], 'r--', label='Perfect (d_c = GT)')

        plt.title("Distance Estimation Analysis")
        plt.xlabel("Ground Truth Distance (m)")
        plt.ylabel("YOLO Estimated Distance (m)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()