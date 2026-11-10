"""
detector.py
-----------
Object detection (YOLOv8n) + video processing.
Uses tracker.py to give every object a unique ID.
"""

import os
import subprocess
import time
from pathlib import Path

import cv2

from tracker import CentroidTracker

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_FILE = BASE_DIR / "yolov8n.pt"

# A few BGR colours (OpenCV uses Blue-Green-Red order)
COLORS = [(255, 99, 71), (60, 179, 113), (30, 144, 255), (255, 165, 0),
          (186, 85, 211), (0, 191, 255), (220, 20, 60), (34, 139, 34)]


def load_model():
    """Load the pre-trained YOLOv8n model (downloaded automatically on first run)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("The 'ultralytics' package is not installed. "
                           "Run: pip install -r requirements.txt")

    # Use the local file if it exists, otherwise let ultralytics download it
    model_name = str(MODEL_FILE) if MODEL_FILE.exists() else "yolov8n.pt"
    try:
        return YOLO(model_name)
    except Exception as error:
        raise RuntimeError(
            "Could not load the YOLOv8n model. Check your internet connection "
            "(needed once to download 'yolov8n.pt') or place yolov8n.pt in the "
            f"project folder. Details: {error}"
        )


def detect_objects(frame, model, confidence=0.4):
    """Run YOLO on one frame and return a list of detections."""
    results = model.predict(frame, conf=confidence, verbose=False)[0]
    names = results.names                      # class id -> class name

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_name": names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": (int(x1), int(y1), int(x2), int(y2)),
        })
    return detections


def draw_boxes(frame, detections, show_boxes=True):
    """Draw boxes with the label 'ID name confidence' on the frame."""
    if not show_boxes:
        return frame

    scale = max(0.5, frame.shape[1] / 1500)    # bigger text for bigger videos
    thickness = max(1, int(scale * 2))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = COLORS[sum(ord(c) for c in det["class_name"]) % len(COLORS)]
        label = f"ID {det.get('id', '?')} {det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # filled background so the text is easy to read
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        top = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + text_w + 4, top + text_h + baseline + 4),
                      color, -1)
        cv2.putText(frame, label, (x1 + 2, top + text_h + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
    return frame


def save_output_video(raw_path, final_path):
    """
    OpenCV writes 'mp4v' videos that most browsers cannot play.
    If ffmpeg (from imageio-ffmpeg) is available, convert the video to H.264
    so it plays inside Streamlit. Returns (final_path, browser_ready).
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [ffmpeg, "-y", "-i", str(raw_path),
                   "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                   "-movflags", "+faststart", str(final_path)]
        subprocess.run(command, check=True, capture_output=True)
        os.remove(raw_path)
        return final_path, True
    except Exception:
        # conversion not possible: keep the OpenCV file (still downloadable)
        os.replace(raw_path, final_path)
        return final_path, False


def process_video(video_path, model, confidence=0.4, show_boxes=True,
                  progress_callback=None):
    """
    Read the video, detect + track objects frame by frame, write the output video.
    Returns a dictionary with the output path and the counting results.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("The video file could not be opened. It may be invalid.")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("No frames could be read. The video may be corrupted.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)        # go back to the first frame

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25
    height, width = first_frame.shape[:2]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = int(time.time())
    raw_path = OUTPUT_DIR / f"raw_{stamp}.mp4"
    final_path = OUTPUT_DIR / f"processed_{stamp}.mp4"

    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output video file.")

    # an object may move up to ~10% of the frame width between two frames
    tracker = CentroidTracker(max_disappeared=15, max_distance=max(60, width * 0.1))
    frames_done = 0
    total_detections = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detect_objects(frame, model, confidence)
            detections = tracker.track_objects(detections)
            frame = draw_boxes(frame, detections, show_boxes)
            writer.write(frame)

            frames_done += 1
            total_detections += len(detections)
            if progress_callback and total_frames > 0:
                progress_callback(min(frames_done / total_frames, 1.0))
    finally:
        cap.release()
        writer.release()

    output_path, browser_ready = save_output_video(raw_path, final_path)
    class_counts = tracker.get_class_counts()

    return {
        "output_path": str(output_path),
        "browser_ready": browser_ready,
        "frames": frames_done,
        "total_detections": total_detections,
        "total_objects": sum(class_counts.values()),
        "class_counts": class_counts,
    }
