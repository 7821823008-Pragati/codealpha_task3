# Object Detection and Tracking System Using Python

Detects and tracks objects in uploaded videos using YOLOv8n, OpenCV and a
simple centroid tracking algorithm, with a Streamlit front end.

## Project Structure

```
Object_Detection_Tracking/
├── app.py            # Streamlit interface (pages, settings, results)
├── detector.py       # YOLO model, drawing, video processing, saving output
├── tracker.py        # Centroid tracking (unique IDs)
├── requirements.txt
├── README.md
└── outputs/          # processed videos are saved here
```

## Installation (Windows 11, Python 3.11)

```
cd Object_Detection_Tracking
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run in VS Code

1. Open the `Object_Detection_Tracking` folder in VS Code (File > Open Folder).
2. Select the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > choose `venv`.
3. Open the terminal (Ctrl+`) and run:
   ```
   venv\Scripts\activate
   streamlit run app.py
   ```
4. The app opens at http://localhost:8501. The first detection downloads
   `yolov8n.pt` (about 6 MB), so internet is needed once.

## Sample Testing Procedure

1. Use a short video (10-30 seconds) with people, cars or animals.
2. Open "Object Detection" in the sidebar and upload the video.
3. Keep the confidence at 0.4 and "Show bounding boxes" ticked.
4. Click "Start Detection" and wait for the progress bar.
5. Check: boxes with `ID name confidence` labels, the same object keeping the
   same ID, counts below the videos, and the download button working.
6. Try confidence 0.7 (fewer objects), hiding boxes, and an invalid file
   (rename a .txt file to .mp4) to see the error message.

## How the Files Are Connected

`app.py` calls `load_model()` and `process_video()` from `detector.py`.
`detector.py` uses `CentroidTracker` from `tracker.py` to assign IDs.
Results flow back to `app.py`, which shows and offers the video for download.

## How Tracking Works

1. Find the centre of every detected box.
2. Measure distance to the centres from the previous frame.
3. Closest pairs of the same class keep the same ID.
4. Unmatched detections get a new ID.
5. Objects missing for more than 15 frames are removed.

## Common Errors and Solutions

| Problem | Solution |
|---|---|
| `streamlit` is not recognized | Activate the venv, or use `python -m streamlit run app.py` |
| Model download fails | Check internet, or download `yolov8n.pt` and place it in the project folder |
| "video file could not be opened" | File is corrupted or unsupported; try another MP4 |
| Processed video does not preview | Run `pip install imageio-ffmpeg`; the downloaded file still plays in VLC |
| Processing is slow | Use a shorter or lower-resolution video; CPU speed is the limit |
| PowerShell blocks venv activation | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Same object gets a new ID | Occlusion or fast motion; raise the confidence slightly or use a steadier video |

## Simple Explanation for Evaluation

"I built a web application in Python where a user uploads a video. OpenCV reads
it frame by frame, and a pre-trained YOLOv8n model detects objects with their
names and confidence scores. I wrote a centroid tracker that compares object
centres between frames to keep the same ID for the same object. The app draws
boxes, counts unique objects, shows the processed video and lets the user
download it. It runs on a normal CPU and uses only free tools."

## Limitations

Centroid tracking can swap IDs when objects cross paths or are hidden for a
long time. Stronger trackers (e.g. DeepSORT) can improve this.
