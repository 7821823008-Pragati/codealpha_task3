"""
app.py
------
Streamlit front end: Home, Object Detection and About pages.
Run with:  streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from detector import load_model, process_video

st.set_page_config(page_title="Object Detection & Tracking", page_icon="🎯",
                   layout="wide")

# ---------- Simple CSS: white + light blue background, dark blue headings ----------
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #ffffff 0%, #e8f3ff 100%); }
h1, h2, h3, h4 { color: #0b2a5b !important; font-family: 'Segoe UI', sans-serif; }
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stAlert"] {
    color: #000000 !important; font-family: 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] { background-color: #dcecff; }
.card {
    background: #ffffff; color: #000000 !important;
    border: 1px solid #cfe3fa; border-radius: 14px;
    padding: 18px 22px; margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(11, 42, 91, 0.08);
}
.card p, .card li, .card b, .card strong { color: #000000 !important; }
.card h4 { margin-top: 0; }
.stButton > button, .stDownloadButton > button {
    background-color: #1e5aa8; color: white; border: none;
    border-radius: 25px; padding: 0.5rem 1.6rem; font-weight: 600;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #0b2a5b; color: white;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading YOLOv8n model...")
def get_model():
    """Load the model only once and reuse it."""
    return load_model()


def card(title, body_html):
    """Show a white information card."""
    st.markdown(f'<div class="card"><h4>{title}</h4>{body_html}</div>',
                unsafe_allow_html=True)


# ------------------------------- HOME PAGE -------------------------------
def show_home():
    st.title("🎯 Object Detection and Tracking System")
    st.write("An AI-based application that finds and follows objects in videos "
             "using Python, OpenCV and YOLOv8n.")

    col1, col2 = st.columns(2)
    with col1:
        card("Project Objective",
             "Detect objects in an uploaded video, give every object a unique ID, "
             "count them and produce a processed video that can be downloaded.")
        card("Technologies Used",
             "<ul><li>Python</li><li>Streamlit (front end)</li>"
             "<li>OpenCV (video processing)</li><li>YOLOv8n (detection)</li>"
             "<li>NumPy and Pandas</li></ul>")
    with col2:
        card("How the System Works",
             "<ol><li>Upload a video (MP4, AVI or MOV).</li>"
             "<li>Every frame is read with OpenCV.</li>"
             "<li>YOLOv8n detects objects and their confidence scores.</li>"
             "<li>Centroid tracking links objects across frames and gives IDs.</li>"
             "<li>Boxes and labels are drawn and the video is saved.</li>"
             "<li>Counts are shown and the video can be downloaded.</li></ol>")
    st.info("Open **Object Detection** from the sidebar to get started.")


# --------------------------- OBJECT DETECTION PAGE ---------------------------
def run_detection(uploaded, confidence, show_boxes):
    """Save the upload to a temp file, process it and keep the result."""
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded.getvalue())
        temp_path = temp_file.name

    progress = st.progress(0.0, text="Processing video... please wait")
    try:
        model = get_model()
        result = process_video(temp_path, model, confidence, show_boxes,
                               progress_callback=lambda p: progress.progress(p))
        result["source"] = (uploaded.name, uploaded.size)
        st.session_state["result"] = result
        progress.empty()
        st.success("Detection completed!")
    except (ValueError, RuntimeError) as error:
        progress.empty()
        st.session_state.pop("result", None)
        st.error(str(error))
    except Exception as error:
        progress.empty()
        st.session_state.pop("result", None)
        st.error(f"Something went wrong: {error}")
    finally:
        os.remove(temp_path)


def show_detection(confidence, show_boxes):
    st.title("Object Detection")
    uploaded = st.file_uploader("Upload a video (MP4, AVI or MOV)",
                                type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.info("Please upload a video to begin.")
        return

    if st.button("Start Detection"):
        run_detection(uploaded, confidence, show_boxes)

    result = st.session_state.get("result")
    if result and result["source"] != (uploaded.name, uploaded.size):
        result = None                           # result belongs to another video

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Video")
        st.video(uploaded.getvalue())
    with col2:
        st.subheader("Processed Video")
        if result:
            with open(result["output_path"], "rb") as video_file:
                processed_bytes = video_file.read()
            st.video(processed_bytes)
            if not result["browser_ready"]:
                st.warning("The preview may not play in your browser, but the "
                           "downloaded video will work in any media player.")
        else:
            st.write("The processed video will appear here.")

    if result:
        st.subheader("Detection Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Objects Detected", result["total_objects"])
        m2.metric("Total Detections (all frames)", result["total_detections"])
        m3.metric("Frames Processed", result["frames"])

        if result["class_counts"]:
            table = pd.DataFrame(
                [{"Object": name, "Count": count}
                 for name, count in sorted(result["class_counts"].items())])
            st.write("**Object-wise count**")
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.info("No objects were detected. Try lowering the confidence threshold.")

        st.download_button("Download Processed Video", data=processed_bytes,
                           file_name="processed_video.mp4", mime="video/mp4")


# -------------------------------- ABOUT PAGE --------------------------------
def show_about():
    st.title("About the Project")
    card("Project Description",
         "This project detects and tracks objects in videos. It uses a "
         "pre-trained YOLOv8n model for detection and a centroid tracking "
         "algorithm to keep the same ID for the same object across frames.")
    col1, col2 = st.columns(2)
    with col1:
        card("Objectives",
             "<ul><li>Accept video files from the user</li>"
             "<li>Detect objects with bounding boxes, names and confidence</li>"
             "<li>Assign and keep unique IDs</li>"
             "<li>Count objects and display the result</li>"
             "<li>Let the user download the processed video</li></ul>")
        card("Working Methodology",
             "<ol><li>Read video frames with OpenCV</li>"
             "<li>Detect objects with YOLOv8n</li>"
             "<li>Find the centre of each box</li>"
             "<li>Match centres with the previous frame to keep IDs</li>"
             "<li>Draw boxes and save the output video</li></ol>")
    with col2:
        card("Tools and Technologies",
             "Python, Streamlit, OpenCV, YOLOv8n (Ultralytics), NumPy, Pandas.")
        card("Advantages and Applications",
             "<b>Advantages:</b> runs on a normal CPU, simple, free and easy to "
             "understand.<br><br><b>Applications:</b> traffic monitoring, "
             "people counting, security surveillance, retail analysis, "
             "smart-city projects.")


# ---------------------------------- SIDEBAR ----------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Object Detection", "About"])

confidence_value = 0.4
show_boxes_value = True
if page == "Object Detection":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Settings")
    confidence_value = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    show_boxes_value = st.sidebar.checkbox("Show bounding boxes", value=True)

if page == "Home":
    show_home()
elif page == "Object Detection":
    show_detection(confidence_value, show_boxes_value)
else:
    show_about()
