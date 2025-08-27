import streamlit as st
import os
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Video Analytics for Customer Time Spent")
st.write("Upload a video to analyze customer dwell time at billing counters. The system will detect and track customers, calculate their time spent in defined zones, and generate a report.")

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8 model from cache or downloads it."""
    model = YOLO('yolov8n.pt')
    return model

model = load_yolo_model()

# --- Main Application Logic ---
uploaded_file = st.file_uploader("Choose a video file to analyze", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.success(f"Video '{uploaded_file.name}' uploaded successfully.")
    
    # --- ROI Definition ---
    # These coordinates should be adjusted based on the specific video's layout.
    # For a real application, you might build a UI to draw these zones.
    st.sidebar.header("Counter Zone Configuration")
    st.sidebar.info("These are predefined zones. In a full application, you could allow users to draw these.")

    counter_rois = {
        "Counter 1": np.array([
            (100, 200), (300, 200), (300, 450), (100, 450)
        ], np.int32),
        "Counter 2": np.array([
            (700, 220), (900, 220), (900, 470), (700, 470)
        ], np.int32),
        "Counter 3": np.array([
            (1300, 200), (1500, 200), (1500, 620), (1300, 620)
        ], np.int32)
    }

    if st.button("Start Analysis", type="primary"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open uploaded video file.")
        else:
            st.info("Video processing started. The annotated video will appear below.")
            
            # Placeholders for video and report
            frame_placeholder = st.empty()
            report_placeholder = st.empty()
            
            # --- Data Structures for Logic ---
            customer_entry_log = {}
            final_results_log = []

            # --- Main Processing Loop ---
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Run YOLOv8 tracking
                results = model.track(frame, persist=True, classes=0, verbose=False)
                annotated_frame = results[0].plot()

                detected_ids_in_frame = set()
                if results[0].boxes.id is not None:
                    detected_ids_in_frame = set(results[0].boxes.id.int().cpu().tolist())
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    for i, box in enumerate(boxes):
                        track_id = track_ids[i]
                        person_point = (int((box[0] + box[2]) / 2), int(box[3]))

                        is_in_any_roi = False
                        for name, roi in counter_rois.items():
                            if cv2.pointPolygonTest(roi, person_point, False) >= 0:
                                is_in_any_roi = True
                                if track_id not in customer_entry_log:
                                    customer_entry_log[track_id] = {
                                        "counter": name, "entry_time": current_time_sec
                                    }
                                break
                        
                        if not is_in_any_roi and track_id in customer_entry_log:
                            entry_data = customer_entry_log.pop(track_id)
                            duration = current_time_sec - entry_data["entry_time"]
                            if duration > 1.0:
                                final_results_log.append({
                                    "CustomerID": track_id, "Counter": entry_data["counter"],
                                    "EntryTime(s)": round(entry_data["entry_time"], 2),
                                    "ExitTime(s)": round(current_time_sec, 2), "Duration(s)": round(duration, 2)
                                })
                
                lost_ids = set(customer_entry_log.keys()) - detected_ids_in_frame
                for track_id in lost_ids:
                    entry_data = customer_entry_log.pop(track_id)
                    duration = current_time_sec - entry_data["entry_time"]
                    if duration > 1.0:
                        final_results_log.append({
                            "CustomerID": track_id, "Counter": entry_data["counter"],
                            "EntryTime(s)": round(entry_data["entry_time"], 2),
                            "ExitTime(s)": round(current_time_sec, 2), "Duration(s)": round(duration, 2)
                        })

                # --- VISUALIZATION in Streamlit ---
                for name, roi in counter_rois.items():
                    cv2.polylines(annotated_frame, [roi], isClosed=True, color=(0, 255, 255), thickness=2)
                
                # Convert color from BGR to RGB for Streamlit display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, use_column_width=True)

            cap.release()
            tfile.close() # Close and delete the temporary file
            os.remove(video_path)

            st.success("Video processing complete!")

            # --- Display Final Report ---
            if final_results_log:
                st.subheader("Customer Activity Report")
                df = pd.DataFrame(final_results_log)
                df = df.sort_values(by=['CustomerID', 'EntryTime(s)']).reset_index(drop=True)
                report_placeholder.dataframe(df)
                
                # Allow downloading the CSV
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(df)
                st.download_button(
                    label="Download Report as CSV",
                    data=csv,
                    file_name='customer_activity_report.csv',
                    mime='text/csv',
                )
            else:
                st.warning("No significant customer activity was logged.")
