import os
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Setup ---
# Create a directory to save the snapshots if it doesn't exist
output_snapshot_dir = 'counter_snapshots'
os.makedirs(output_snapshot_dir, exist_ok=True)

model = YOLO('yolov8n.pt')
video_path = r"F:\Intership\input.mp4" # Using raw string for Windows path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# --- ACTION REQUIRED: Define your THREE counter zones (ROIs) ---
# Find the coordinates for all three counters from your video frame.
counter_rois = {
    "Counter 1": np.array([
        (100, 200), (300, 200), (300, 450), (100, 450)
    ], np.int32),
    "Counter 2": np.array([
        (700, 220), (900, 220), (900, 470), (700, 470)
    ], np.int32),
    # CORRECTED: Counter 3 is now same size as Counter 2, but shifted right
    "Counter 3": np.array([
        (1300, 320), (1500, 200), (1500, 620), (1300, 620)
    ], np.int32)
}

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
    results = model.track(frame, persist=True, classes=0)
    annotated_frame = results[0].plot() # Get annotated frame early for snapshots

    detected_ids_in_frame = set()
    if results[0].boxes.id is not None:
        detected_ids_in_frame = set(results[0].boxes.id.int().cpu().tolist())
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # --- Logic for Entry and Exit ---
        for i, box in enumerate(boxes):
            track_id = track_ids[i]
            person_point = (int((box[0] + box[2]) / 2), int(box[3])) # Bottom-center

            is_in_any_roi = False
            for name, roi in counter_rois.items():
                if cv2.pointPolygonTest(roi, person_point, False) >= 0:
                    is_in_any_roi = True
                    
                    # --- ENTRY & SNAPSHOT LOGIC ---
                    if track_id not in customer_entry_log:
                        print(f"EVENT: Customer {track_id} entered {name} at {current_time_sec:.2f}s.")
                        customer_entry_log[track_id] = {
                            "counter": name,
                            "entry_time": current_time_sec
                        }
                        
                        # --- Take Snapshot ---
                        x, y, w, h = cv2.boundingRect(roi)
                        counter_snapshot = annotated_frame[y:y+h, x:x+w]
                        snapshot_filename = os.path.join(output_snapshot_dir, f"customer_{track_id}_at_{name}_time_{int(current_time_sec)}.jpg")
                        cv2.imwrite(snapshot_filename, counter_snapshot)
                        print(f"  -> Saved snapshot to {snapshot_filename}")

                    break # Person found, move to the next person

            # --- EXIT LOGIC (Person moves out of zone) ---
            if not is_in_any_roi and track_id in customer_entry_log:
                entry_data = customer_entry_log.pop(track_id)
                duration = current_time_sec - entry_data["entry_time"]
                if duration > 1.0: # Log only if duration is significant
                    print(f"EVENT: Customer {track_id} exited {entry_data['counter']} after {duration:.2f}s.")
                    final_results_log.append({
                        "CustomerID": track_id, "Counter": entry_data["counter"],
                        "EntryTime(s)": round(entry_data["entry_time"], 2),
                        "ExitTime(s)": round(current_time_sec, 2), "Duration(s)": round(duration, 2)
                    })

    # --- EXIT LOGIC (Person disappears from frame) ---
    lost_ids = set(customer_entry_log.keys()) - detected_ids_in_frame
    for track_id in lost_ids:
        entry_data = customer_entry_log.pop(track_id)
        duration = current_time_sec - entry_data["entry_time"]
        if duration > 1.0: # Log only if duration is significant
            print(f"EVENT: Customer {track_id} (lost) exited {entry_data['counter']} after {duration:.2f}s.")
            final_results_log.append({
                "CustomerID": track_id, "Counter": entry_data["counter"],
                "EntryTime(s)": round(entry_data["entry_time"], 2),
                "ExitTime(s)": round(current_time_sec, 2), "Duration(s)": round(duration, 2)
            })

    # --- VISUALIZATION ---
    for name, roi in counter_rois.items():
        cv2.polylines(annotated_frame, [roi], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.imshow("Video Analytics", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# --- FINAL REPORT GENERATION ---
print("\n--- Customer Activity Report ---")
if final_results_log:
    df = pd.DataFrame(final_results_log)
    # Sort by CustomerID and then by Entry Time for a logical report
    df = df.sort_values(by=['CustomerID', 'EntryTime(s)']).reset_index(drop=True)
    
    print(df.to_string())
    output_csv_path = 'customer_activity_report.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"\nReport successfully saved to {output_csv_path}")
else:
    print("No significant customer activity was logged.")
