# Multi-Camera Football Player Tracking System

This project implements a **multi-camera player tracking system** using **YOLOv11** for object detection and **BoxMOT (DeepOCSORT)** for robust tracking. It supports **cross-camera ID matching** between a standard broadcast view and a top-down tacticam view.

It’s built for sports analytics use cases such as:
- Performance tracking
- Player heatmaps
- Cross-view player analysis

---

## 📂 Project Structure

tracking_system/

├── base_tracker.py  ->   Base class for common tracker structure

├── player_tracker.py  ->  YOLO-based player tracker

├── cross_camera_matcher.py  ->  Cross-camera ID matching logic

├── global_id_manager.py  ->  Manages consistent global IDs

├── comprehensive_tracking_system.py  ->  Full system integration

├── visualizer.py  ->  Graphs, histograms, and visual summaries

├── metrics.py  ->  Accuracy, precision, recall, F1 scoring

├── main.py # Entry point for system execution


---

## Core Features

| Feature                         | Description |
|----------------------------------|-------------|
| YOLOv11 Detection             | Detect players and classify them with high confidence. |
| DeepOCSORT + BoxMOT          | Tracks each player across frames in a single camera. |
| Cross-Camera ID Matching     | Uses appearance, position, and motion features. |
| Accuracy Evaluation          | Precision, recall, F1-score based on ID consistency. |
| Visual Analysis              | Match similarity histogram, ID tracking stats, overlays. |

---

## Setup Instructions

install dependencies pip install -r requirements.txt

How It Works
1. Detection and Tracking
player_tracker.py uses YOLOv11 + BoxMOT to detect and track players.
It registers new players, tracks their centroids and bounding boxes across frames.

3. Cross-View Player Matching
cross_camera_matcher.py calculates a similarity matrix between two views based on:

- Appearance (color histogram)
- Relative Position
- Motion history

example Image : 
![image](https://github.com/user-attachments/assets/bd37a69d-95d9-42cc-8523-1630cf5b46be)
left sied is boadcast tracker and right side is tacticam tracker

it Uses the Hungarian Algorithm for optimal matching.


Combined Simirity heatplot is given as : 
![image](https://github.com/user-attachments/assets/db2b4c12-dbbf-49d6-b57b-abab6ee7035f)

Positional Similarity heatplot is given as :
![image](https://github.com/user-attachments/assets/ca53b901-1fab-4771-a017-798e3f3941b0)

3. Global ID Management
global_id_manager.py ensures consistent player IDs across both views.
Handles new, disappearing, and reappearing players.

To view full model in working you can see the  "combined_tracking" mp4 file in the Repo.




