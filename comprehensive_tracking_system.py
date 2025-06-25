from player_tracker import PlayerTracker
from cross_camera_matcher import CrossCameraPlayerMatcher
from global_id_manager import GlobalIDManager
from visualizer import TrackingVisualizer
from metrics import AccuracyMetrics
import os, cv2, json, numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque


class ComprehensivePlayerTrackingSystem:
    def __init__(self, model_path="best.pt"):
        """
        Initialize the comprehensive tracking system
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Initialize trackers for both cameras
        self.broadcast_tracker = PlayerTracker(max_disappeared=15, max_distance=80)
        self.tacticam_tracker = PlayerTracker(max_disappeared=15, max_distance=80)

        # Initialize cross-camera matcher
        self.matcher = CrossCameraPlayerMatcher()

        # Initialize global ID manager
        self.global_id_manager = GlobalIDManager()

        # Storage for results
        self.tracking_results = {
            'broadcast': [],
            'tacticam': [],
            'matches': []
        }

    def process_synchronized_videos(self, broadcast_path, tacticam_path,
                                    output_dir="output", max_frames=200,
                                    confidence_threshold=0.4):
        """
        Process both videos synchronously and perform cross-camera matching
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Open video captures
        cap_broadcast = cv2.VideoCapture(broadcast_path)
        cap_tacticam = cv2.VideoCapture(tacticam_path)

        if not cap_broadcast.isOpened() or not cap_tacticam.isOpened():
            print("Error: Could not open one or both video files")
            return

        # Get video properties
        fps_b = int(cap_broadcast.get(cv2.CAP_PROP_FPS))
        fps_t = int(cap_tacticam.get(cv2.CAP_PROP_FPS))
        width_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_broadcast = cv2.VideoWriter(
            os.path.join(output_dir, 'broadcast_tracked.mp4'),
            fourcc, fps_b, (width_b, height_b)
        )
        out_tacticam = cv2.VideoWriter(
            os.path.join(output_dir, 'tacticam_tracked.mp4'),
            fourcc, fps_t, (width_t, height_t)
        )
        out_combined = cv2.VideoWriter(
            os.path.join(output_dir, 'combined_tracking.mp4'),
            fourcc, min(fps_b, fps_t), (width_b + width_t, max(height_b, height_t))
        )

        frame_count = 0

        print("Starting synchronized video processing...")

        while frame_count < max_frames:
            # Read frames from both videos
            ret_b, frame_b = cap_broadcast.read()
            ret_t, frame_t = cap_tacticam.read()

            if not ret_b or not ret_t:
                break

            # Run detection on both frames
            results_b = self.model(frame_b, conf=confidence_threshold, verbose=False)
            results_t = self.model(frame_t, conf=confidence_threshold, verbose=False)

            # Extract player detections
            detections_b = self.extract_player_detections(results_b)
            detections_t = self.extract_player_detections(results_t)

            # Update trackers
            tracked_b = self.broadcast_tracker.update(detections_b, frame_count)
            tracked_t = self.tacticam_tracker.update(detections_t, frame_count)

            # Filter for players only
            players_b = {k: v for k, v in tracked_b.items() if v['class_name'] == 'player'}
            players_t = {k: v for k, v in tracked_t.items() if v['class_name'] == 'player'}

            matches = {}
            if players_b and players_t:
                matches = self.matcher.match_players(
                    players_b, players_t, frame_b, frame_t,
                    (height_b, width_b), (height_t, width_t)
                )

                # Assign global IDs to matched players
                for broadcast_id, match_data in matches.items():
                    tacticam_id = match_data['tacticam_id']
                    global_id = self.global_id_manager.assign_global_id(broadcast_id, tacticam_id)

                    # Add global ID to the match data
                    match_data['global_id'] = global_id

            # Clean up unused global IDs
            active_broadcast_ids = set(players_b.keys())
            active_tacticam_ids = set(players_t.keys())
            self.global_id_manager.cleanup_unused_ids(active_broadcast_ids, active_tacticam_ids)

            # Draw tracking results
            annotated_b = self.draw_tracking_results(frame_b, tracked_b, matches, 'broadcast')
            annotated_t = self.draw_tracking_results(frame_t, tracked_t, matches, 'tacticam')

            # Create combined view
            combined_frame = self.create_combined_view(annotated_b, annotated_t, matches)

            # Write frames
            out_broadcast.write(annotated_b)
            out_tacticam.write(annotated_t)
            out_combined.write(combined_frame)

            # Store results
            self.tracking_results['broadcast'].append({
                'frame': frame_count,
                'tracked_objects': dict(tracked_b),
                'players_only': dict(players_b)
            })
            self.tracking_results['tacticam'].append({
                'frame': frame_count,
                'tracked_objects': dict(tracked_t),
                'players_only': dict(players_t)
            })
            self.tracking_results['matches'].append({
                'frame': frame_count,
                'matches': dict(matches)
            })

            frame_count += 1

            if frame_count % 20 == 0:
                print(f"Processed {frame_count}/{max_frames} frames - "
                      f"B: {len(players_b)} players, T: {len(players_t)} players, "
                      f"Matches: {len(matches)}")

        # Cleanup
        cap_broadcast.release()
        cap_tacticam.release()
        out_broadcast.release()
        out_tacticam.release()
        out_combined.release()
        cv2.destroyAllWindows()

        # Save tracking results
        self.save_results(output_dir)

        print(f"\nProcessing complete! Results saved in '{output_dir}' directory")
        self.print_final_statistics()

    def extract_player_detections(self, results):
        """
        Extract detection data from YOLO results
        """
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'area': float((x2 - x1) * (y2 - y1))
                    }
                    detections.append(detection)

        return detections

    def draw_tracking_results(self, frame, tracked_objects, matches, camera_type):
        annotated_frame = frame.copy()

        for obj_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            class_name = obj_data['class_name']
            confidence = obj_data['confidence']

            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            # Choose color based on class and tracking status
            if class_name == 'player':
                # Get global ID for this player
                global_id = self.global_id_manager.get_global_id(camera_type, obj_id)

                # Check if this player is matched
                is_matched = False
                matched_id = None

                if camera_type == 'broadcast':
                    if obj_id in matches:
                        is_matched = True
                        matched_id = matches[obj_id]['tacticam_id']
                        color = (0, 255, 0)  # Green for matched
                    else:
                        color = (0, 0, 255)  # Red for unmatched
                else:  # tacticam
                    for b_id, match_data in matches.items():
                        if match_data['tacticam_id'] == obj_id:
                            is_matched = True
                            matched_id = b_id
                            color = (0, 255, 0)  # Green for matched
                            break
                    else:
                        color = (0, 0, 255)  # Red for unmatched
            else:
                color = (255, 255, 0)  # Yellow for non-players
                global_id = None

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Create label with global ID
            if class_name == 'player':
                if global_id is not None:
                    # Show global ID prominently
                    main_label = f"ID:{global_id}"
                    detail_label = f"({camera_type[0].upper()}{obj_id}, {confidence:.2f})"
                else:
                    # No global ID assigned yet
                    main_label = f"{camera_type[0].upper()}{obj_id}"
                    detail_label = f"({confidence:.2f})"
            else:
                main_label = f"{class_name[0].upper()}{obj_id}"
                detail_label = f"({confidence:.2f})"

            # Draw main label (larger, more prominent)
            main_label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - main_label_size[1] - 25),
                          (x1 + main_label_size[0], y1 - 15), color, -1)
            cv2.putText(annotated_frame, main_label, (x1, y1 - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw detail label (smaller)
            detail_label_size = cv2.getTextSize(detail_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - detail_label_size[1] - 10),
                          (x1 + detail_label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, detail_label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw center point with global ID for players
            if class_name == 'player' and global_id is not None:
                center_x, center_y = int(obj_data['centroid'][0]), int(obj_data['centroid'][1])
                cv2.circle(annotated_frame, (center_x, center_y), 8, color, -1)
                cv2.putText(annotated_frame, str(global_id), (center_x - 10, center_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw trajectory for players
            if class_name == 'player' and len(obj_data['history']) > 1:
                points = list(obj_data['history']) + [obj_data['centroid']]
                for i in range(1, len(points)):
                    cv2.line(annotated_frame,
                             (int(points[i - 1][0]), int(points[i - 1][1])),
                             (int(points[i][0]), int(points[i][1])),
                             color, 2)

        # Add camera label
        cv2.putText(annotated_frame, camera_type.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return annotated_frame

    def create_combined_view(self, frame_b, frame_t, matches):
        """
        Create side-by-side combined view
        """
        h_b, w_b = frame_b.shape[:2]
        h_t, w_t = frame_t.shape[:2]

        # Resize frames to same height
        target_height = max(h_b, h_t)

        if h_b != target_height:
            frame_b = cv2.resize(frame_b, (int(w_b * target_height / h_b), target_height))
        if h_t != target_height:
            frame_t = cv2.resize(frame_t, (int(w_t * target_height / h_t), target_height))

        # Combine frames
        combined = np.hstack([frame_b, frame_t])

        # Add match information
        match_text = f"Matches: {len(matches)}"
        cv2.putText(combined, match_text, (10, target_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return combined

    def save_results(self, output_dir):
        """
        Save tracking results to files
        """

        # Convert numpy arrays and deques to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, deque):  # Handle deque objects
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        results_to_save = convert_numpy(self.tracking_results)

        with open(os.path.join(output_dir, 'tracking_results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)

        # Save player mappings
        mappings = self.extract_player_mappings()
        with open(os.path.join(output_dir, 'player_mappings.json'), 'w') as f:
            json.dump(mappings, f, indent=2)

        print(f"Results saved to {output_dir}/")

    def extract_player_mappings(self):
        """
        Extract consistent player mappings across all frames
        """
        # Count how often each broadcast player is matched to each tacticam player
        mapping_counts = defaultdict(lambda: defaultdict(int))

        for frame_matches in self.tracking_results['matches']:
            for b_id, match_data in frame_matches['matches'].items():
                t_id = match_data['tacticam_id']
                mapping_counts[b_id][t_id] += 1

        # Extract most consistent mappings
        final_mappings = {}
        for b_id, t_matches in mapping_counts.items():
            if t_matches:
                # Find the tacticam ID that was matched most often
                best_t_id = max(t_matches.items(), key=lambda x: x[1])
                final_mappings[f"broadcast_{b_id}"] = {
                    "tacticam_id": best_t_id[0],
                    "match_count": best_t_id[1],
                    "confidence": best_t_id[1] / sum(t_matches.values())
                }

        return final_mappings

    def print_final_statistics(self):
        """
        Print final tracking and matching statistics
        """
        print("\n=== Final Tracking Statistics ===")

        # Count unique players in each camera
        broadcast_players = set()
        tacticam_players = set()
        total_matches = []

        for frame_data in self.tracking_results['broadcast']:
            broadcast_players.update(frame_data['players_only'].keys())

        for frame_data in self.tracking_results['tacticam']:
            tacticam_players.update(frame_data['players_only'].keys())

        for frame_matches in self.tracking_results['matches']:
            total_matches.extend(frame_matches['matches'].values())

        print(f"Unique players tracked in broadcast: {len(broadcast_players)}")
        print(f"Unique players tracked in tacticam: {len(tacticam_players)}")
        print(f"Total match attempts: {len(total_matches)}")

        if total_matches:
            avg_similarity = np.mean([m['similarity'] for m in total_matches])
            print(f"Average match similarity: {avg_similarity:.3f}")

        # Print player mappings
        mappings = self.extract_player_mappings()
        print(f"\nConsistent player mappings found: {len(mappings)}")
        for b_id, mapping in mappings.items():
            print(f"  {b_id} ↔ tacticam_{mapping['tacticam_id']} "
                  f"(confidence: {mapping['confidence']:.2f}, "
                  f"matches: {mapping['match_count']})")