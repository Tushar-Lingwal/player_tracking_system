from collections import deque
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from base_tracker import BaseTracker

class PlayerTracker(BaseTracker):
    def __init__(self, max_disappeared=30, max_distance=100):
        super().__init__()
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, confidence, class_name, frame_id):
        self.objects[self.next_id] = {
            'centroid': centroid, 'bbox': bbox,
            'confidence': confidence, 'class_name': class_name,
            'frame_id': frame_id, 'history': deque(maxlen=30), 'features': []
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections, frame_id):
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            # No existing objects, register all detections
            for detection in detections:
                centroid = detection['center']
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                self.register(centroid, bbox, confidence, class_name, frame_id)
        else:
            # Get centroids of existing objects and new detections
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[obj_id]['centroid'] for obj_id in object_ids])
            detection_centroids = np.array([det['center'] for det in detections])

            # Compute distance matrix
            D = cdist(object_centroids, detection_centroids)

            # Hungarian algorithm for optimal assignment
            rows, cols = linear_sum_assignment(D)

            used_row_indices = set()
            used_col_indices = set()

            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row >= len(object_ids) or D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                detection = detections[col]

                # Update object
                old_centroid = self.objects[object_id]['centroid']
                self.objects[object_id]['history'].append(old_centroid)
                self.objects[object_id]['centroid'] = detection['center']
                self.objects[object_id]['bbox'] = detection['bbox']
                self.objects[object_id]['confidence'] = detection['confidence']
                self.objects[object_id]['frame_id'] = frame_id

                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, len(object_ids))).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            # Mark unmatched objects as disappeared
            for row in unused_row_indices:
                if row < len(object_ids):
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # Register new objects for unmatched detections
            for col in unused_col_indices:
                detection = detections[col]
                centroid = detection['center']
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                self.register(centroid, bbox, confidence, class_name, frame_id)

        return self.objects

