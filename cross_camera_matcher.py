import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class CrossCameraPlayerMatcher:
    def __init__(self, appearance_weight=0.4, position_weight=0.3, motion_weight=0.3):
        self.appearance_weight = appearance_weight
        self.position_weight = position_weight
        self.motion_weight = motion_weight

    def extract_appearance_features(self, frame, bbox):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(64)  # Return zero vector for invalid bbox

        # Extract player region
        player_region = frame[y1:y2, x1:x2]

        if player_region.size == 0:
            return np.zeros(64)

        # Resize to standard size
        player_region = cv2.resize(player_region, (64, 128))

        # Extract color histogram features
        hist_b = cv2.calcHist([player_region], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([player_region], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([player_region], [2], None, [16], [0, 256])

        # Normalize histograms
        hist_b = hist_b.flatten() / (hist_b.sum() + 1e-6)
        hist_g = hist_g.flatten() / (hist_g.sum() + 1e-6)
        hist_r = hist_r.flatten() / (hist_r.sum() + 1e-6)

        # Combine histograms
        color_features = np.concatenate([hist_b, hist_g, hist_r])

        # Add texture features (LBP-like)
        gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)

        # Simple texture feature: gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        texture_feature = np.mean(gradient_magnitude)

        # Combine features
        features = np.concatenate([color_features, [texture_feature]])

        return features[:64]  # Ensure fixed size

    def normalize_position(self, position, frame_shape):
        x, y = position
        h, w = frame_shape[:2]
        return [x / w, y / h]

    def compute_similarity_matrix(self, broadcast_players, tacticam_players,
                                  broadcast_frame, tacticam_frame,
                                  broadcast_shape, tacticam_shape,
                                  output_dir=None):
        if not broadcast_players or not tacticam_players:
            return np.array([]), [], []

        n_broadcast = len(broadcast_players)
        n_tacticam = len(tacticam_players)

        similarity_matrix = np.zeros((n_broadcast, n_tacticam))
        appearance_matrix = np.zeros((n_broadcast, n_tacticam))
        position_matrix = np.zeros((n_broadcast, n_tacticam))
        motion_matrix = np.zeros((n_broadcast, n_tacticam))

        broadcast_ids = list(broadcast_players.keys())
        tacticam_ids = list(tacticam_players.keys())

        for i, b_id in enumerate(broadcast_ids):
            b_player = broadcast_players[b_id]
            b_features = self.extract_appearance_features(broadcast_frame, b_player['bbox'])
            b_pos_norm = self.normalize_position(b_player['centroid'], broadcast_shape)

            for j, t_id in enumerate(tacticam_ids):
                t_player = tacticam_players[t_id]
                t_features = self.extract_appearance_features(tacticam_frame, t_player['bbox'])
                t_pos_norm = self.normalize_position(t_player['centroid'], tacticam_shape)

                # Appearance similarity
                appearance_sim = np.dot(b_features, t_features) / (
                        np.linalg.norm(b_features) * np.linalg.norm(t_features) + 1e-8)

                # Position similarity
                pos_dist = np.linalg.norm(np.array(b_pos_norm) - np.array(t_pos_norm))
                position_sim = 1.0 / (1.0 + pos_dist)

                # Motion similarity
                motion_sim = 0.5  # default neutral
                if b_player['history'] and t_player['history']:
                    b_motion = np.array(b_player['centroid']) - np.array(b_player['history'][-1])
                    t_motion = np.array(t_player['centroid']) - np.array(t_player['history'][-1])
                    b_motion_norm = b_motion / (np.linalg.norm(b_motion) + 1e-8)
                    t_motion_norm = t_motion / (np.linalg.norm(t_motion) + 1e-8)
                    motion_sim = max(0, np.dot(b_motion_norm, t_motion_norm))

                # Store individual components
                appearance_matrix[i, j] = appearance_sim
                position_matrix[i, j] = position_sim
                motion_matrix[i, j] = motion_sim

                # Final similarity
                total_sim = (self.appearance_weight * appearance_sim +
                             self.position_weight * position_sim +
                             self.motion_weight * motion_sim)
                similarity_matrix[i, j] = total_sim

        # Visualize
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.visualize_similarity_matrices(appearance_matrix, position_matrix, motion_matrix, similarity_matrix,
                                               output_dir)

        return similarity_matrix, broadcast_ids, tacticam_ids

    def match_players(self, broadcast_players, tacticam_players,
                      broadcast_frame, tacticam_frame,
                      broadcast_shape, tacticam_shape, threshold=0.5):
        sim_matrix, b_ids, t_ids = self.compute_similarity_matrix(
            broadcast_players, tacticam_players,
            broadcast_frame, tacticam_frame,
            broadcast_shape, tacticam_shape,
            output_dir="tracking_output"
        )

        if sim_matrix.size == 0:
            return {}

        # Convert similarity to cost (Hungarian algorithm minimizes cost)
        cost_matrix = 1.0 - sim_matrix

        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = {}
        for row, col in zip(row_indices, col_indices):
            if sim_matrix[row, col] > threshold:  # Only accept matches above threshold
                broadcast_id = b_ids[row]
                tacticam_id = t_ids[col]
                matches[broadcast_id] = {
                    'tacticam_id': tacticam_id,
                    'similarity': sim_matrix[row, col]
                }

        return matches

    def visualize_similarity_matrices(self, app_sim, pos_sim, mot_sim, total_sim, output_dir):
        matrices = {
            "Appearance Similarity": app_sim,
            "Position Similarity": pos_sim,
            "Motion Similarity": mot_sim,
            "Combined Similarity": total_sim,
        }

        for title, matrix in matrices.items():
            plt.figure(figsize=(10, 7))
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
                        xticklabels=True, yticklabels=True)
            plt.title(title)
            plt.xlabel("Tacticam Players")
            plt.ylabel("Broadcast Players")
            plt.tight_layout()
            filename = title.lower().replace(" ", "_") + ".png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

