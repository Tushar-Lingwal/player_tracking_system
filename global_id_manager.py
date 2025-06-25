class GlobalIDManager:
    def __init__(self):
        self.next_global_id = 1
        self.broadcast_to_global = {}
        self.tacticam_to_global = {}
        self.global_id_assignments = {}

    def assign_global_id(self, broadcast_id, tacticam_id):
        """
        Assign a global ID to matched players
        """
        # Check if either player already has a global ID
        global_id = None

        if broadcast_id in self.broadcast_to_global:
            global_id = self.broadcast_to_global[broadcast_id]
        elif tacticam_id in self.tacticam_to_global:
            global_id = self.tacticam_to_global[tacticam_id]
        else:
            # Assign new global ID
            global_id = self.next_global_id
            self.next_global_id += 1

        # Update mappings
        self.broadcast_to_global[broadcast_id] = global_id
        self.tacticam_to_global[tacticam_id] = global_id
        self.global_id_assignments[global_id] = {
            'broadcast_id': broadcast_id,
            'tacticam_id': tacticam_id
        }

        return global_id

    def get_global_id(self, camera_type, tracker_id):
        """
        Get global ID for a tracker ID
        """
        if camera_type == 'broadcast':
            return self.broadcast_to_global.get(tracker_id, None)
        else:  # tacticam
            return self.tacticam_to_global.get(tracker_id, None)

    def cleanup_unused_ids(self, active_broadcast_ids, active_tacticam_ids):
        """
        Remove mappings for IDs that are no longer active
        """
        # Clean broadcast mappings
        to_remove = []
        for b_id, g_id in self.broadcast_to_global.items():
            if b_id not in active_broadcast_ids:
                to_remove.append((b_id, g_id))

        for b_id, g_id in to_remove:
            del self.broadcast_to_global[b_id]
            if g_id in self.global_id_assignments:
                del self.global_id_assignments[g_id]

        # Clean tacticam mappings
        to_remove = []
        for t_id, g_id in self.tacticam_to_global.items():
            if t_id not in active_tacticam_ids:
                to_remove.append((t_id, g_id))

        for t_id, g_id in to_remove:
            del self.tacticam_to_global[t_id]
