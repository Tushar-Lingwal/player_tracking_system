from comprehensive_tracking_system import ComprehensivePlayerTrackingSystem

def main():
    system = ComprehensivePlayerTrackingSystem("best.pt")
    system.process_synchronized_videos(
        broadcast_path="broadcast.mp4",
        tacticam_path="tacticam.mp4",
        output_dir="tracking_output",
        max_frames=100,
        confidence_threshold=0.4
    )

if __name__ == "__main__":
    main()
