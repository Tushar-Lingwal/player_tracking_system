import matplotlib.pyplot as plt

class TrackingVisualizer:
    def __init__(self):
        pass

    def plot_match_similarity_histogram(self, similarities, output_dir):
        plt.hist(similarities, bins=20, color='skyblue', edgecolor='black')
        plt.title("Match Similarity Distribution")
        plt.xlabel("Similarity Score")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig(f"{output_dir}/similarity_distribution.png")
        plt.close()
