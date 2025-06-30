import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def visualize_tsne(
    csv_path,
    xrf_columns,
    label_column="City",
    xrf_transform="none",
    perplexity=30,
    n_iter=1000,
    random_state=42,
    n_components=2
):
    """
    Visualize XRF data using t-SNE.

    Parameters:
        csv_path: path to the CSV file.
        xrf_columns: list of XRF feature columns.
        label_column: column to color the points by (e.g., 'City', 'YearPeriod').
        xrf_transform: one of ['none', 'sqrt', 'log'].
        perplexity: t-SNE perplexity.
        n_iter: number of iterations.
        random_state: reproducibility seed.
        n_components: 2 for 2D plot, 3 for 3D plot
    """
    data = pd.read_csv(csv_path)
    xrf = data[xrf_columns].values.astype(np.float32)

    # Transformation
    if xrf_transform == "sqrt":
        xrf = np.sqrt(np.clip(xrf, a_min=0, a_max=None))
    elif xrf_transform == "log":
        xrf = np.log1p(np.clip(xrf, a_min=0, a_max=None))

    # Fit t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state
    )
    X_embedded = tsne.fit_transform(xrf)

    # Encode label for coloring
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data[label_column])
    label_names = label_encoder.classes_

    # Plot
    num_classes = len(label_names)
    palette = sns.color_palette("hls", num_classes)

    if n_components == 2:
        plt.figure(figsize=(12, 10))
        for i, name in enumerate(label_names):
            idx = (labels == i)
            plt.scatter(
                X_embedded[idx, 0], X_embedded[idx, 1],
                label=str(name), color=palette[i], s=18, alpha=0.8, edgecolors='w', linewidths=0.3
            )
        plt.title(f"t-SNE of XRF Data ({xrf_transform} transform) colored by {label_column}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title=label_column, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", ncol=1)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i, name in enumerate(label_names):
            idx = (labels == i)
            ax.scatter(
                X_embedded[idx, 0], X_embedded[idx, 1], X_embedded[idx, 2],
                label=str(name), color=palette[i], s=20, alpha=0.7, edgecolors='w', linewidths=0.3
            )
        ax.set_title(f"3D t-SNE of XRF Data ({xrf_transform} transform) colored by {label_column}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.legend(title=label_column, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.tight_layout()
        plt.show()


visualize_tsne(
    csv_path="../data/final_xrf_metal_data.csv",
    xrf_columns=[f"ch_{str(i).zfill(4)}" for i in range(1, 2049)],
    # xrf_columns=['Fe', 'Ni', 'Cu', 'Zn', 'Ag', 'Cd', 'Sn', 'Sb', 'Au', 'Hg', 'Pb', 'Bi'],
    label_column="PeriodGroup",
    xrf_transform="none",
    n_components=2  # Change to 3 for 3D plot
)