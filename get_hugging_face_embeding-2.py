"""This module is used to extract embeddings from the provided Hugging Face model"""

import os
import warnings
from random import randint
from typing import List, Optional
import streamlit as st
import openai
from openai import OpenAI
import base64
import tempfile
import re
import bpy  # pylint: disable=import-error
import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import plotly.express as px  # pylint: disable=import-error
import torch  # pylint: disable=import-error
import umap  # pylint: disable=import-error
from sklearn.decomposition import PCA  # pylint: disable=import-error
from sklearn.manifold import MDS, TSNE, Isomap  # pylint: disable=import-error
from transformers import AutoModel, AutoTokenizer  # pylint: disable=import-error

# stop future warning
warnings.filterwarnings("ignore")

# Constants
N_COMPONENTS = 3
PERPLEXITY = 30
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# Call the function with your image file
add_bg_from_local("/Users/akalpitdawkhar/Desktop/School/SEM_3/XAI/XAI_FinalProject_GradCAM/background.png")

# Main title
# Main title, centered
st.markdown(
    "<h1 style='text-align: center;'>🕶️ PlotVerseXR  🕶️</h1>", unsafe_allow_html=True
)

# Subheader with custom style
st.markdown(
    "<h2 style='text-align: center; color: #FF5733;'>Visualize 3D Data like its meant to be seen!</h2>",
    unsafe_allow_html=True,
)

# Step 1: Model Input
st.header("Step 1: Model Input")
HUGGING_MODEL = st.text_input("Enter the Hugging Face model name or path:",value="sentence-transformers/all-MiniLM-L6-v2",max_chars=200)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    st.warning("CUDA not available, using CPU. Performance may be slower.")

# Step 2: Ask for the file output path
st.header("Step 2: Output File Path")
output_file_path = st.text_input("Enter the output file path for the .dae file:",value="output.dae",)


# Step 3: Dropdown to choose input type
st.header("Step 3: Choose Input Type")
input_type = st.selectbox("Select the input type:",options=["Enter domains to generate words", "Sentences", "Upload CSV"],)


# Reduction algorithm options
reduction_algorithms = ["T-SNE", "UMAP", "PCA", "MDS", "ISOMAP"]
st.header("Step 4: Dimensionality Reduction Algorithm")
default_index = reduction_algorithms.index("UMAP")
method = st.selectbox("Select a dimensionality reduction algorithm:",options=reduction_algorithms,index=default_index)


class HuggingFaceEmbeddingViz:
    """This class is used to extract embeddings from the provided Hugging Face model"""

    def __init__(self, model_name: str, device_: torch.device = torch.device("cpu")):
        """This class is used to extract embeddings from the provided Hugging Face model

        Args:
            model_name (str): The name of the Hugging Face model to use
            device_ (torch.device, optional): The device to use for the model.
            Defaults to torch.device("cpu")
        """

        try:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).to(device_)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

        except (OSError, ValueError) as e_err:
            raise RuntimeError(
                f"Failed to load model or tokenizer for {model_name}: {e_err}"
            ) from e_err

        self.model_name = model_name
        self.device = device_

    def get_model_embeddings(self, text_list: List[str]) -> np.ndarray:
        """This function is used to extract embeddings for the passed text as
        defined by the LLM models embedding space

        Args:
            text_list (List[str]): The list of text to extract embeddings from

        Returns:
            np.ndarray: The embeddings of the text

        """
        if self.device.type == "cpu" and not os.getenv("TEST_ENV"):
            raise RuntimeError("This model requires a CUDA-enabled GPU to run.")

        embeddings = []
        # Dynamically set max_length based on the model
        max_length = self.tokenizer.model_max_length

        with torch.no_grad():
            for entry in text_list:
                input_data = self.tokenizer(
                    entry, return_tensors="pt", truncation=True, max_length=max_length
                ).to(self.device)
                input_data = {k: v.to(self.device) for k, v in input_data.items()}
                last_hidden_state = self.model(**input_data).last_hidden_state
                sentence_embedding = last_hidden_state[:, 0, :].cpu().numpy().flatten()
                embeddings.append(sentence_embedding)

        return np.array(embeddings)

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, file_path: str) -> None:
        """Save embeddings to a file

        Args:
            embeddings (np.ndarray): The embeddings to save
            file_path (str): The path to the file where embeddings will be saved
        """
        np.save(file_path, embeddings)

    @staticmethod
    def load_embeddings(file_path: str) -> np.ndarray:
        """Load embeddings from a file

        Args:
            file_path (str): The path to the file from which embeddings will be loaded

        Returns:
            np.ndarray: The loaded embeddings
        """
        return np.load(file_path)

    def generate_visualization(
        self,
        embeddings: np.ndarray,
        labels_: Optional[List[str]] = None,
        color_: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """This function is used to generate a visualization of the
        embedding space of the LLM model, using the plotly library

        Args:
            embeddings (np.ndarray): The embeddings to visualize
            labels_ (Optional[List[str]], optional): Labels for the embeddings. Defaults to None.
            color_ (Optional[List[str]], optional): Color for the embeddings. Defaults to None.
            **kwargs: Additional keyword arguments
                - method (str): The method to use for visualization. Defaults to "umap".
                - plot (bool): Whether to plot the embeddings. Defaults to False.
        Returns:
            pd.DataFrame: A DataFrame containing the reduced embeddings
        """

        # method = kwargs.get("method", "umap")

        reducer, title, x_label, y_label, z_label = self._get_reducer(method, labels_)
        reduced_embeddings = reducer.fit_transform(embeddings)

        plot = kwargs.get("plot", False)

        if plot:
            self._plot_embeddings(
                reduced_embeddings,
                labels_,
                color_,
                title=title,
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
            )

        reduced_embeddings_df = pd.DataFrame(
            reduced_embeddings, columns=[x_label, y_label, z_label]
        )
        if labels_ is not None:
            reduced_embeddings_df["Label"] = labels_

        if color_ is not None:
            reduced_embeddings_df["sector"] = color_

        return reduced_embeddings_df

    def _get_reducer(self, method: str, labels_: Optional[List[str]]) -> tuple:
        """Helper function to get the appropriate reducer and labels for the visualization method"""
        if method == "pca":
            reducer = PCA(n_components=N_COMPONENTS)
            title = f"PCA of {self.model_name} embedding space"
        elif method == "tsne":
            if len(labels_) > PERPLEXITY:
                perplexity = PERPLEXITY
            else:
                perplexity = len(labels_) - 1

            reducer = TSNE(
                n_components=N_COMPONENTS,
                perplexity=perplexity,
                n_iter=300,
                random_state=RANDOM_STATE,
            )
            title = f"t-SNE of {self.model_name} embedding space"
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=N_COMPONENTS,
                n_neighbors=N_NEIGHBORS,
                min_dist=MIN_DIST,
                random_state=RANDOM_STATE,
            )
            title = f"UMAP of {self.model_name} Embeddings"
        elif method == "mds":
            reducer = MDS(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
            title = f"MDS of {self.model_name} embedding space"
        elif method == "isomap":
            reducer = Isomap(n_components=N_COMPONENTS)
            title = f"Isomap of {self.model_name} embedding space"
        else:
            raise ValueError(
                "Invalid method. Choose from 'pca', 'tsne', 'umap', 'mds', or 'isomap'."
            )

        if method != "pca":
            x_label = "Component 1"
            y_label = "Component 2"
            z_label = "Component 3"
        else:
            x_label = "Principal Component 1"
            y_label = "Principal Component 2"
            z_label = "Principal Component 3"

        return reducer, title, x_label, y_label, z_label

    @staticmethod
    def _plot_embeddings(
        embeddings: np.ndarray,
        labels_: Optional[List[str]],
        color_: Optional[List[str]],
        **kwargs,
    ) -> None:
        """Helper function to plot embeddings using Plotly

        Args:
            embeddings (np.ndarray): The embeddings to plot
            labels_ (Optional[List[str]]): Labels for the embeddings
            color_ (Optional[List[str]]): Color for the embeddings
            **kwargs: Additional keyword arguments
                - title (str): The title of the plot. Defaults to "Embeddings Visualization".
                - x_label (str): The x-axis label. Defaults to "Component 1".
                - y_label (str): The y-axis label. Defaults to "Component 2".
                - z_label (str): The z-axis label. Defaults to "Component 3".
        Returns:
            None
        """

        # Set default values of the plot
        title = kwargs.get("title", "Embeddings Visualization")
        x_label = kwargs.get("x_label", "Component 1")
        y_label = kwargs.get("y_label", "Component 2")
        z_label = kwargs.get("z_label", "Component 3")

        title_font_size = max(10, 18 - len(title) // 10)  # Dynamic title size

        fig = px.scatter_3d(
            embeddings,
            x=0,
            y=1,
            z=2,
            text=labels_,
            title=title,
            labels={"0": x_label, "1": y_label, "2": z_label},
            color=color_,
        )
        fig.update_layout(
            title={
                "text": title,
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {
                    "size": title_font_size,
                    "color": "black",
                    "family": "Arial",
                    "weight": "bold",
                },
            },
            legend_title_text="Category",
            legend=dict(
                x=1,
                y=1,
                traceorder="normal",
                font=dict(family="Arial", size=12, color="black"),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.update_traces(marker={"size": 4})  # Reduced marker size
        fig.update_layout(legend_title_text="Category")
        fig.show()

    def generate_3d_visualization(
        self,
        embeddings: pd.DataFrame,
        output_file: str,
    ) -> None:
        """Generate a 3D visualization using Blender

        Args:
            embeddings (pd.DataFrame): The embeddings to visualize
            labels_ (List[str]): Labels for the embeddings
            color_ (List[str]): Color for the embeddings
            output_file (str): The path to the output file
        """
        # rename the first three columns to x, y, z
        embeddings.columns = ["x", "y", "z"] + list(embeddings.columns[3:])

        # Clear existing mesh objects in the scene
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="MESH")
        bpy.ops.object.delete()

        # Generate unique colors for each label
        unique_sectors = embeddings["sector"].unique()
        sector_colors = {
            label: (
                randint(0, 255) / 255,
                randint(0, 255) / 255,
                randint(0, 255) / 255,
                1,
            )
            for label in unique_sectors
        }

        # Function to create a sphere at a given location
        def create_sphere(location, color, radius=0.1):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
            obj = bpy.context.object
            mat = bpy.data.materials.new(name="Material")
            mat.diffuse_color = color
            obj.data.materials.append(mat)
            return obj

        # Plotting the data
        for _, row in embeddings.iterrows():
            point_location = (row["x"], row["y"], row["z"])
            sector = row["sector"]
            label = row["Label"]
            color = sector_colors.get(
                sector, (1, 1, 1, 1)
            )  # Default to white if sector not found
            _ = create_sphere(point_location, color)
            location = point_location
            text_obj = bpy.ops.object.text_add(
                location=(location[0] + 0.1, location[1], location[2])
            )
            text_obj = bpy.context.object
            text_obj.data.body = label
            text_obj.scale = (
                0.2,
                0.2,
                0.2,
            )  # Adjust the scale of the text if necessary
            text_obj.rotation_euler = (
                1.5708,
                0,
                0,
            )  # Rotate the text 90 degrees around the X-axis
            bpy.ops.object.convert(target="MESH")

        # Create a legend
        legend_x = max(embeddings["x"]) + 2  # Position legend to the right of the chart
        legend_y = max(embeddings["y"])
        legend_z = 0
        
        # Print the output file path
        print(f"Exporting to: {output_file}")

        # Export to DAE
        try:
            bpy.ops.wm.collada_export(filepath=output_file)
            print("Export successful.")
        except Exception as e:
            print(f"An error occurred during export: {e}")

        # Define materials dictionary
        materials = {
            sector: bpy.data.materials.new(name=sector) for sector in unique_sectors
        }
        for sector, color in sector_colors.items():
            materials[sector].diffuse_color = color

        for i, (category, material) in enumerate(materials.items()):
            # Create a small sphere for the legend
            legend_sphere_size = 0.2
            legend_sphere_location = (legend_x, legend_y - i * 1, legend_z)
            create_sphere(
                legend_sphere_location, material.diffuse_color, legend_sphere_size
            )

            # Create a text object for the legend
            text_obj = bpy.ops.object.text_add(
                location=(legend_x + 0.5, legend_y - i * 1, legend_z)
            )
            text_obj = bpy.context.object
            text_obj.data.body = category
            # Set the text color to match the sector color
            text_material = bpy.data.materials.new(name=f"{category}_Text")
            text_material.diffuse_color = sector_colors[category]
            if text_obj.data.materials:
                text_obj.data.materials[0] = text_material
            else:
                text_obj.data.materials.append(text_material)
            bpy.ops.object.convert(target="MESH")

        # Export to DAE
        bpy.ops.wm.collada_export(filepath=output_file)


if __name__ == "__main__":

    # Assign device to make sure GPU is used when available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU. Performance may be slower.")

    # Words List (Organized by Domain)
    words = [
        # Data Science
        "Algorithm",
        "Regression",
        "Data Scientist",
        "Feature",
        "Model",
        "Prediction",
        "Clustering",
        "Neural Network",
        "Overfitting",
        "Normalization",
        # Finance
        "Equity",
        "Dividend",
        "Derivative",
        "Arbitrage",
        "Trader",
        "Bond",
        "Portfolio",
        "Asset",
        "Interest",
        "Yield",
        # Healthcare
        "Diagnosis",
        "Therapy",
        "Immunology",
        "Doctor",
        "Biopsy",
        "Vaccination",
        "Prescription",
        "Surgery",
        "Pathology",
        "Cardiology",
        # Technology
        "Encryption",
        "Protocol",
        "Hacker",
        "Processor",
        "Network",
        "Firewall",
        "Algorithm",
        "Data Center",
        "API",
        "Cloud Computing",
        # Sports
        "Basketball",
        "Football",
        "Soccer",
        "Fencing",
        "Badminton",
        "Tennis",
        "Swimming",
        "Running",
        "Cycling",
        "Volleyball",
    ]

    # Domains List
    domains = (
        ["Data Science"] * 10
        + ["Finance"] * 10
        + ["Healthcare"] * 10
        + ["Technology"] * 10
        + ["Sports"] * 10
    )

    # Hugging Face Model examples
    # HUGGING_MODEL = "dunzhang/stella_en_400M_v5"
    HUGGING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # HUGGING_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    # HUGGING_MODEL = "facebook/bart-large"
    # HUGGING_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    # Initialize the class
    hf_embedding_viz = HuggingFaceEmbeddingViz(HUGGING_MODEL, device)

    # Generate word embeddings for stella model
    embeddings_ = hf_embedding_viz.get_model_embeddings(words)

    # Save embeddings to a file
    # hf_embedding_viz.save_embeddings(embeddings_, "embeddings.npy")

    # Load embeddings from a file
    # loaded_embeddings = hf_embedding_viz.load_embeddings("embeddings.npy")

    # # Generate PCA visualization using loaded embeddings
    # reduced_embeddings_pca_loaded = hf_embedding_viz.generate_visualization(
    #     embeddings_, labels_=words, color_=domains, method="pca", plot=False
    # )

    # # Generate TSNE visualization
    # reduced_embeddings_tsne = hf_embedding_viz.generate_visualization(
    #     embeddings_, labels_=words, color_=domains, method="tsne", plot=False
    # )

    # Generate UMAP visualization
    reduced_embeddings_umap = hf_embedding_viz.generate_visualization(
        embeddings_, labels_=words, color_=domains, method="umap", plot=False
    )

    # # Generate MDS visualization
    # reduced_embeddings_mds = hf_embedding_viz.generate_visualization(
    #     embeddings_, labels_=words, color_=domains, method="mds", plot=False
    # )

    # Generate Isomap visualization
    reduced_embeddings_isomap = hf_embedding_viz.generate_visualization(
        embeddings_,
        labels_=words,
        color_=domains,
        method="isomap",
        plot=True,
    )

    # Generate 3D visualization
    hf_embedding_viz.generate_3d_visualization(
        reduced_embeddings_umap,
        output_file="3d_visualization.dae",
    )
