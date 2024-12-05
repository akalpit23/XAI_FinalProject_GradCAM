import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import torch
from transformers import AutoModel, AutoTokenizer
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
import plotly.express as px
import base64
import collada
from collada import Collada, geometry, source, scene
import tempfile
import bpy
from random import randint

# Constants
N_COMPONENTS = 3  # For 3D visualization
PERPLEXITY = 30
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42

# Function to add background image from local file (optional)
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

# Uncomment and specify the path to your background image if desired
add_bg_from_local("/Users/akalpitdawkhar/Desktop/School/SEM_3/XAI/XAI_FinalProject_GradCAM/background.png")

# Main title
st.markdown(
    "<h1 style='text-align: center;'>🕶️ PlotVerseXR 🕶️</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center; color: #FF5733;'>Visualize 3D Data like it's meant to be seen!</h2>",
    unsafe_allow_html=True,
)

# Step 1: Model Input
st.header("Step 1: Model Input")
model_name = st.text_input(
    "Enter the Hugging Face model name or path:",
    value="sentence-transformers/all-MiniLM-L6-v2",
    max_chars=200,
)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    st.warning("CUDA not available, using CPU. Performance may be slower.")

# Load model and tokenizer
@st.cache_resource
def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model(model_name)

# Step 2: Ask for the file output path
st.header("Step 2: Output File Path")
output_file_path = st.text_input(
    "Enter the output file path for the .dae file:",
    value="/Users/akalpitdawkhar/Desktop/School/SEM_3/XAI/XAI_FinalProject_GradCAM/output.dae",
)

# Step 3: Dropdown to choose input type
st.header("Step 3: Choose Input Type")
input_type = st.selectbox(
    "Select the input type:",
    options=["Words", "Sentences", "Upload CSV"],
)

# Reduction algorithm options
reduction_algorithms = ["T-SNE", "UMAP", "PCA", "MDS", "ISOMAP"]
st.header("Step 4: Dimensionality Reduction Algorithm")
default_index = reduction_algorithms.index("UMAP")
selected_algorithm = st.selectbox(
    "Select a dimensionality reduction algorithm:",
    options=reduction_algorithms,
    index=default_index,
)

def get_reducer(algorithm_name):
    if algorithm_name == "PCA":
        reducer = PCA(n_components=N_COMPONENTS)
    elif algorithm_name == "T-SNE":
        reducer = TSNE(
            n_components=N_COMPONENTS,
            perplexity=PERPLEXITY,
            n_iter=1000,
            random_state=RANDOM_STATE,
        )
    elif algorithm_name == "UMAP":
        reducer = umap.UMAP(
            n_components=N_COMPONENTS,
            n_neighbors=N_NEIGHBORS,
            min_dist=MIN_DIST,
            random_state=RANDOM_STATE,
        )
    elif algorithm_name == "MDS":
        reducer = MDS(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    elif algorithm_name == "ISOMAP":
        reducer = Isomap(n_components=N_COMPONENTS)
    else:
        reducer = PCA(n_components=N_COMPONENTS)
    return reducer

# Define the HuggingFaceEmbeddingViz class
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

    def get_model_embeddings(self, text_list):
        """This function is used to extract embeddings for the passed text as
        defined by the LLM models embedding space

        Args:
            text_list (List[str]): The list of text to extract embeddings from

        Returns:
            np.ndarray: The embeddings of the text

        """
        embeddings = []
        max_length = self.tokenizer.model_max_length

        with torch.no_grad():
            for text in text_list:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                embeddings.append(embedding)
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
        embeddings,
        labels_=None,
        color_=None,
        method="umap",
        plot=False,
    ):
        reducer = get_reducer(method.capitalize())
        reduced_embeddings = reducer.fit_transform(embeddings)

        if plot:
            self._plot_embeddings(
                reduced_embeddings,
                labels_,
                color_,
                method=method,
            )

        reduced_embeddings_df = pd.DataFrame(
            reduced_embeddings, columns=["x", "y", "z"]
        )
        if labels_ is not None:
            reduced_embeddings_df["Label"] = labels_

        if color_ is not None:
            reduced_embeddings_df["sector"] = color_

        return reduced_embeddings_df

    @staticmethod
    def _plot_embeddings(
        embeddings,
        labels_,
        color_,
        method,
    ):
        fig = px.scatter_3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            text=labels_,
            color=color_,
            title=f"{method.upper()} Visualization",
        )
        fig.update_traces(marker={"size": 4})
        st.plotly_chart(fig)

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
            obj = bpy.context.active_object
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


# Initialize the embedding visualization class
hf_embedding_viz = HuggingFaceEmbeddingViz(model_name, device)

# Main logic based on input type
if input_type == "Words":
    st.subheader("Words Input")
    # Provide options to select multiple domains (up to 5)
    domains = [
        'Data Science', 'Finance', 'Healthcare', 'Technology', 'Sports',
        'Arts', 'Education', 'Science', 'Business', 'Environment',
        'Politics', 'Philosophy', 'Literature', 'Mathematics', 'Music',
        'History', 'Law', 'Psychology', 'Sociology', 'Engineering',
        'Agriculture', 'Astronomy', 'Biology', 'Chemistry', 'Travel',
        'Food & Culinary Arts', 'Fashion', 'Health & Wellness'
    ]
    selected_domains = st.multiselect(
        'Select up to 5 domains:',
        options=domains,
    )

    if len(selected_domains) > 5:
        st.error("You can select up to 5 domains only.")
    else:
        if st.button("Generate Embeddings and Export"):
            if not selected_domains:
                st.warning("Please select at least one domain.")
            else:
                # Predefined words for each domain
                predefined_words = {
                    'Data Science': ['Algorithm', 'Regression', 'Clustering', 'Neural Network', 'Data Mining', 'Deep Learning', 'Big Data', 'Visualization', 'Statistics', 'Predictive Modeling'],
                    'Finance': ['Stocks', 'Investment', 'Portfolio', 'Banking', 'Interest', 'Capital', 'Dividend', 'Market', 'Economy', 'Risk'],
                    'Healthcare': ['Medicine', 'Diagnosis', 'Treatment', 'Surgery', 'Therapy', 'Pharmacy', 'Doctor', 'Patient', 'Nursing', 'Wellness'],
                    'Technology': ['Software', 'Hardware', 'AI', 'Machine Learning', 'Blockchain', 'Cybersecurity', 'Cloud Computing', 'Internet', 'Programming', 'Robotics'],
                    'Sports': ['Football', 'Basketball', 'Tennis', 'Soccer', 'Baseball', 'Cricket', 'Swimming', 'Athletics', 'Gymnastics', 'Hockey'],
                    'Arts': ['Painting', 'Sculpture', 'Dance', 'Theater', 'Photography', 'Music', 'Literature', 'Film', 'Design', 'Architecture'],
                    'Education': ['Learning', 'Teaching', 'School', 'Curriculum', 'Student', 'Teacher', 'Classroom', 'Assessment', 'Research', 'Degree'],
                    'Science': ['Biology', 'Chemistry', 'Physics', 'Astronomy', 'Geology', 'Genetics', 'Ecology', 'Mathematics', 'Zoology', 'Botany'],
                    'Business': ['Management', 'Marketing', 'Sales', 'Strategy', 'Entrepreneurship', 'Leadership', 'Finance', 'Economics', 'Innovation', 'Operations'],
                    'Environment': ['Climate', 'Sustainability', 'Pollution', 'Conservation', 'Ecosystem', 'Recycling', 'Biodiversity', 'Energy', 'Habitat', 'Renewable'],
                    'Politics': ['Government', 'Policy', 'Election', 'Democracy', 'Law', 'Diplomacy', 'Legislation', 'Campaign', 'Rights', 'Justice'],
                    'Philosophy': ['Ethics', 'Logic', 'Metaphysics', 'Epistemology', 'Existentialism', 'Aesthetics', 'Morality', 'Reason', 'Consciousness', 'Ontology'],
                    'Literature': ['Novel', 'Poetry', 'Drama', 'Fiction', 'Prose', 'Narrative', 'Genre', 'Character', 'Plot', 'Theme'],
                    'Mathematics': ['Algebra', 'Calculus', 'Geometry', 'Statistics', 'Probability', 'Number Theory', 'Topology', 'Equations', 'Function', 'Logic'],
                    'Music': ['Melody', 'Harmony', 'Rhythm', 'Composition', 'Instrument', 'Genre', 'Orchestra', 'Band', 'Song', 'Performance'],
                    'History': ['Ancient', 'Medieval', 'Modern', 'Revolution', 'War', 'Civilization', 'Empire', 'Culture', 'Archaeology', 'Chronology'],
                    'Law': ['Justice', 'Legislation', 'Regulation', 'Constitution', 'Court', 'Contract', 'Rights', 'Criminal', 'Civil', 'Attorney'],
                    'Psychology': ['Behavior', 'Mind', 'Cognition', 'Emotion', 'Perception', 'Personality', 'Development', 'Therapy', 'Disorder', 'Memory'],
                    'Sociology': ['Society', 'Culture', 'Socialization', 'Institution', 'Group', 'Class', 'Inequality', 'Interaction', 'Community', 'Identity'],
                    'Engineering': ['Design', 'Construction', 'Mechanics', 'Electrical', 'Civil', 'Chemical', 'Computer', 'System', 'Material', 'Technology'],
                    'Agriculture': ['Farming', 'Crop', 'Livestock', 'Harvest', 'Soil', 'Irrigation', 'Pesticide', 'Agronomy', 'Horticulture', 'Sustainability'],
                    'Astronomy': ['Planet', 'Star', 'Galaxy', 'Universe', 'Orbit', 'Cosmology', 'Telescope', 'Asteroid', 'Comet', 'Black Hole'],
                    'Biology': ['Cell', 'Gene', 'Evolution', 'Ecology', 'Anatomy', 'Microbiology', 'Botany', 'Zoology', 'Physiology', 'Genetics'],
                    'Chemistry': ['Element', 'Compound', 'Reaction', 'Molecule', 'Atom', 'Organic', 'Inorganic', 'Analytical', 'Biochemistry', 'Periodic Table'],
                    'Travel': ['Tourism', 'Destination', 'Adventure', 'Journey', 'Flight', 'Accommodation', 'Culture', 'Exploration', 'Backpacking', 'Vacation'],
                    'Food & Culinary Arts': ['Cuisine', 'Recipe', 'Ingredient', 'Cooking', 'Flavor', 'Dish', 'Gastronomy', 'Chef', 'Baking', 'Nutrition'],
                    'Fashion': ['Style', 'Design', 'Clothing', 'Trend', 'Textile', 'Runway', 'Model', 'Apparel', 'Accessory', 'Brand'],
                    'Health & Wellness': ['Fitness', 'Nutrition', 'Exercise', 'Meditation', 'Mindfulness', 'Diet', 'Sleep', 'Stress', 'Yoga', 'Hydration'],
                }

                # Retrieve words for selected domains
                all_words = []
                domain_labels = []
                for domain in selected_domains:
                    words = predefined_words.get(domain, [])
                    all_words.extend(words)
                    domain_labels.extend([domain]*len(words))

                if all_words:
                    st.write("Words from selected domains:", all_words)
                    embeddings = hf_embedding_viz.get_model_embeddings(all_words)
                    embeddings_df = hf_embedding_viz.generate_visualization(
                        embeddings,
                        labels_=all_words,
                        color_=domain_labels,
                        method=selected_algorithm.lower(),
                        plot=True,
                    )
                    # Export to .dae
                    hf_embedding_viz.generate_3d_visualization(embeddings_df, output_file_path)
                    st.success(f"Output exported to {output_file_path}")
                else:
                    st.warning("No words were retrieved.")
elif input_type == "Sentences":
    st.subheader("Sentences Input")
    text_input = st.text_area(
        "Enter text (max 2000 characters):",
        max_chars=2000,
    )
    if st.button("Process Text and Export"):
        # Process text, extract words, assign to domains, and generate embeddings
        def process_text(text):
            # Extract words and assign to domains (simplified example)
            words = re.findall(r'\b\w+\b', text)
            words = list(set(words))  # Remove duplicates
            words.sort()
            # Assign words to domains (dummy domains for example)
            domains_list = ['Domain ' + str((i % 5) + 1) for i in range(len(words))]
            return words, domains_list

        words, domains_list = process_text(text_input)
        if words:
            st.write("Extracted Words:", words)
            embeddings = hf_embedding_viz.get_model_embeddings(words)
            embeddings_df = hf_embedding_viz.generate_visualization(
                embeddings,
                labels_=words,
                color_=domains_list,
                method=selected_algorithm.lower(),
                plot=True,
            )
            # Export to .dae
            hf_embedding_viz.generate_3d_visualization(embeddings_df, output_file_path)
            st.success(f"Output exported to {output_file_path}")
        else:
            st.warning("No words extracted from the input text.")
elif input_type == "Upload CSV":
    st.subheader("Upload CSV")

    # Initialize session state variables if they don't exist
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "text_column" not in st.session_state:
        st.session_state.text_column = None
    if "domain_column" not in st.session_state:
        st.session_state.domain_column = None

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file)
        # Store the DataFrame in session state
        st.session_state.uploaded_df = df
        st.write("Uploaded Data:", df.head())
    elif st.session_state.uploaded_df is not None:
        # Use the DataFrame from session state
        df = st.session_state.uploaded_df
        st.write("Uploaded Data (from session):", df.head())
    else:
        st.info("Awaiting CSV file upload.")
        df = None

    if df is not None:
        # Select the column containing text data
        text_column = st.selectbox(
            "Select the column containing text data:",
            options=df.columns.tolist(),
            index=0,
            key="text_column_selectbox"
        )

        # Store the selected text column in session state
        st.session_state.text_column = text_column

        # Select the column containing domain labels
        domain_column = st.selectbox(
            "Select the column containing domain labels:",
            options=df.columns.tolist(),
            index=1,
            key="domain_column_selectbox"
        )

        # Store the selected domain column in session state
        st.session_state.domain_column = domain_column

        if st.button("Process CSV and Export"):
            try:
                texts = df[st.session_state.text_column].astype(str).tolist()
                domains = df[st.session_state.domain_column].astype(str).tolist()

                embeddings = hf_embedding_viz.get_model_embeddings(texts)
                embeddings_df = hf_embedding_viz.generate_visualization(
                    embeddings,
                    labels_=texts,
                    color_=domains,  # Use domain labels for coloring
                    method=selected_algorithm.lower(),
                    plot=True,
                )
                # Export to .dae (implement your export functionality here)
                hf_embedding_viz.generate_3d_visualization(
                    embeddings_df, output_file_path
                )
                st.success(f"Output exported to {output_file_path}")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")
        