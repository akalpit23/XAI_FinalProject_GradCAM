import streamlit as st
from openai import OpenAI
import plotly.graph_objects as go
import numpy as np
import json
import pandas as pd
import re
import bpy

import base64


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
add_bg_from_local("newbg.jpeg")


client = OpenAI(
    # This is the default and can be omitted
    api_key="OPEN_AI_KEY",
)


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

# Step 1: Upload CSV File
# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")


def process_file(file):
    """Process the uploaded CSV file and display its content."""
    df = None  # Initialize df to None
    try:
        if file is not None:
            df = pd.read_csv(file)
            st.write(
                "Data preview:", df.head()
            )  # Display the first few rows of the data
        else:
            st.warning("No file uploaded.")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

    return df


# Call the processing function only if a file is uploaded
if uploaded_file is not None:
    df = process_file(uploaded_file)
    if df is not None:
        print()
        # st.write("Full DataFrame:", df)
    else:
        st.warning("Failed to process the file.")


user_input = st.text_input("Enter a description of the plot you want to create:")


@st.cache_data
def generate_plotly_code(description, df):
    # Extract column names from the DataFrame for context
    column_names = df.columns.tolist()
    columns_info = ", ".join(column_names)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an assistant that generates 3d blender code based on user description. 
                Here is some example code.
                
                import bpy
import pandas as pd
import bpy

# import bmesh
from mathutils import Vector
import pandas as pd
from random import randint

# Assuming 'df' is already loaded with the required data
df = pd.read_csv(
    r"C:\\Users\\rakee\\Downloads\\embeddings_umap.csv"
)  # Normally you would load your data here


# Clear existing mesh objects in the scene
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.select_by_type(type="MESH")
bpy.ops.object.delete()

# Define colors for sectors (you might need to adjust this based on actual sectors)

# Generate unique colors for each label
unique_sectors = df["sector"].unique()
sector_colors = {
    label: (randint(0, 255) / 255, randint(0, 255) / 255, randint(0, 255) / 255, 1)
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
for index, row in df.iterrows():
    point_location = (row[0], row[1], row[2])

    sector = row["sector"]
    label = row["Label"]
    color = sector_colors.get(
        sector, (1, 1, 1, 1)
    )  # Default to white if sector not found

    sphere = create_sphere(point_location, color)

    location = point_location
    text_obj = bpy.ops.object.text_add(
        location=(location[0] + 0.1, location[1], location[2])
    )
    text_obj = bpy.context.object
    text_obj.data.body = label
    text_obj.scale = (0.2, 0.2, 0.2)  # Adjust the scale of the text if necessary
    text_obj.rotation_euler = (
        1.5708,
        0,
        0,
    )  # Rotate the text 90 degrees around the X-axis
    bpy.ops.object.convert(target="MESH")

# Create a legend
legend_x = max(df["0"]) + 2  # Position legend to the right of the chart
legend_y = max(df["1"])
legend_z = 0

# Define materials dictionary
materials = {sector: bpy.data.materials.new(name=sector) for sector in unique_sectors}
for sector, color in sector_colors.items():
    materials[sector].diffuse_color = color

for i, (category, material) in enumerate(materials.items()):
    # Create a small sphere for the legend
    legend_sphere_size = 0.2
    legend_sphere_location = (legend_x, legend_y - i * 0.5, legend_z)
    # create_sphere(legend_sphere_location, material.diffuse_color, legend_sphere_size)

    # Create a text object for the legend
    text_obj = bpy.ops.object.text_add(
        location=(legend_x + 0.5, legend_y - i * 0.5, legend_z)
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
bpy.ops.wm.collada_export(filepath="output.dae")""",
            },
            {
                "role": "user",
                "content": (
                    f"Write blender friendly code to generate 3d plot based on the data and user description:\n\n"
                    f"{description}\n\n"
                    # f"Only use df for data and dont create random data.\n"
                    f"The available data is in the dataframe named df.\n"
                    f"The available data columns are: {columns_info}.\n"
                ),
            },
        ],
        model="gpt-4o",
        max_tokens=1500,
        temperature=0,
    )

    code = chat_completion.choices[0].message.content.strip()
    return code


@st.cache_data
def update_plotly_code(generated_code, adjustment_prompt):
    """
    Update the 3D blender friendly code based on the user-provided adjustment prompt.

    Parameters:
    - generated_code (str): The original  3D blender code to be modified.
    - adjustment_prompt (str): The user's request for modifications.

    Returns:
    - str: The updated 3D blender code.
    """
    # Generate Updated Plot Code
    update_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that modifies 3D blender code based on user requests.",
            },
            {
                "role": "user",
                "content": f"Modify the following code to {adjustment_prompt}.\n\n"
                f"Original code:\n{generated_code}",
            },
        ],
        max_tokens=1500,
        temperature=0,
    )
    updated_code = update_response.choices[0].message.content.strip()
    return updated_code


def extract_code(script):
    # Use regex to match code blocks within triple backticks
    code_blocks = re.findall(r"```python(.*?)```", script, re.DOTALL)
    return "".join(code_blocks).strip()


def export_scene_to_dae(filepath):
    # Ensure the file has a .dae extension
    if not filepath.lower().endswith(".dae"):
        filepath += ".dae"

    # Export the scene to DAE format
    bpy.ops.wm.collada_export(
        filepath=filepath,
        check_existing=True,
        filter_glob="*.dae",
        apply_modifiers=True,
        selected=False,
        include_children=True,
        include_armatures=True,
        include_shapekeys=True,
        use_texture_copies=True,
    )

    print(f"Exported to: {filepath}")


# Set the export path (change this to your desired location)
# export_path = bpy.path.abspath("untitled.dae")

# if user_input:
#     ########## First ##########
#     code = generate_plotly_code(user_input, df)
#     st.subheader("Generated Plotly Code")
#     code  = extract_code(code)
#     imports_end = re.search(r'^(?!import).*$', code, re.MULTILINE)
#     if imports_end:
#         insert_position = imports_end.start()
#         new_lines = "import csv\n\ndf = pd.read_csv('filename.csv')\n"
#         modified_code = code[:insert_position] + new_lines + code[insert_position:]


#     ########## Second ##########
#     adjustment_prompt = st.text_input("Tell what changes you want to make to the plot:")
#     print(adjustment_prompt)
#     updated_code = update_plotly_code(code, adjustment_prompt)
#     #print("sexesesxsexesesesexsexsexesxexsxesexsexsexsex")
#     # print(updated_code)
#     print(extract_code(updated_code))
#     file_reader = "pd.read_csv('3d_points.csv')"
#     code = extract_code(updated_code)
#     imports_end = re.search(r'^(?!import).*$', code, re.MULTILINE)
#     if imports_end:
#         insert_position = imports_end.start()
#         new_lines = "import csv\n\ndf = pd.read_csv('filename.csv')\n"
#         modified_code = code[:insert_position] + new_lines + code[insert_position:]


#     # exec(code)
#     # export_scene_to_dae(export_path)
#     st.subheader("Updated Plotly Code")
#     st.code(updated_code, language="python")

import re
import streamlit as st
import pandas as pd


def add_imports_and_df(code, filename):
    imports_end = re.search(r"^(?!import).*$", code, re.MULTILINE)
    if imports_end:
        insert_position = imports_end.start()
        new_lines = (
            f"import csv\n\nimport pandas as pd\n\ndf = pd.read_csv('{filename}')\n"
        )
        return code[:insert_position] + new_lines + code[insert_position:]
    return code


def execute_code(code):
    st.code(code)
    try:
        exec(code)
    except Exception as e:
        st.error(f"Error executing code: {str(e)}")


if user_input:
    ########## First ##########
    initial_code = generate_plotly_code(user_input, df)
    st.subheader("Generated Plotly Code")
    initial_code = extract_code(initial_code)
    modified_initial_code = add_imports_and_df(initial_code, "embeddings_umap.csv")

    export_code = """
# Set export path
export_path = bpy.path.abspath("umap.dae")

def export_scene_to_dae(filepath):
    # Ensure the file has a .dae extension
    if not filepath.lower().endswith(".dae"):
        filepath += ".dae"

    # Export the scene to DAE format
    bpy.ops.wm.collada_export(
        filepath=filepath,
        check_existing=True,
        filter_glob="*.dae",
        apply_modifiers=True,
        selected=False,
        include_children=True,
        include_armatures=True,
        include_shapekeys=True,
        use_texture_copies=True,
    )

    print(f"Exported to: {filepath}")

# Export scene
export_scene_to_dae(export_path)
"""
    modified_initial_code += export_code
    with st.expander("View Generated Code"):
        st.code(modified_initial_code, language="python")

    # add the modified code to a python file and save it
    with open("new.py", "w") as f:
        f.write(modified_initial_code)

    import subprocess

    # Run the other Python file
    try:
        result = subprocess.run(
            ["python", "new.py"], capture_output=True, text=True, check=True
        )

        # Print the standard output
        st.write("Standard Output:")
        st.write(result.stdout)

        # Print the standard error
        st.write("Standard Error:")
        st.write(result.stderr)

        # Print the return code
        st.write("Return Code:", result.returncode)

    except subprocess.CalledProcessError as e:
        st.write("An error occurred while running the subprocess:")
        st.write("Standard Output:", e.stdout)
        st.write("Standard Error:", e.stderr)
        st.write("Return Code:", e.returncode)

    # subprocess.run(["python", "Desktop/exporter.py"])
    # export_path = bpy.path.abspath("12345.dae")
    # export_scene_to_dae(export_path)

    ########## Second ##########
    adjustment_prompt = st.text_input("Tell what changes you want to make to the plot:")
    if adjustment_prompt:
        updated_code = update_plotly_code(initial_code, adjustment_prompt)
        updated_code = extract_code(updated_code)
        modified_updated_code = add_imports_and_df(updated_code, "embeddings_umap.csv")

        st.subheader("Updated Plotly Code")
        st.code(modified_updated_code, language="python")

    # add the modified code to a python file and save it
    with open("new.py", "w") as f:
        f.write(modified_initial_code)

    import subprocess

    # Run the other Python file
    try:
        result = subprocess.run(
            ["python", "new.py"], capture_output=True, text=True, check=True
        )

        # Print the standard output
        st.write("Standard Output:")
        st.write(result.stdout)

        # Print the standard error
        st.write("Standard Error:")
        st.write(result.stderr)

        # Print the return code
        st.write("Return Code:", result.returncode)

    except subprocess.CalledProcessError as e:
        st.write("An error occurred while running the subprocess:")
        st.write("Standard Output:", e.stdout)
        st.write("Standard Error:", e.stderr)
        st.write("Return Code:", e.returncode)
