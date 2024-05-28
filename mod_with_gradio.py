import gradio as gr
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionXLPipeline

# Load the model
pipe_xl = StableDiffusionXLPipeline.from_pretrained("segmind/Segmind-Vega", variant="fp16")
pipe_xl.to("cpu")

# Function to resize an image
def resize_image(image_path_or_data, width=600, height=600):
    try:
        if isinstance(image_path_or_data, str):
            img = Image.open(image_path_or_data)
        else:
            img = Image.fromarray(image_path_or_data)

        resized_img = img.resize((width, height))
        return resized_img
    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        return None

# Function to extract information from Excel and generate a prompt
def extract_info_from_excel(primary_key):
    try:
        excel_file_path = "F:\\dr\\Emp.xlsx"
        df = pd.read_excel(excel_file_path)

        if primary_key in df['ID'].values:
            selected_data = df[df['ID'] == primary_key].squeeze()

            output_prompt = (
                f"Make a greeting card for {selected_data['NAME']} who likes the season {selected_data['SEASON']}, "
                f"likes to visit {selected_data['TRAVEL']}," 
                f" favourite colour is {selected_data['COLOUR']}, "
                f"likes to eat {selected_data['FOOD']} food," 
                f" likes flower {selected_data['FLOWER']}, "
                f"listens to {selected_data['MUSIC']} music, "
                f"and likes to do {selected_data['ACTIVITY']}."
            )

            # Model generates an image based on the prompt
            generated_image = pipe_xl(prompt=output_prompt, negative_prompt="(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch)").images[0]

            # Resize the generated image
            # resized_generated_image = resize_image(generated_image)

            return output_prompt, generated_image
        else:
            return f"No data found for ID {primary_key}", None
    except Exception as e:
        return f"Error: {str(e)}", None

# Create a Gradio interface
iface = gr.Interface(
    fn=extract_info_from_excel,
    inputs=["number"],  # Employee ID as number
    outputs=["text", "image"],  # Two output sections: text and image
    live=True,
    title="Employee Information Extractor",
    description="Enter Employee ID to extract information and generate a greeting card image."
)

# Launch the Gradio interface
iface.launch()
