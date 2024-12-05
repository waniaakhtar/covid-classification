import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import base64

image_path = ''

# Load the pre-trained model (you'll need to replace 'trained_model.pt' with the actual path to your model file)
model = torch.load('trained_model.pt', map_location="cpu")
model.eval()

# Define class names for the possible image classifications
class_names = ["infected", "normal"]

# Define a series of image transformations to be applied to each loaded image (same as done for training images)
loader = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Ensure image has 3 channels
    transforms.Normalize([0.252, 0.293, 0.288], [0.146, 0.191, 0.193])
])

# Define a function to load an image, preprocess it, and convert it to a tensor
def image_loader(image):
    """Load an image and return it as a CUDA tensor (assumes GPU usage)"""
    image = Image.open(image)  # Open the image file
    image = loader(image).float()  # Apply the defined transformations to the image
    image = Variable(image, requires_grad=True)  # Create a PyTorch variable with gradients enabled
    image = image.unsqueeze(0)  # Reshape the image tensor (not necessary for ResNet)
    return image  # Return the preprocessed image tensor

# Function to check if the image looks like an X-ray based on model confidence or heuristic
def check_if_xray(output):
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_index = torch.max(probabilities, 1)
    
    # Here we assume that if confidence is below a certain threshold, the image is unexpected
    if confidence.item() < 0.6:  # Threshold can be adjusted
        return False
    return True

st.markdown("""
  <style>
    .css-o18uir.e16nr0p33 {
      margin-top: -50px;
    }
    .centered-content {
      display: flex;
      justify-content: center;
      align-items: center;
      flex-direction: column;
    }
    .centered-image img {
      display: block;
      margin-left: auto;
      margin-right: auto;
      height: 350px !important;  /* Set the fixed height */
      object-fit: contain;  /* Maintain aspect ratio */
      padding-bottom: 20px; /* Add padding below the image */
    }
    .stButton {
      text-align: center;
      margin: auto;
    }
  </style>
""", unsafe_allow_html=True)

# Create a Streamlit web app
st.title("COVID-19 Classifier")

# Upload an image
uploaded_image = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Center the image display
    st.markdown("<div class='centered-content'>", unsafe_allow_html=True)
    
    # Convert image to base64
    encoded_image = base64.b64encode(uploaded_image.getvalue()).decode()

    # Display the uploaded image with a fixed height and centered alignment
    st.markdown(f"<div class='centered-image'><img src='data:image/png;base64,{encoded_image}'/></div>", unsafe_allow_html=True)

    # Center the "Classify" button
    if st.button("Classify",):
        image_path = "temp.jpg"  # Temporary image path

        # Save the uploaded image to a temporary file
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        try:
            # Load and preprocess the uploaded image
            image = image_loader(image_path)

            # Perform a prediction using the loaded model
            output = model(image)

            # Check if the image is likely to be an X-ray
            if not check_if_xray(output):
                st.markdown(
                    "<p style='color:orange;font-size:25px;text-align:center;'>Unexpected Image: It does not look like an X-Ray</p>",
                    unsafe_allow_html=True,
                )
            else:
                index = output.data.cpu().numpy().argmax()
                predicted_class = class_names[index]

                # Determine the color for the classification label (red for infected, green for normal)
                color = "red" if predicted_class == "infected" else "lime"

                # Display the result in text form with the determined color
                st.markdown(
                    f"<p style='color:{color};font-size:25px;text-align:center;'>Classified as {predicted_class}</p>",
                    unsafe_allow_html=True,
                )

        except RuntimeError as e:
            if "size of tensor a" in str(e):
                st.markdown(
                    "<p style='color:orange;font-size:25px;text-align:center;'>Unexpected Image: It does not look like an X-Ray</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"An unexpected error occurred: {str(e)}")

# Clean up the temporary image file
if os.path.exists(image_path):
    os.remove(image_path)
