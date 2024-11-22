import streamlit as st
import numpy as np
from PIL import Image

# Debugging: Try importing YOLO from ultralytics
try:
    from ultralytics import YOLO
    import torch
    st.write("Ultralytics YOLO imported successfully!")
except ImportError as e:
    st.error(f"Failed to import YOLO. Error: {e}")
    raise

# Page Title
st.title("YOLO Object Detection Debug Mode")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Load the image using PIL
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load image. Error: {e}")
        raise

    # Initialize YOLO model
    try:
        model_path = 'yolov8n.pt'  # Update with your model path if different
        model = YOLO(model_path)
        st.write("YOLO model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load YOLO model. Error: {e}")
        raise

    # Check device availability (GPU/CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"Using device: {device}")
    model.to(device)

    try:
        # Perform inference
        results = model(img)

        # Plot annotated image
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Annotated Image", use_container_width=True)

        # Log detected objects
        st.subheader("Detected Objects:")
        for box in results[0].boxes:
            try:
                # Extract class and confidence
                cls = int(box.cls.cpu().numpy())  # Convert to integer
                conf = float(box.conf.cpu().numpy())  # Convert to float
                st.write(f"Class: {cls}, Confidence: {conf:.2f}")
            except Exception as e:
                st.error(f"Failed to parse bounding box data. Error: {e}")

    except Exception as e:
        st.error(f"Error during inference. Error: {e}")
        raise

else:
    st.info("Please upload an image to begin detection.")






