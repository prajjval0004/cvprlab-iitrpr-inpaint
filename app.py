import streamlit as st
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import io

# Import the model class from the repository's file
try:
    from model_directional_query_od import Inpainting
except ImportError:
    st.error("Could not import 'Inpainting' from 'model_directional_query_od.py'. Ensure the file is in the same directory.")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Blind Omni-Wav Net Inpainting",
    page_icon="🖼️",
    layout="wide"
)

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(model_path):
    """
    Initializes the Inpainting model and loads weights.
    Cached to prevent reloading on every UI interaction.
    """
    # Architecture parameters as defined in the repository's test.py
    model = Inpainting(
        num_blocks=[4, 6, 6, 8], 
        num_heads=[1, 2, 4, 8], 
        channels=[48//3, 96//3, 192//3, 384//3], 
        num_refinement=4, 
        expansion_factor=2.66
    )
    
    try:
        # Load state dict (handles CPU/GPU mapping)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        return None  # Handle error in main loop to show UI message

def preprocess_image(image):
    """
    1. Converts PIL image to Tensor [0, 1].
    2. Pads the image to be a multiple of 32.
    """
    image = image.convert('RGB')
    w, h = image.size
    
    # ToTensor converts [0, 255] -> [0.0, 1.0]
    x = T.ToTensor()(image).unsqueeze(0).to(DEVICE)
    
    # Calculate padding to make dimensions multiples of 32
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
    return x, h, w

def postprocess_output(out_tensor, h, w):
    """
    1. Crops the output back to original dimensions.
    2. Clamps values to [0, 1].
    3. Converts to PIL Image.
    """
    out_cropped = out_tensor[:, :, :h, :w]
    out_clamped = torch.clamp(out_cropped, 0, 1)
    out_byte = out_clamped.mul(255).clamp(0, 255).byte()
    out_np = out_byte.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(out_np)

# --- Main UI ---

st.title("Blind Omni-Wav Net Inpainting")
st.markdown("""
This app runs the **Blind Image Inpainting** model via Omni-dimensional Gated Attention.
Upload a corrupted image to restore it.
""")

# --- Sidebar: Model Selection ---
with st.sidebar:
    st.header("Model Settings")
    
    # Define available checkpoints map
    CHECKPOINTS = {
        "CelebA-HQ (Faces)": "celeb.pth",
        "Paris Street View": "paris.pth",
        "ImageNet": "imagenet.pth",
        "Places2": "place.pth"
    }
    
    # Dropdown for selection
    selected_model_name = st.selectbox("Select Model Dataset", list(CHECKPOINTS.keys()))
    
    # Allow user to specify directory or use default 'checkpoints/'
    base_dir = st.text_input("Checkpoints Directory", value="checkpoints")
    
    # Construct full path
    filename = CHECKPOINTS[selected_model_name]
    full_path = os.path.join(base_dir, filename)
    
    st.caption(f"Looking for: `{full_path}`")
    st.info(f"Using Device: **{DEVICE}**")

# --- Load Model Logic ---
model = None
if not os.path.exists(full_path):
    st.warning(f"⚠️ Checkpoint file not found: `{full_path}`")
    st.markdown(f"**Action Required:** Please download `{filename}` and place it in the `{base_dir}/` folder.")
else:
    with st.spinner(f"Loading {selected_model_name} model..."):
        model = load_model(full_path)
    
    if model is None:
        st.error(f"Failed to load weights from `{full_path}`. The file might be corrupted.")
    else:
        st.success(f"✅ Loaded: {selected_model_name}")

# --- Inference UI ---
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    # Display Original
    original_pil = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original / Corrupted")
        # Legacy parameter for Python 3.6 Streamlit
        st.image(original_pil, use_column_width=True)

    # Run Inference
    if st.button("Restoration Start"):
        with st.spinner("Running Inference..."):
            try:
                # Preprocess
                input_tensor, orig_h, orig_w = preprocess_image(original_pil)
                
                # Inference
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                
                # Postprocess
                result_pil = postprocess_output(output_tensor, orig_h, orig_w)
                
                # Display Result
                with col2:
                    st.subheader("Restored Output")
                    st.image(result_pil, use_column_width=True)
                
                # Download Button
                st.markdown("---")
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Result",
                    data=byte_im,
                    file_name=f"restored_{uploaded_file.name}",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
                st.write("Tip: Ensure the input image isn't too large for your GPU memory.")