import warnings
import torch
import clip
import gradio as gr
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

# --- 1. Setup ---
print("Initializing App...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Relative path for Hugging Face
finetuned_model_path = "checkpoints/finetuned_mnist_clip.pt"

# --- 2. Load Your Fine-Tuned Model ---
print("Loading fine-tuned model...")
# Load the architecture
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

# Load weights (map_location ensures it works on CPU or GPU)
if os.path.exists(finetuned_model_path):
    model.load_state_dict(torch.load(finetuned_model_path, map_location="cpu"))
    print("Model weights loaded successfully.")
else:
    print("WARNING: Model file not found! Loading default CLIP weights.")

model = model.to(device)
model.eval()

# --- 3. Load MNIST Test Data ---
print("Loading MNIST Test Data...")
# This will download the data to the local folder in the HF Space
mnist_test_transformed = MNIST(root="./data", train=False, download=True, transform=preprocess)
original_mnist_test = MNIST(root="./data", train=False, download=True)

# --- 4. Generate Embeddings on Startup ---
print("Generating embeddings for the 10,000 test images (this may take a minute)...")
batch_size = 128
data_loader = DataLoader(mnist_test_transformed, batch_size=batch_size, shuffle=False)

all_embeddings_list = []
all_labels_list = []

with torch.no_grad():
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        batch_embeddings = model.encode_image(images)
        all_embeddings_list.append(batch_embeddings.cpu())
        all_labels_list.append(labels.cpu())

all_features_cpu = torch.cat(all_embeddings_list)
all_labels_cpu = torch.cat(all_labels_list)
all_features_gpu = all_features_cpu.to(device)
print("Embeddings generated.")

# --- 5. Pre-compute UMAP ---
print("Fitting UMAP...")
embeddings_np = all_features_cpu.numpy()
labels_np = all_labels_cpu.numpy()

reducer = umap.UMAP(
    n_neighbors=15, min_dist=0.1, n_components=2,
    metric='cosine', random_state=42, verbose=False
)
umap_embeddings_2d = reducer.fit_transform(embeddings_np)
print("UMAP Ready.")

# --- 6. Helper Functions ---

def calculate_similarity(features1, features2):
    features1 = features1 / features1.norm(dim=-1, keepdim=True)
    features2 = features2 / features2.norm(dim=-1, keepdim=True)
    return features1 @ features2.T


def preprocess_sketch(sketch_input_data):
    sketch_array = None
    if sketch_input_data is None: return None
    if isinstance(sketch_input_data, dict):
        if 'image' in sketch_input_data and isinstance(sketch_input_data['image'], np.ndarray):
             sketch_array = sketch_input_data['image']
        elif 'composite' in sketch_input_data and isinstance(sketch_input_data['composite'], np.ndarray):
             sketch_array = sketch_input_data['composite']
    elif isinstance(sketch_input_data, np.ndarray):
        sketch_array = sketch_input_data
    
    if sketch_array is None or sketch_array.size == 0: return None

    try:
        pil_image = Image.fromarray(sketch_array).convert("L")
    except Exception: return None

    pil_image = ImageOps.invert(pil_image)
    bbox = pil_image.getbbox()
    if bbox is None: return Image.new('L', (28, 28), color=0)
    pil_image = pil_image.crop(bbox)
    
    width, height = pil_image.size
    target_size = max(width, height)
    padding = ((target_size - width) // 2, (target_size - height) // 2,
               target_size - width - (target_size - width) // 2,
               target_size - height - (target_size - height) // 2)
    pil_image = ImageOps.expand(pil_image, padding, fill=0)
    pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
    return pil_image

# --- Search Functions ---

def text_search_and_plot(digit_text, top_k=3):
    if not digit_text: return [], None
    
    digit_map = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }
    digit_word = digit_map.get(digit_text, "")
    if not digit_word: return [], None

    prompt = f"A handwritten digit {digit_word}"
    text_token = clip.tokenize([prompt]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_token)

    similarities = calculate_similarity(text_features, all_features_gpu)
    _, top_indices = torch.topk(similarities.squeeze(0), top_k)
    top_indices = top_indices.cpu().numpy().tolist()
    
    result_images = [original_mnist_test[idx][0] for idx in top_indices]

    # Plot
    text_features_cpu = text_features.cpu().numpy()
    text_embedding_2d = reducer.transform(text_features_cpu)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.scatterplot(x=umap_embeddings_2d[:, 0], y=umap_embeddings_2d[:, 1],
                    hue=labels_np, palette=sns.color_palette("tab10", 10),
                    s=10, alpha=0.6, ax=ax)
    ax.scatter(text_embedding_2d[0, 0], text_embedding_2d[0, 1],
               marker='s', color='black', s=100, label=f'Text: "{prompt}"')
    
    ax.set_title('UMAP Projection (Image Embeddings + Text Query)')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    dot_marker_size_scale = 6
    for i in range(min(10, len(handles) -1)):
         if hasattr(handles[i], 'set_sizes'):
             current_size = handles[i].get_sizes()[0]
             handles[i].set_sizes([current_size * dot_marker_size_scale])
    ax.legend(handles, labels, title='Digit Label / Text', loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.close(fig)

    return result_images, fig

def sketch_search_and_plot(sketch_image, top_k=3):
    if sketch_image is None: return [], None
    
    preprocessed_sketch_pil = preprocess_sketch(sketch_image)
    if preprocessed_sketch_pil is None: return [], None

    clip_input_tensor = preprocess(preprocessed_sketch_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        sketch_features = model.encode_image(clip_input_tensor)

    similarities = calculate_similarity(sketch_features, all_features_gpu)
    _, top_indices = torch.topk(similarities.squeeze(0), top_k)
    top_indices = top_indices.cpu().numpy().tolist()

    result_images = [original_mnist_test[idx][0] for idx in top_indices]

    # Plot
    try:
        sketch_features_cpu = sketch_features.cpu().numpy()
        sketch_embedding_2d = reducer.transform(sketch_features_cpu)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sns.scatterplot(x=umap_embeddings_2d[:, 0], y=umap_embeddings_2d[:, 1],
                        hue=labels_np, palette=sns.color_palette("tab10", 10),
                        s=10, alpha=0.6, ax=ax)
        ax.scatter(sketch_embedding_2d[0, 0], sketch_embedding_2d[0, 1],
                   marker='s', color='black', s=100, label='Your Sketch')
        ax.set_title('UMAP Projection (Image Embeddings + Sketch Query)')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        handles, labels = ax.get_legend_handles_labels()
        dot_marker_size_scale = 6
        for i in range(min(10, len(handles) -1)):
             if hasattr(handles[i], 'set_sizes'):
                 current_size = handles[i].get_sizes()[0]
                 handles[i].set_sizes([current_size * dot_marker_size_scale])
        ax.legend(handles, labels, title='Digit Label / Sketch', loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.close(fig)
        plot_output = fig
    except Exception: plot_output = None

    return result_images, plot_output

# --- 7. Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as app_interface:
    gr.Markdown(
        """
        # CLIP MNIST Search Engine 🧠🖼️✏️
        Use CLIP embeddings to search the MNIST dataset using text prompts or sketches.
        """
    )

    with gr.Tabs():
        # --- Text Search Tab ---
        with gr.TabItem("Text Search"):
            with gr.Row():
                # --- Left Column (Inputs & Results) ---
                with gr.Column():
                    gr.Markdown("### 1. Select a Digit")
                    text_input = gr.Radio(
                        choices=[str(i) for i in range(10)],
                        label="Select a Digit",
                        info="Choose the digit you want to search for."
                    )
                    submit_btn_text = gr.Button("Search Text", variant="primary")
                    
                    gr.Markdown("### 2. Search Results")
                    gallery_text = gr.Gallery(
                        label="Top 3 Matches", columns=3, object_fit="contain",
                        height=300, preview=True
                    )
                
                # --- Right Column (Visualization) ---
                with gr.Column():
                    gr.Markdown("### Live UMAP Visualization")
                    plot_text = gr.Plot(
                        label="Embedding Space"
                    )

        # --- Sketch Search Tab ---
        with gr.TabItem("Sketch Search"):
            with gr.Row():
                # --- Left Column (Inputs & Results) ---
                with gr.Column():
                    gr.Markdown("### 1. Draw a Digit")
                    sketch_input = gr.Sketchpad(
                        label="Sketchpad",
                        type="numpy",
                        height=300
                     )
                    submit_btn_sketch = gr.Button("Search Sketch", variant="primary")
                    
                    gr.Markdown("### 2. Search Results")
                    gallery_sketch = gr.Gallery(
                        label="Top 3 Matches", columns=3, object_fit="contain",
                        height=300, preview=True
                    )

                # --- Right Column (Visualization) ---
                with gr.Column():
                    gr.Markdown("### Live UMAP Visualization")
                    plot_sketch = gr.Plot(
                        label="Embedding Space"
                    )

    # --- Event Handlers ---
    submit_btn_text.click(
        fn=text_search_and_plot,
        inputs=text_input,
        outputs=[gallery_text, plot_text],
        show_progress="full"
    )

    submit_btn_sketch.click(
        fn=sketch_search_and_plot,
        inputs=sketch_input,
        outputs=[gallery_sketch, plot_sketch],
        show_progress="full"
    )
    
# Launch the app
app_interface.launch()