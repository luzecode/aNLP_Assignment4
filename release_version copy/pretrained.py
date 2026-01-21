import os
import urllib.request
import json
from model.model_gpt import GPTModel
import tiktoken
import torch
from train import generate_and_print_sample

def load_pretrained_gpt(start_context):

    """
    TODO Complete this function to load pretrained GPT-2 weights.
    
    Steps to implement:
    1. Set up the device (CPU or CUDA)
    2. Define the file name and download URL for the pretrained weights
    3. Download the pretrained model from HuggingFace (handle potential download errors)
    4. Define the correct GPT_CONFIG for the pretrained model
    5. Create a GPTModel instance with this config
    6. Load the state dictionary into the model (use map_location for device compatibility)
    7. Set the model to evaluation mode
    8. Move the model to the appropriate device
    9. Use generate_and_print_sample() to generate text
    
    Hints:
    - Use urllib.request.urlretrieve() to download the file
    - Use os.path.exists() to check if file exists
    - Use try-except blocks to handle errors
    - The HuggingFace URL format is:
      "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"
    
    Important:
    - The pretrained model configuration MUST match the saved weights!
    - context_length: 1024 (not 256)
    - qkv_bias: True (not False)
    - Otherwise you'll get shape mismatch errors
    """
    # 1. Set up the device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PRETRAINED] Using device: {device}")
    
    # 2. Define the file name and download URL for the pretrained weights
    file_name = "gpt2-small-124M.pth"
    base_url = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main"
    download_url = f"{base_url}/{file_name}"
    
    # 3. Download the pretrained model from HuggingFace (handle potential download errors)
    if not os.path.exists(file_name):
        print(f"[PRETRAINED] Downloading {file_name} from HuggingFace...")
        try:
            urllib.request.urlretrieve(download_url, file_name)
            print(f"[PRETRAINED] Download complete: {file_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to download pretrained model: {e}")
    else:
        print(f"[PRETRAINED] Found existing file: {file_name}")
    
    # 4. Define the correct GPT_CONFIG for the pretrained model
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
             # Get the pretrained config from JSON
    GPT_CONFIG_PRETRAINED = config_data.get("GPT_CONFIG_PRETRAINED")
    # 5. Create a GPTModel instance with this config
    model = GPTModel(GPT_CONFIG_PRETRAINED)
    
    # 6. Load the state dictionary into the model (use map_location for device compatibility)
    try:
        state_dict = torch.load(file_name, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict: {e}")

    # 7. Set the model to evaluation mode
    model.eval()
    
    # 8. Move the model to the appropriate device
    model.to(device)
    
    # 9. Use generate_and_print_sample() to generate text
    tokenizer = tiktoken.get_encoding("gpt2")
    
    generate_and_print_sample(
        model=model,
        tokenizer=tokenizer,
        device=device,
        start_context=start_context
    )
    return model