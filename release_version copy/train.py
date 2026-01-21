"""
Supplementary code for the "Build a Large Language Model From Scratch" book
by Sebastian Raschka.

Book link: http://mng.bz/orYv
Code repository: https://github.com/rasbt/LLMs-from-scratch

© Sebastian Raschka. All rights reserved.
© Modifications for ANLP assignment: TA team ANLP2025

This code is provided for educational purposes and may be used, modified,
and distributed for non-commercial purposes, provided that proper attribution
to the book and author is included.
"""

import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken
import math
import json

import argparse


from model.model_gpt import GPTModel
from model.utils import create_dataloader_v1, generate_text_simple, decode_1, decode_2


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  
    model.train()
    


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, clipping=False):
    """
    TODO: 
        Implement clipping/warm-up rate. You may modify this function as you see fit.
    """
    
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """ 
    TODO: Implement a function that plots training and validation loss.

    Requirements:
    - Create a figure with matplotlib.
    - Plot train and validation loss against the number of epochs.
    - Add axis labels and a legend.
    - Add a second x-axis showing the number of tokens seen.
    - Save the figure as "loss-plot.png" and display it.

    Hints:
    - Use plt.subplots() to create a figure and an axis.
    - Use ax.plot(...) to plot lines.
    - Use ax.twiny() to create a second x-axis on top.
    - Use fig.tight_layout() before saving.
    - Use plt.savefig(...) and plt.show() at the end.
    """
     # Create figure and primary axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training and validation losses against epochs_seen
    ax.plot(epochs_seen, train_losses, label="Training loss", linewidth=2)
    ax.plot(epochs_seen, val_losses, label="Validation loss", linewidth=2)
    
    # Add labels and legend
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create second x-axis on top showing tokens seen
    ax2 = ax.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot to sync axes
    ax2.set_xlabel("Tokens seen")
    
    # Adjust layout and save
    fig.tight_layout()
    plt.savefig("loss-plot.png")
    plt.show()
    

