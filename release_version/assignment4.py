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
import tiktoken
import json

import argparse

from model.model_gpt import GPTModel
from model.utils import create_dataloader_v1, generate_text_simple, decode_1, decode_2
from pretrained import load_pretrained_gpt

from train import (
    text_to_token_ids,
    token_ids_to_text,
    train_model,
    plot_losses,
)


def main(gpt_config, settings, train_text_path: str = "./data/frankenstein.txt"):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tiktoken.get_encoding("gpt2")

    with open(train_text_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    ##############################
    # Set up dataloaders
    ##############################

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    ##############################
    # Train model
    ##############################

    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="My beloved Sister,",
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 4 CLI")

    # ACTION: train, decode, perplexity, pretrained-model
    parser.add_argument(
        "action",
        choices=["train", "decode", "perplexity", "pretrained-model"],
        help="Which action to run.",
    )

    # Positional arguments whose meaning depends on action:
    #   train:          arg1 = optional train_text_path
    #   decode:         arg1 = model_number, arg2 = decoder_name, arg3 = optional start_text
    #   perplexity:     arg1 = model_number, arg2 = test_text_path
    #   pretrained-model: no extra args
    parser.add_argument("arg1", nargs="?", help="Meaning depends on action")
    parser.add_argument("arg2", nargs="?", help="Meaning depends on action")
    parser.add_argument("arg3", nargs="?", help="Meaning depends on action")

    args = parser.parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)

    GPT_CONFIG = config["GPT_CONFIG"]
    OTHER_SETTINGS = config["OTHER_SETTINGS"]
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################################################################
    # ACTION 1 — TRAIN
    ###################################################################
    if args.action == "train":
        # python assignment4.py train [train_text_path]
        train_text_path = args.arg1 or "./data/frankenstein.txt"
        print(f"[TRAIN] Using training text: {train_text_path}")
        model_number = int(args.arg2) if args.arg2 is not None else 0 
        print(f"[TRAIN] Using training text: {train_text_path}")
        print(f"[TRAIN] Training model index {model_number} with config name '{GPT_CONFIG[model_number].get('name', model_number)}'")


        """
        TODO: Modify the config.json file to train different models.
        Keep in mind the computational cost of increasing or decreasing certain parameters.
        """
        

        train_losses, val_losses, tokens_seen, model = main(
            GPT_CONFIG[model_number],
            OTHER_SETTINGS[model_number],
            train_text_path=train_text_path,
        )

        epochs_tensor = torch.linspace(
            0,
            OTHER_SETTINGS[model_number]["num_epochs"],
            len(train_losses),
        )
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig(f"loss_{model_number}.png")
        print(f"[TRAIN] Saved loss plot to loss_{model_number}.png")

        model_path = f"model_{model_number}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"[TRAIN] Saved model state dict to {model_path}")

    ###################################################################
    # ACTION 2 — DECODE
    ###################################################################
    elif args.action == "decode":
        # python assignment4.py decode <model_number> <decoder_name> [start_text]
        if args.arg1 is None or args.arg2 is None:
            raise ValueError(
                "Usage: python assignment4.py decode <model_number> <decoder_name> [start_text]\n"
                "  decoder_name ∈ {decoder_default, decoder_1, decoder_2}"
            )

        try:
            model_num = int(args.arg1)
        except ValueError:
            raise ValueError("model_number must be an integer (e.g., 0, 1, 2).")


        decoder_name = args.arg2
        start_text = args.arg3 or "My beloved Sister,"

        model_path = f"model_{model_num}.pth"
        print(f"[DECODE] Using model file: {model_path}")
        print(f"[DECODE] Decoder: {decoder_name}")
        print(f"[DECODE] Start text: {start_text!r}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Train first or choose another model_number."
            )

        model = GPTModel(GPT_CONFIG[model_num])
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        encoded = text_to_token_ids(
            text=start_text,
            tokenizer=tokenizer,
        ).to(device)

        context_size = model.pos_emb.weight.shape[0]

        with torch.no_grad():
            if decoder_name == "decoder_default":
                token_ids = generate_text_simple(
                    model=model,
                    idx=encoded,
                    max_new_tokens=50,
                    context_size=context_size,
                )
            elif decoder_name == "decoder_1":
                token_ids = decode_1(
                    model=model,
                    idx=encoded,
                    max_new_tokens=50,
                    context_size=context_size,
                )
            elif decoder_name == "decoder_2":
                token_ids = decode_2(
                    model=model,
                    idx=encoded,
                    max_new_tokens=50,
                    context_size=context_size,
                )
            else:
                raise ValueError(
                    "decoder_name must be one of: decoder_default, decoder_1, decoder_2."
                )

        decoded_text = token_ids_to_text(token_ids, tokenizer=tokenizer)
        print("\n[DECODE] Generated text:\n")
        print(decoded_text.replace("\n", " "))

    ###################################################################
    # ACTION 3 — PERPLEXITY
    ###################################################################
    elif args.action == "perplexity":
        # python assignment4.py perplexity <model_number> [test_text_path]
        if args.arg1 is None:
            raise ValueError(
                "Usage: python assignment4.py perplexity <model_number> [test_text_path]"
            )

        try:
            model_num = int(args.arg1)
        except ValueError:
            raise ValueError("model_number must be an integer (e.g., 0, 1, 2).")

        test_text_path = args.arg2 or "data/die_automata.txt"

        print(f"[PERPLEXITY] Using model_{model_num}.pth on text file: {test_text_path}")

        model_path = f"model_{model_num}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Train first or choose another model_number."
            )

        if not os.path.exists(test_text_path):
            raise FileNotFoundError(
                f"Text file '{test_text_path}' not found. Provide a valid path."
            )

        model = GPTModel(GPT_CONFIG[model_num])
        
        """
        TODO:
        
        # 1) Load the trained model and get it ready for inference.
        # 2) Read the test text and turn it into token IDs.
        # 3) Create input and target sequences that are offset by one token.
        # 4) Process the data in chunks that fit the model’s context size.
        # 5) For each chunk, run the model and collect the log-probabilities
        #    of the correct next tokens.
        # 6) Compute the average negative log-likelihood over all tokens.
        # 7) Convert this value to perplexity and print the result.
        
        """
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        with open(test_text_path, "r", encoding="utf-8") as f:
            token_ids = text_to_token_ids(f.read(), tokenizer).to(device).squeeze() 

        N = token_ids.numel()
        if N < 2:
            raise ValueError("[PERPLEXITY] Test text too short (need at least 2 tokens).")

        context_size = model.pos_emb.weight.size(0)

        total_log_prob = 0.0
        total_count = 0

        with torch.no_grad():
            for i in range(0, N - 1, context_size):
                end = min(i + context_size, N - 1)

                x = token_ids[i:end].unsqueeze(0)     
                y = token_ids[i + 1:end + 1].unsqueeze(0)   

                logits = model(x)                           
                log_probs = torch.log_softmax(logits, dim=-1)

                token_log_probs = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

                total_log_prob += token_log_probs.sum().item()
                total_count += y.numel()

        avg_nll = -total_log_prob / total_count
        ppl = float(torch.exp(torch.tensor(avg_nll)))

        print(f"[PERPLEXITY] Tokens evaluated: {total_count}")
        print(f"[PERPLEXITY] Average NLL: {avg_nll:.4f}")
        print(f"[PERPLEXITY] Perplexity: {ppl:.4f}")


    ###################################################################
    # ACTION 4 — PRETRAINED MODEL (BONUS)
    ###################################################################
    elif args.action == "pretrained-model":
        # python assignment4.py pretrained-model
        print("[PRETRAINED] Loading pretrained GPT model...")
        start_context = "My beloved Sister,"
        load_pretrained_gpt(start_context=start_context)
