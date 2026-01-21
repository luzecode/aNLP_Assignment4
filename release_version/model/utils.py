import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def generate_text_simple(model, idx, max_new_tokens, context_size):

    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def decode_1(model, idx, max_new_tokens, context_size, k=50, temperature=1.0):
    """
    TODO: Implement a decoding algorithm that uses top-k.
    You can use the code from generate_text_simple as a base.
    You can modify the inputs however you think it is necessary.
    And remember to check pytorch official documentation.
    
    Args:
        model: A trained language model.

        idx: 
            Initial token indices.

        max_new_tokens: int
            Number of new tokens to generate.

        context_size: int
            Maximum number of most recent tokens used as model context
            at each generation step (sliding window over idx).
    
    Return:
        The generated sequence of token indices.
    
    
    """
    for _ in range(max_new_tokens):
        # extract the recent tokens up to the context_size
        idx_cond = idx[:, -context_size:]
        
        # perform inference without tracking gradients (faster and saves memory)
        with torch.no_grad():
            logits = model(idx_cond)

        # focusing is put on logits of the last token in the sequence
        logits = logits[:, -1, :]

        #temperature scaling to adjust the randomness of predictions
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
    
        # only the top-k logits are kept while the rest is masked
        top_k = min(k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

        # converting logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
        # next token based on the probability distribution
        idx_next = torch.multinomial(probabilities, num_samples=1)

        # append the sampled token
        idx = torch.cat((idx, idx_next), dim=1)
    
    # return the generated sequence
    return idx


def decode_2(model, idx, max_new_tokens, context_size, p=0.9, temperature=1.0):
    """ 
    TODO: Implement a decoding algorithm that uses a decoding strategy of your choice.
    You can use the code from generate_text_simple as a base.
    You can modify the inputs however you think it is necessary.
    And remember to check pytorch official documentation.
    """
    for _ in range(max_new_tokens):
        # extract the recent tokens up to the context_size
        idx_cond = idx[:, -context_size:]

        # Perform inference without tracking gradients (faster and saves memory)
        with torch.no_grad():
            logits = model(idx_cond)
        
        # focus is put on logits of the last token in the sequence
        logits = logits[:, -1, :]
        
        # temperature scaling to adjust the randomness of predictions
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # sorting logits in descending order and get their indices
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # these logits are converted to probabilities using softmax
        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        
        # cumulative probabilities are computed for nucleus sampling
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mask the tokens which cumulative probability exceeds the threshold p
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 0] = False  # Always keep at least one token
    
        # set logits of masked tokens to negative infinity to exclude them (0 after softmax)
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # calculate probabilities after masking
        probabilities = torch.nn.functional.softmax(sorted_logits, dim=-1)
        
        # the next token based on the updated probability distribution
        sampled_sorted_idx = torch.multinomial(probabilities, num_samples=1)
        
        # map the sampled token back to the original indices
        idx_next = sorted_indices.gather(-1, sampled_sorted_idx)

        # append the sampled token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx