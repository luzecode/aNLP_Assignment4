import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # The output dimension needs to be fully divisible by number of heads, bacause each head will operate on d_out // num_heads dimensions
        assert d_out % num_heads == 0
        # Storing the parateters in self. variables
        self.d_out = d_out # output dimensions of multihead attention block
        self.num_heads = num_heads # num attention heads
        self.head_dim = d_out // num_heads # how many output dimensions per head

        # with Linear Projection the input is transformed into keys, queries and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # output projection: combines multi-head outputs back to d_out dimensions 
        self.out_proj = nn.Linear(d_out, d_out)

        #dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # causal mask: blocks future tokens (1s above diagonal → masked to -inf)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """the forward pass of the mutli-head attention, taking the Tensor as input"""
        #extract information from input tensor x (b = batch size, num_tokens = sequence length, d_in = input dimension)
        b, num_tokens, d_in = x.shape

        # project the input to keys, queries and values
        keys = self.W_key(x)  
        queries = self.W_query(x) 
        values = self.W_value(x)

        # reshape the Ks, Qs and Vs, From: (batch_size, num_tokens, d_out), To:(batch_size, num_tokens, num_heads, head_dim)
        #  splitting the d_out into the num_heads and head_dim 
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # takeing the transpose to switch order because more efficient for batched matric multiplication across heads
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores via dot product between queries and keys, represents how much one token should attend to another
        attn_scores = queries @ keys.transpose(2, 3) 

        #boolean mask for the sequency length, slicing the defined mask to fit the sequence length and "TRUE" indicated positions that should eb masked
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # applying the causal mask to the true positions, -inf will become 0 after softmax.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # applying softmax to get attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        #applying dropout to regualize the attention weights
        attn_weights = self.dropout(attn_weights)

        # computing the weighted sum for values with the attentin weight, then this is transposed
        context_vec = (attn_weights @ values).transpose(1, 2)

        # reshape merges all outputs from heads back into single representation
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)

        # final linear projection, allows model to learn to combine th einformation from the heads
        context_vec = self.out_proj(context_vec)  

        return context_vec


class LayerNorm(nn.Module):
    """normalizes the inputs across the feature dimensions"""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 #prevents division by 0
        self.scale = nn.Parameter(torch.ones(emb_dim)) # scale parameter, to scale normalized output
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # shift parameter, to shift normalized output

    def forward(self, x):
        # get the mean across last dimesnion (dim = -1)
        mean = x.mean(dim=-1, keepdim=True)
        # get variance across last dimesnion dim = -1
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # normalize by substracting the mean and deviding by the standard deviation (eps for avoiding devision by 0)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
       
        #this allows the model to eadrn the optimal scale and offset 
        return self.scale * norm_x + self.shift


class GELU(nn.Module): 
    """Gaussian Error Linear Unit (GELU) activation function: smooth activation function combining reLu and droput"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #GELU formula 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # container for the feed-forward layers
        self.layers = nn.Sequential(
            #first linear layer allwos for richer intermediate representations
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            #gelu activation. non-linearity between two linear layers
            GELU(),
            #second linear layer returns to the originla dimension for the residual connection
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        #pass through all layers sequentailly
        return self.layers(x)


class TransformerBlock(nn.Module):
    """single TRansformer Bloc (decoder layer), combines Multihead attention and feedforward netweork"""

    def __init__(self, cfg):
        super().__init__()
        # getting MutiHeadATtention Block as sub layer
        self.att = MultiHeadAttention(
            #parameters
            d_in=cfg["emb_dim"], #input dimension
            d_out=cfg["emb_dim"], # output dimesnion (should be the same as input)
            context_length=cfg["context_length"], # maximum sequence length
            num_heads=cfg["n_heads"], # number of attention heads
            dropout=cfg["drop_rate"], # dropout rate
            qkv_bias=cfg["qkv_bias"]) # bias in projections
        
        #getting the feed-forward network as sub-layer
        self.ff = FeedForward(cfg)
        #LayerNorm before attention
        self.norm1 = LayerNorm(cfg["emb_dim"])
        #LayerNorm before feed-forward
        self.norm2 = LayerNorm(cfg["emb_dim"])
        #dropout for regularization with residual connection
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x): #showing the architecture inside of a transformer block
        #MultiHeadAttention Sub-Layer

        # save input for later residual connection
        shortcut = x
        # PreNormalization: normalize before attention
        x = self.norm1(x)
        # apply multihead self-attention
        x = self.att(x)   
        # apply dropout for regularization
        x = self.drop_shortcut(x)
        # residual connection: x after dropout + the input from earlier as residual
        # this helps gradient flow and allows learning identity mappings
        x = x + shortcut 
        #FeedForward Network Sub-layer

        # save input for later residual connection
        shortcut = x
        # Pre-normalization: normalize before feed-forward
        x = self.norm2(x)
        # apply feed-forward network
        x = self.ff(x)
        # apply dropout for regularization
        x = self.drop_shortcut(x)
        # Residual connection: add input back
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """Complete GPT Model"""
    def __init__(self, cfg):
        super().__init__()

        # token embedding layer to map token IDs to vectors: vocab_size tokens, each represented by emb_dim dimensional vector
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # position embedding layer maps each position (0 to context_length-1) to an emb_dim vector, these are learned
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # dropout applied to embeddings for regularization
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # stack of transformaer blocks, depending on how many layers
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Final layer normalization before output projection. This stabilizes the representations before the final linear layer
        self.final_norm = LayerNorm(cfg["emb_dim"])


        # Output projection head: maps from emb_dim to vocab_size. here logits are produces for each vocabulary token
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # extract batch size and sequence length from input shape
        batch_size, seq_len = in_idx.shape
        
        # get token embeddings. each token ID is replaced by its learned embedding vector
        tok_embeds = self.tok_emb(in_idx)
        
        # get position embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # add token and position embeddings to give each token information about both its identity and position
        x = tok_embeds + pos_embeds
        
        # apply dropout to combined embeddings for regularization
        x = self.drop_emb(x)
        
        # pass through all transformer blocks sequentially, each block updates the representations through attention and feed-forward
        x = self.trf_blocks(x)
        
        # apply final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary size to get logits, which when passed through softmax produce probabilities
        logits = self.out_head(x)
        
        return logits