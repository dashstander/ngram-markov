import einops
import numpy as np
import torch
from transformer_lens import HookedTransformerConfig


def int_to_base_x(base: int, minlen: int, arr: np.ndarray):
    result = np.zeros((len(arr), minlen), dtype=int)
    arr_copy = arr.copy()

    for i in range(minlen - 1, -1, -1):
        result[:, i] = arr_copy % base
        arr_copy //= base

    return result


def create_ngrams(tensor, n):
    assert n >= 1
    stride = tensor.stride()
    size = tensor.size()

    # Create a new stride that moves by 1 element in the original tensor
    new_stride = (stride[0], stride[0])

    # Calculate the new shape of the tensor
    new_size = (size[0] - n + 1, n)

    # Use as_strided to create a new view of the tensor with the desired shape and stride
    ngrams = torch.as_strided(tensor, size=new_size, stride=new_stride)
    return ngrams


def nanogpt_to_hooked_transformer_config(model_args):
    cfg_dict = {
        "d_model": model_args['n_embed'],
        "n_layers": model_args["n_layer"],
        "d_mlp": model_args["n_embed"] * 4,
        "d_head": model_args["n_embed"] // model_args['n_head'],
        "n_heads": model_args["n_head"],
        "n_ctx": model_args["block_size"],
        "d_vocab": model_args["vocab_size"],
        "tokenizer_name": None,
        "act_fn": 'gelu',
        "attn_only": False,
        "positional_embedding_type": "standard",
        "normalization_type": "LN",
        "original_architecture": 'nanogpt'
    }
    return HookedTransformerConfig(**cfg_dict)



def convert_nanogpt_weights(old_state_dict, cfg: HookedTransformerConfig):
    """For https://github.com/karpathy/nanoGPT
    There are two complications with converting nanogpt models:
    The first is that some state dicts have an unwanted prefix on keys that needs to be removed.
    The second is that the models can be saved with or without bias. By default, there
    is no bias. This function can handle both cases."""
    # Nanogpt models saved after torch.compile() have this unwanted prefix
    # This is a simple way to remove it
    unwanted_prefix = "_orig_mod."
    for k, v in list(old_state_dict.items()):
        if k.startswith(unwanted_prefix):
            old_state_dict[k[len(unwanted_prefix) :]] = old_state_dict.pop(k)

    new_state_dict = {}
    new_state_dict["pos_embed.W_pos"] = old_state_dict["transformer.wpe.weight"]
    new_state_dict["embed.W_E"] = old_state_dict["transformer.wte.weight"]

    new_state_dict["ln_final.w"] = old_state_dict["transformer.ln_f.weight"]
    new_state_dict["ln_final.b"] = torch.zeros_like(old_state_dict["transformer.ln_f.weight"])
    new_state_dict["unembed.W_U"] = old_state_dict["lm_head.weight"].T


    bias = False
    if "transformer.ln_f.bias" in old_state_dict:
        bias = True
        new_state_dict["ln_final.b"] = old_state_dict["transformer.ln_f.bias"]
    else:
         new_state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    for layer in range(cfg.n_layers):
        layer_key = f"transformer.h.{layer}"

        new_state_dict[f"blocks.{layer}.ln1.w"] = old_state_dict[f"{layer_key}.ln_1.weight"]
        # A bias of zeros is required for folding layer norm
        new_state_dict[f"blocks.{layer}.ln1.b"] = torch.zeros_like(
            old_state_dict[f"{layer_key}.ln_1.weight"]
        )
        new_state_dict[f"blocks.{layer}.ln2.w"] = old_state_dict[f"{layer_key}.ln_2.weight"]
        new_state_dict[f"blocks.{layer}.ln2.b"] = torch.zeros_like(
            old_state_dict[f"{layer_key}.ln_2.weight"]
        )

        new_state_dict[f'blocks.{layer}.attn.mask'] = torch.tril(
            torch.ones((cfg.n_ctx, cfg.n_ctx)).bool()
        )
        new_state_dict[f'blocks.{layer}.attn.IGNORE'] = torch.tensor(-torch.inf)

        W = old_state_dict[f"{layer_key}.attn.c_attn.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_Q"] = W_Q
        new_state_dict[f"blocks.{layer}.attn.W_K"] = W_K
        new_state_dict[f"blocks.{layer}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_O"] = W_O

        new_state_dict[f"blocks.{layer}.mlp.W_in"] = old_state_dict[
            f"{layer_key}.mlp.c_fc.weight"
        ].T
        new_state_dict[f"blocks.{layer}.mlp.W_out"] = old_state_dict[
            f"{layer_key}.mlp.c_proj.weight"
        ].T

        if bias:
            new_state_dict[f"blocks.{layer}.ln1.b"] = old_state_dict[f"{layer_key}.ln_1.bias"]
            new_state_dict[f"blocks.{layer}.ln2.b"] = old_state_dict[f"{layer_key}.ln_2.bias"]
            new_state_dict[f"blocks.{layer}.mlp.b_in"] = old_state_dict[
                f"{layer_key}.mlp.c_fc.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_out"] = old_state_dict[
                f"{layer_key}.mlp.c_proj.bias"
            ]

            B = old_state_dict[f"{layer_key}.attn.c_attn.bias"]
            B_Q, B_K, B_V = torch.tensor_split(B, 3, dim=0)
            B_Q = einops.rearrange(B_Q, "(i h)->i h", i=cfg.n_heads)
            B_K = einops.rearrange(B_K, "(i h)->i h", i=cfg.n_heads)
            B_V = einops.rearrange(B_V, "(i h)->i h", i=cfg.n_heads)
            new_state_dict[f"blocks.{layer}.attn.b_Q"] = B_Q
            new_state_dict[f"blocks.{layer}.attn.b_K"] = B_K
            new_state_dict[f"blocks.{layer}.attn.b_V"] = B_V
            new_state_dict[f"blocks.{layer}.attn.b_O"] = old_state_dict[
                f"{layer_key}.attn.c_proj.bias"
            ]
        else:
            new_state_dict[f"blocks.{layer}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)
            new_state_dict[f"blocks.{layer}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)
            new_state_dict[f"blocks.{layer}.attn.b_Q"] = torch.zeros(
                (cfg.n_heads, cfg.d_head), dtype=cfg.dtype)
            new_state_dict[f"blocks.{layer}.attn.b_K"] = torch.zeros(
                cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
            new_state_dict[f"blocks.{layer}.attn.b_V"] = torch.zeros(
                cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
            new_state_dict[f"blocks.{layer}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)
            
    return new_state_dicts



def save_to_s3(fs, weights, optimizer, config, rng, bucket, step):
    with fs.open(f'{bucket}/{step}.pth', mode='wb') as file:
        torch.save(
            {
                'model': weights,
                'optimizer': optimizer, 
                'config': config,
                'rng': rng
            }, 
            file
        )