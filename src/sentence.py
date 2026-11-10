import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# -----------------------------------------------------------------------------
# Aliases / defaults
# -----------------------------------------------------------------------------

ALIASES = {
    # Plain pretrained BERT: a *token* encoder with no sentence-level
    # training on top, unlike sbert/e5 whose weights were tuned to make one
    # particular pooling produce good sentence embeddings. That makes it the
    # control for the pooling experiments -- vary data.encoder.pooling over
    # this family and nothing about the representation is co-adapted to any
    # of the strategies being compared.
    "bert": {"bert", "bert-base"},
    # ELECTRA and RoBERTa are the other raw token encoders -- pretrained
    # stacks with no sentence-level objective on top, so like `bert` they
    # privilege no pooling strategy and can serve as controls.
    #   electra: same architecture AND the same WordPiece vocab as bert, so
    #     its subword segmentation is byte-identical. The bias test counts
    #     subwords per tag, so bert and electra share a null distribution and
    #     their z-scores compare cell for cell. What differs is the
    #     pretraining objective (replaced-token detection, not MLM).
    #   roberta: same architecture, but BPE with a 50k vocab -- it segments
    #     differently, so its per-tag token counts and therefore its null
    #     differ. Still interpretable, but it varies tokenisation as well as
    #     pretraining, and is not directly comparable at the token level.
    "electra": {"electra"},
    "roberta": {"roberta"},
    "sbert": {"sbert"},
    "e5": {"e5", "retrieval", "gte"},
    # SimCSE: the other lineage of purpose-built sentence encoder. Unlike e5 it is
    # NOT a retrieval model -- supervised contrastive training on NLI, no
    # query:/passage: prefix convention, so none of e5's prefix handling applies.
    # And unlike sbert (a 6-layer 384-dim distilled MiniLM) it is a full BERT-base
    # stack on the same WordPiece vocabulary, so it sits in the bert-base
    # tokenizer group and its per-tag counts -- hence its null distribution --
    # match bert, electra and sbert cell for cell.
    "simcse": {"simcse"},
    "llm": {"llm"},
}

ALIAS_TO_CANON = {
    alias: canon
    for canon, aliases in ALIASES.items()
    for alias in aliases
}

DEFAULT_MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "roberta": "roberta-base",
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base-v2",
    "simcse": "princeton-nlp/sup-simcse-bert-base-uncased",
    "llm": "EleutherAI/pythia-410m",
}

TOKENIZER_GROUPS = {
    # electra sits in the bert-base group because its tokenizer is the same
    # WordPiece vocabulary, verified to produce identical ids; roberta needs
    # its own because BPE does not.
    "bert-base": {"bert", "bert-base", "electra", "sbert", "e5", "retrieval", "gte",
                  "simcse"},
    "roberta": {"roberta"},
    "gpt": {"llm"},
}

CANONICAL_TOKENIZERS = {
    "bert-base": "bert-base-uncased",
    "roberta": "roberta-base",
    "gpt": "EleutherAI/pythia-410m",
}

# Extra from_pretrained kwargs, per tokenizer group. Every dataset here
# arrives as a list of words and is encoded with is_split_into_words=True
# (see encode_examples in src/data.py); RoBERTa's byte-level BPE refuses
# pre-tokenized input unless it is told that each word starts on a space
# boundary. Deliberately NOT applied to the gpt group: pythia's tokenizer
# accepts word lists as-is, and setting the flag there would change its
# segmentation and silently invalidate every existing llm run.
TOKENIZER_KWARGS = {
    "roberta": {"add_prefix_space": True},
}


def resolve_tokenizer_group(family: str) -> str:
    family = family.lower()
    for group, families in TOKENIZER_GROUPS.items():
        if family in families:
            return group
    raise ValueError(f"Unknown encoder family: {family}")


def resolve_tokenizer(family: str) -> AutoTokenizer:
    group = resolve_tokenizer_group(family)
    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_TOKENIZERS[group], use_fast=True, **TOKENIZER_KWARGS.get(group, {})
    )

    if tokenizer.pad_token is None:
        fallback = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
        if fallback is None:
            raise ValueError(
                f"Tokenizer {tokenizer.name_or_path} has no pad/eos/bos/unk token available for padding."
            )
        tokenizer.pad_token = fallback

    tokenizer.padding_side = "right"
    return tokenizer


# -----------------------------------------------------------------------------
# Token embedding backends
# -----------------------------------------------------------------------------

def _sdpa_context(device: torch.device):
    """Pin scaled_dot_product_attention to its reference kernel on CUDA.

    The selector's whole mechanism depends on gradients flowing back through
    the *attention mask* (the soft mask z enters as `attention_mask`), so this
    attention runs with autograd enabled rather than under no_grad. CUDA's
    memory-efficient backward kernel rejects the resulting broadcast mask --
    `RuntimeError: LSE is not correctly aligned (strideH)` -- because the
    bias is [B, 1, 1, T] with a zero stride over the head dimension.

    SDPBackend.MATH is the reference implementation, so this is a kernel
    choice and not an approximation: same values, more memory. Left untouched
    on CPU, which has no such restriction and is the path every result in the
    paper was produced on.

    Only applied when gradients are actually being tracked. The defect is in
    the memory-efficient *backward* kernel, so under torch.no_grad -- every
    evaluation path, and the whole of the brute-force oracle search -- the
    fast kernels are both correct and dramatically quicker. Forcing MATH
    there would buy nothing and cost a large multiple in throughput on
    workloads that are pure forward passes.
    """
    if device.type != "cuda" or not torch.is_grad_enabled():
        return contextlib.nullcontext()
    return sdpa_kernel([SDPBackend.MATH])


def bert_token_embeddings(
    model: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_index: int | None = None,
) -> torch.Tensor:
    """
    Returns hidden states [B, T, D] for BERT-style encoders.
    layer_index=None uses all blocks; 0 returns embeddings, N runs N blocks.
    Adds log(attention_mask) to attention scores, supporting fractional weights.
    """
    if layer_index is not None and not 0 <= layer_index <= len(model.encoder.layer):
        raise ValueError(f"Invalid BERT layer index: {layer_index}")
    hidden_states = model.embeddings(input_ids)
    key_mask = attention_mask[:, None, None, :].type_as(hidden_states)  # [B,1,1,T]

    for layer in model.encoder.layer[:layer_index]:
        attn = layer.attention.self
        bsz, seq_len, _ = hidden_states.size()

        q = attn.query(hidden_states)
        k = attn.key(hidden_states)
        v = attn.value(hidden_states)

        q = q.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
        k = k.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
        v = v.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)

        attn_bias = torch.log(key_mask.clamp(min=1e-9))  # 0 for valid, ≈-inf for masked
        with _sdpa_context(hidden_states.device):
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, attn.all_head_size)
        attn_out = layer.attention.output(context, hidden_states)
        hidden_states = layer.output(layer.intermediate(attn_out), attn_out)

    return hidden_states


def gpt_token_embeddings(
    model: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns last hidden states [B, T, D] for GPT-NeoX-style causal encoders,
    keeping FRACTIONAL attention_mask values differentiable.

    Why this is hand-written rather than a call to model(...): HuggingFace turns
    attention_mask into an additive bias inside create_causal_mask(), which
    treats it as a boolean keep/drop and detaches it. Passing a soft selector
    weight through that path produced an output with requires_grad=False, so the
    selector got no gradient and could never train -- which is why llm runs only
    ever existed as oracles. This mirrors bert_token_embeddings: add
    log(attention_mask) to the attention scores, so a weight of 1 contributes 0,
    a weight of 0 contributes -inf, and anything between is a smooth
    down-weighting the selector can be optimised through.

    The causal mask is applied on top of it. Everything else -- the fused QKV
    projection and its [B, T, H, 3*head] layout, partial rotary over the first
    rotary_ndims dimensions, and the parallel residual -- follows
    modeling_gpt_neox exactly, and is checked against the HF forward in
    tests/test_gpt_mask.py for an all-ones mask.
    """
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

    hidden_states = model.emb_dropout(model.embed_in(input_ids))
    bsz, seq_len, _ = hidden_states.shape

    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    cos, sin = model.rotary_emb(hidden_states, position_ids)

    # -inf above the diagonal: a token may not attend to its own future.
    causal = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device,
                        dtype=hidden_states.dtype).triu(1)
    # log(w): 0 where fully kept, -inf where dropped, smooth in between. Clamped
    # because the selector's soft z can sit marginally outside [0, 1].
    key_bias = torch.log(attention_mask.clamp(1e-9, 1.0)).type_as(hidden_states)
    attn_bias = causal[None, None, :, :] + key_bias[:, None, None, :]

    for layer in model.layers:
        attn = layer.attention
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        qkv = attn.query_key_value(normed).view(bsz, seq_len, -1, 3 * attn.head_size).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        with _sdpa_context(hidden_states.device):
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias,
                                                     scale=attn.scaling)
        context = context.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = layer.post_attention_dropout(attn.dense(context))

        if layer.use_parallel_residual:            # pythia: x + attn(ln1(x)) + mlp(ln2(x))
            mlp_out = layer.post_mlp_dropout(layer.mlp(layer.post_attention_layernorm(residual)))
            hidden_states = mlp_out + attn_out + residual
        else:
            attn_out = attn_out + residual
            mlp_out = layer.post_mlp_dropout(layer.mlp(layer.post_attention_layernorm(attn_out)))
            hidden_states = mlp_out + attn_out

    return model.final_layer_norm(hidden_states)


# -----------------------------------------------------------------------------
# Pooling strategies
# -----------------------------------------------------------------------------
#
# A pooling strategy is the second half of an experiment's identity: an
# experiment is a (token encoder, pooling strategy) pair. The token encoder
# decides what each token's hidden state is; the strategy decides how a set
# of those states becomes one sentence vector. Holding the encoder fixed and
# varying the strategy is what isolates whether a token-selection bias is a
# property of the representation or of the reduction applied to it.

POOLING_STRATEGIES = ("mean", "max", "min", "last", "cls")

def masked_pool(
    token_emb: torch.Tensor,
    pool_mask: torch.Tensor,
    strategy: str,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce token states [B, L, D] to one vector per sequence [B, D].

    mean: weighted average. min/max: scale each token vector by its score,
    then take the coordinate-wise extremum across the scaled vectors. A
    zero-scored token therefore contributes a zero *vector* that competes in
    the extremum like any other. Scores are not clipped -- the selector
    renormalises z to sum to k, so they can exceed 1, and leaving them
    unclipped is safe because the reduction is scale-equivariant
    (amax(a*x) == a*amax(x) for a > 0) and the reconstruction loss is a
    cosine: inflating every score uniformly cannot move it, so only the
    relative pattern of scores -- actual selection -- carries signal.
    last: last positive-score position; a hard index, so it passes no
    gradient to the scores at all and the selector trains through the
    attention path alone. Empty selections produce zeros.
    cls: position 0, i.e. the [CLS] token, for encoders trained against it
    (SimCSE). Needs model.keep_special=true. Like `last` it is a fixed index and
    passes no gradient to the scores, so the selector trains purely through what
    the mask does to attention -- which is the whole mechanism here, since CLS's
    own representation is built by attending over the surviving tokens.

    valid_mask marks real tokens so batch padding stays out of the extrema.
    It only bites when nothing is deselected (pool_full, or rho = 1): once
    any real token scores zero, its zero vector is already in the pool and
    padding's adds nothing. Callers working on padding-free rows -- the
    oracle's per-(n, k) groups, mask_optimality's trimmed sentences -- can
    omit it. See tests/test_pooling.py, which pins all of this.
    """
    if strategy not in POOLING_STRATEGIES:
        raise ValueError(
            f"Unknown pooling strategy {strategy!r}; expected one of {', '.join(POOLING_STRATEGIES)}."
        )

    mask = pool_mask.unsqueeze(-1).type_as(token_emb)

    if strategy == "mean":
        return (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    if strategy in ("max", "min"):
        weighted = token_emb * mask
        if valid_mask is not None:
            valid = valid_mask.bool().unsqueeze(-1)
            weighted = weighted.masked_fill(~valid, -torch.inf if strategy == "max" else torch.inf)
        pooled = weighted.amax(dim=1) if strategy == "max" else weighted.amin(dim=1)
        if valid_mask is not None:
            pooled = torch.where(valid.any(dim=1), pooled, torch.zeros_like(pooled))
        return pooled

    if strategy == "cls":
        # Position 0, which is the [CLS] token -- so this requires
        # model.keep_special=true, since strip_special_tokens() otherwise removes
        # it and position 0 would be the first ordinary word instead. The token is
        # not a word, so the selector never votes on it; RationaleSelectorModel
        # keeps every attended non-word position unmasked so that CLS survives
        # into the suppressed sentence and the loss compares CLS-of-full against
        # CLS-of-suppressed. No mask weighting is applied here for the same
        # reason: CLS is always present on both sides.
        return token_emb[:, 0]

    # strategy == "last": the TRUE last real token of the sequence -- the last
    # position of valid_mask -- whatever the selector kept. Before 2026-10-09
    # this was the last KEPT token (last position of pool_mask), so a selector
    # that dropped the final word was compared against the full sentence's
    # last-token state using an earlier token's state. Pair with
    # model.selector.keep_last, which guarantees that token is never masked.
    # Without valid_mask (pool_full, where nothing is masked) the pooling mask
    # marks exactly the real tokens, so it is used instead.
    real = (valid_mask if valid_mask is not None else pool_mask) > 0
    occupied = real.any(dim=1, keepdim=True)
    positions = torch.arange(token_emb.shape[1], device=token_emb.device)
    last_idx = (real.to(token_emb.dtype) * (positions + 1).to(token_emb.dtype)).max(dim=1).values.long() - 1
    pooled = token_emb[torch.arange(token_emb.shape[0], device=token_emb.device), last_idx.clamp(min=0)]
    return torch.where(occupied, pooled, torch.zeros_like(pooled))


# -----------------------------------------------------------------------------
# Encoder interface: token_embeddings (expensive, once) + pool (cheap, many times)
# -----------------------------------------------------------------------------

class SentenceEncoder(nn.Module):
    """
    - token_embeddings(ids, attn) runs the transformer with the supplied mask.
    - pool(token_emb, pool_mask, valid_mask) computes sentence repr for an arbitrary
      pool_mask. pool_mask is where your selector's g lives (e.g.,
      pool_mask = attn * g). valid_mask is the original padding mask, not g.
    - pool_full(token_emb, attn_mask) computes the whole-sentence embedding
      (the *entire* real sequence, not a subset) -- the reconstruction target.

    Both use the SAME configured strategy (data.encoder.pooling). That is
    deliberate: the selector is trained to reproduce pool_full's output from
    a subset via pool, so if the two used different reductions the selector
    would be learning a cross-operator mapping rather than answering "which
    tokens suffice under this pooling strategy". Keeping them identical is
    what makes results comparable across strategies.
    """
    def __init__(self, normalize: bool, pooling: str = "mean") -> None:
        super().__init__()
        self.normalize = normalize
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(
                f"Unknown pooling strategy {pooling!r}; expected one of {', '.join(POOLING_STRATEGIES)}."
            )
        self.pooling = pooling

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pool(
        self, token_emb: torch.Tensor, pool_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sent_emb = masked_pool(token_emb, pool_mask, self.pooling, valid_mask)
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb

    def pool_full(self, token_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.pool(token_emb, attn_mask, valid_mask=attn_mask)


# -----------------------------------------------------------------------------
# Concrete encoders
# -----------------------------------------------------------------------------

class FrozenSBERT(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool, pooling: str = "mean") -> None:
        super().__init__(normalize, pooling)
        self.model = SentenceTransformer(model_name)
        self.backbone = self.model[0].auto_model

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return bert_token_embeddings(self.backbone, input_ids, attention_mask)


class _FrozenHFEncoder(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool, pooling: str = "mean") -> None:
        super().__init__(normalize, pooling)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False


class _BertStackEncoder(_FrozenHFEncoder):
    """Any BERT-architecture stack loaded through AutoModel.

    Shared by the raw `bert` family and by `e5`: they differ in weights and
    in E5's "query: " input prefix (applied in src/data.py at tokenisation
    time), not in how token states are produced.
    """

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return bert_token_embeddings(self.model, input_ids, attention_mask)


class FrozenBERT(_BertStackEncoder):
    pass


class FrozenELECTRA(_BertStackEncoder):
    pass


class FrozenRoBERTa(_BertStackEncoder):
    pass


class FrozenE5(_BertStackEncoder):
    pass


class FrozenSimCSE(_BertStackEncoder):
    pass


class FrozenLLMEncoder(_FrozenHFEncoder):
    """Causal encoder.

    Note on `pooling=last`: causal attention means only the final real token
    has attended to the whole sequence, which is why last-token extraction is
    the standard way to embed a causal LM's sentence (GritLM / LLM2Vec /
    PromptEOL). That convention now lives in the shared `last` strategy
    rather than being hardcoded here, so it applies to whichever encoder is
    configured with it -- and so this encoder can equally be run under mean /
    max / min, which is the point of making pooling a free axis. Assumes
    right-padding, as this repo's tokenizers are configured.
    """

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return gpt_token_embeddings(self.model, input_ids, attention_mask)

    def next_token_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Next-token logits from final-layer states (the output of token_embeddings).

        Pythia's own output layer (embed_out), frozen and loaded on first use.
        Deliberately NOT a registered submodule: the selector holds this encoder
        as a child, so a registered head would enter every checkpoint and the
        optimizer's parameter list.
        """
        head = self.__dict__.get("_lm_head")
        if head is None:
            from transformers import AutoModelForCausalLM
            head = AutoModelForCausalLM.from_pretrained(self.model_name).embed_out.eval()
            for prm in head.parameters():
                prm.requires_grad = False
            self.__dict__["_lm_head"] = head
        if next(head.parameters()).device != state.device:
            head.to(state.device)
        return head(state)


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

# One entry per canonical family, same keys as DEFAULT_MODEL_NAMES. A table
# rather than a branch chain so that adding an encoder is three dict entries
# and (usually) a `pass`-bodied subclass of an existing stack.
ENCODER_CLASSES = {
    "bert": FrozenBERT,
    "electra": FrozenELECTRA,
    "roberta": FrozenRoBERTa,
    "sbert": FrozenSBERT,
    "e5": FrozenE5,
    "simcse": FrozenSimCSE,
    "llm": FrozenLLMEncoder,
}


def build_sentence_encoder(
    family: str,
    encoder_name: str | None,
    device: str | None = None,
    pooling: str = "mean",
) -> tuple[SentenceEncoder, AutoTokenizer]:
    """Build the (token encoder, pooling strategy) pair an experiment is defined by.

    `family` picks the token encoder, `pooling` the reduction applied to its
    token states (see POOLING_STRATEGIES). The two are independent by design:
    the research question is whether a selection bias follows the encoder or
    the pooling, which is only answerable if either can be varied alone.
    """
    family = ALIAS_TO_CANON[family.lower()]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if encoder_name in {None, "None", "null", "NULL"}:
        encoder_name = DEFAULT_MODEL_NAMES[family]

    tokenizer = resolve_tokenizer(family)

    encoder = ENCODER_CLASSES[family](encoder_name, normalize=False, pooling=pooling)
    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer
