#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .transformer import AdaLNSelfAttn
from .modules import BITWISE_VAE, MimiModelWrapper, Wav2Vec2Model, Wav2Vec2Config, StyleEncoder

class BitwiseARModel(nn.Module):
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()
        # build basic vae
        self.basic_vae = BITWISE_VAE(model_cfg=model_cfg["VAE_CONFIG"])
        self.patch_nums = self.basic_vae.patch_nums
        self.vqfeat_embed = nn.Linear(self.basic_vae.code_dim, 768)
        # style encoder
        self.style_encoder = StyleEncoder()
        self.style_cond_embed = nn.Linear(128, 768)
        # audio encoder
        if model_cfg["AR_CONFIG"]['AUDIO_ENCODER'] == 'wav2vec':
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
            self.audio_encoder = Wav2Vec2Model(config)
            self.audio_feature_dim = 1024
        elif model_cfg["AR_CONFIG"]['AUDIO_ENCODER'] == 'mimi':
            self.audio_encoder = MimiModelWrapper()
            self.audio_feature_dim = 512
        else:
            raise ValueError("Invalid audio encoder: {}".format(model_cfg["AR_CONFIG"]['AUDIO_ENCODER']))
        # autoregressive generator
        self.attn_depth = model_cfg["AR_CONFIG"]['T_DEPTH']
        dpr = [x.item() for x in torch.linspace(0, 0.1 * self.attn_depth/24, self.attn_depth)]
        self.attn_blocks = nn.ModuleList([
            AdaLNSelfAttn(embed_dim=768, cond_dim=self.audio_feature_dim, num_heads=model_cfg["AR_CONFIG"]['T_NUM_HEADS'], drop_path=dpr[depth_idx])
            for depth_idx in range(self.attn_depth)
        ])
        # logits head part
        self.cond_logits_head = AdaLNBeforeHead(embed_dim=768, cond_dim=self.audio_feature_dim)
        self.logits_head = nn.Linear(768, self.basic_vae.code_dim * 2)
        self.null_style_cond = nn.Parameter(torch.randn(1, 1, 768) * 0.5)
        # absolute position and level embedding
        self.prev_ratio = model_cfg["AR_CONFIG"]['PREV_RATIO']
        pos_embed = torch.empty(1, sum(self.patch_nums), 768) # 1, L, C
        nn.init.trunc_normal_(pos_embed, mean=0, std=math.sqrt(1 / 768 / 3))
        self.pos_embed = nn.Parameter(pos_embed)
        prev_pos_embed = torch.empty(1, sum(self.patch_nums) * self.prev_ratio, 768)
        nn.init.trunc_normal_(prev_pos_embed, mean=0, std=math.sqrt(1 / 768 / 3))
        self.prev_pos_embed = nn.Parameter(prev_pos_embed)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), 768)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=math.sqrt(1 / 768 / 3))
        attn_bias_for_masking, lvl_idx = self.build_attn_mask(self.patch_nums)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking)
        self.register_buffer('lvl_idx', lvl_idx)

        # Streaming state variables
        self._streaming_state_initialized = False
        self._streaming_batch_size = None
        self._streaming_motion_style_cond = None
        self._streaming_prev_code_bits = None
        self._streaming_prev_attn_feat = None
        self._streaming_lvl_pos_embed = None
        self._streaming_prev_lvl_pos_embed = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def expected_audio_chunk_length(self):
        """
        The expected length of audio chunks (in samples) for streaming inference.
        Audio chunks passed to `inference_streaming_chunk` should ideally be of this length.
        The last chunk may be shorter and should be padded by the caller to this length.
        """
        return int(self.patch_nums[-1] / 25.0 * 16000)

    @torch.no_grad()
    def _initialize_streaming_inference(self, batch_size, style_motion=None):
        assert batch_size == 1, "Only support batch size 1 for streaming inference."
        self._streaming_batch_size = batch_size

        if style_motion is not None:
            motion_style = self.style_encoder(style_motion).detach()
            self._streaming_motion_style_cond = self.style_cond_embed(motion_style)[:, None]
            self._streaming_motion_style_cond = self._streaming_motion_style_cond * 1.1 - self.null_style_cond * 0.1
        else:
            print("No style motion provided, use default style condition for streaming.")
            self._streaming_motion_style_cond = self.null_style_cond.repeat(self._streaming_batch_size, 1, 1)

        prev_motion = torch.zeros(self._streaming_batch_size, self.patch_nums[-1], self.basic_vae.motion_dim, device=self.device)
        self._streaming_prev_code_bits, _ = self.basic_vae.quant_to_vqidx(prev_motion, this_motion=None)
        
        prev_vqfeat = self.basic_vae.vqidx_to_ms_vqfeat(self._streaming_prev_code_bits)
        base_prev_attn_feat_content = torch.cat([self._streaming_motion_style_cond, self.vqfeat_embed(prev_vqfeat)], dim=1)
        # Initialize prev_attn_feat with prev_ratio repetitions of the initial context
        self._streaming_prev_attn_feat = base_prev_attn_feat_content.repeat(1, self.prev_ratio, 1)

        # Precompute position embeddings for streaming
        self._streaming_lvl_pos_embed = (self.lvl_embed(self.lvl_idx) + self.pos_embed).detach()
        self._streaming_prev_lvl_pos_embed = (self.lvl_embed(self.lvl_idx).repeat(1, self.prev_ratio, 1) + self.prev_pos_embed).detach()
        
        self._streaming_state_initialized = True

    @torch.no_grad()
    def _process_audio_chunk(self, audio_chunk):
        if not self._streaming_state_initialized:
            raise RuntimeError("Streaming state not initialized. Call inference_streaming_start first.")
        
        if audio_chunk.shape[-1] != self.expected_audio_chunk_length:
            # This is a safeguard; ideally the caller handles padding to expected_audio_chunk_length
            print(f"Warning: Audio chunk length {audio_chunk.shape[-1]} does not match expected {self.expected_audio_chunk_length}. Ensure proper padding.")

        # Audio feature extraction for the current chunk
        split_audio_feat = self.audio_encoder(audio_chunk).permute(0, 2, 1) # B, L, C -> B, C, L
        split_audio_feats = [F.interpolate(split_audio_feat, size=(pn), mode='area').permute(0, 2, 1) for pn in self.patch_nums] # B, L, C
        split_audio_cond = torch.cat(split_audio_feats, dim=1).detach()
        
        next_ar_vqfeat = self._streaming_motion_style_cond.clone() 

        for pidx, pn in enumerate(self.patch_nums):
            patch_audio_cond = split_audio_cond[:, :sum(self.patch_nums[:pidx+1])]
            current_context_len = sum(self.patch_nums[:pidx+1]) # Length of current vq features being generated
            total_prev_context_len = self._streaming_prev_attn_feat.shape[1] # Length of prev_attn_feat (style_cond + prev_ratio * vq_feats)

            # Slice the precomputed attention bias mask
            # Current features (next_ar_vqfeat) length is 1 (style) + current_context_len (being generated)
            # However, next_ar_vqfeat starts with 1 (style) and grows up to 1 + sum(patch_nums)
            # The mask dimensions are related to the number of tokens.
            # Current tokens = next_ar_vqfeat.shape[1]
            # Previous tokens = _streaming_prev_attn_feat.shape[1]
            # Mask should be [:, :, :current_tokens, :current_tokens + previous_tokens]
            mask_current_dim = next_ar_vqfeat.shape[1]
            # The original mask `attn_bias_for_masking` is (1,1,L, L_prev + L_curr)
            # L is sum(self.patch_nums). L_prev is sum(self.patch_nums)*self.prev_ratio
            # We need to select the part of the mask relevant to the current generation step (pidx)
            # The current features being generated correspond to sum(self.patch_nums[:pidx+1])
            # The prev_attn_feat has a fixed structure.
            
            # The slicing of attn_bias_for_masking needs to correspond to the actual token lengths.
            # The length of tokens for the current chunk generation is sum(self.patch_nums).
            # The length of the historical context is self._streaming_prev_attn_feat.shape[1].
            # The original mask is built for L = sum(patch_nums).
            # We need the part of the mask that corresponds to the current hierarchical level pidx.
            # The first dimension of the mask (dim=2) is for "query" tokens (current context)
            # The second dimension of the mask (dim=3) is for "key/value" tokens (previous context + current context)
            
            # For the pidx-th patch group, we are predicting `pn` tokens.
            # The `next_ar_vqfeat` grows from just style_cond to style_cond + all generated tokens for this chunk.
            # The `patch_audio_cond` also grows.

            # Let's simplify the mask indexing based on the original inference method's loop:
            # `patch_attn_bias = self.attn_bias_for_masking[:, :, :sum(self.patch_nums[:pidx+1]), :sum(self.patch_nums[:pidx+1])+sum(self.patch_nums)*self.prev_ratio]`
            # Here, the second sum(self.patch_nums)*self.prev_ratio corresponds to the VQ part of prev_attn_feat.
            # Our self._streaming_prev_attn_feat = [style_cond, VQ_feats_repeated]
            # So, total_prev_context_len already includes the style token.
            # The original mask's third dim slice: sum(self.patch_nums[:pidx+1]) is correct for current generated tokens.
            # The original mask's fourth dim slice needs to be sum(self.patch_nums[:pidx+1]) for current tokens,
            # PLUS self._streaming_prev_attn_feat.shape[1] for all previous context tokens.
            # The precomputed attn_bias_for_masking has shape (1, 1, L, L_prev_vq + L_curr_vq)
            # L_curr_vq = sum(self.patch_nums), L_prev_vq = sum(self.patch_nums) * self.prev_ratio
            # The part of the mask for "current processing window" is what we need.
            # Max current tokens in a window = 1 (style) + sum(patch_nums)
            # Let's use the full mask for the chunk and rely on model to learn from pos_embeds
            # No, the mask logic from original should be replicated.

            num_current_tokens_for_mask = sum(self.patch_nums[:pidx+1])
            # The total length of the key/value sequence for attention is:
            # num_current_tokens_for_mask (from current generation) + total_prev_context_len (from _streaming_prev_attn_feat)
            
            # The original attn_bias_for_masking last dimension is L + L*prev_ratio (VQ parts only)
            # Our prev_attn_feat includes style_cond. The mask might need adjustment or style_cond is handled as "0" bias.
            # Assuming the original mask's zero_attn_bias_for_masking part handles the style_cond implicitly.
            patch_attn_bias = self.attn_bias_for_masking[:, :, :num_current_tokens_for_mask, :num_current_tokens_for_mask + sum(self.patch_nums)*self.prev_ratio]


            attn_feat = next_ar_vqfeat + self._streaming_lvl_pos_embed[:, :next_ar_vqfeat.shape[1]]
            for bidx in range(self.attn_depth):
                # The prev_lvl_pos_embed might need slicing if prev_attn_feat structure changed.
                # Original: prev_attn_feat + prev_lvl_pos_embed
                # Our prev_attn_feat is [style_cond, vq_feat_repeated]
                # Our prev_lvl_pos_embed is for [vq_feat_repeated]
                # We need to add positional embeddings only to the VQ part of prev_attn_feat
                
                # Correct prev_attn_feat_with_pos:
                # The first token of _streaming_prev_attn_feat is style_cond, which does not get pos_embed.
                # _streaming_prev_lvl_pos_embed corresponds to the VQ part.
                prev_tokens_for_attn = self._streaming_prev_attn_feat.clone()
                if prev_tokens_for_attn.shape[1] > 1 : # If there's more than just style_cond
                     prev_tokens_for_attn[:, 1:] = prev_tokens_for_attn[:, 1:] + self._streaming_prev_lvl_pos_embed[:, :prev_tokens_for_attn.shape[1]-1]


                attn_feat = self.attn_blocks[bidx](
                    attn_feat, 
                    prev_tokens_for_attn,
                    patch_audio_cond, 
                    attn_bias=patch_attn_bias
                )
            pred_motion_logits = self.logits_head(self.cond_logits_head(attn_feat, patch_audio_cond))
            pred_motion_bits = pred_motion_logits.view(pred_motion_logits.shape[0], pred_motion_logits.shape[1], -1, 2).argmax(dim=-1)
            if pidx < len(self.patch_nums) - 1:
                next_ar_vqfeat_content = self.basic_vae.vqidx_to_ar_vqfeat(pidx, pred_motion_bits)
                next_ar_vqfeat = torch.cat([self._streaming_motion_style_cond, self.vqfeat_embed(next_ar_vqfeat_content)], dim=1)
        
        _, split_pred_motion = self.basic_vae.vqidx_to_motion(self._streaming_prev_code_bits, pred_motion_bits)
        
        # Update state for the next chunk
        new_prev_code_bits, _ = self.basic_vae.quant_to_vqidx(split_pred_motion, this_motion=None)
        self._streaming_prev_code_bits = new_prev_code_bits.detach()
        
        new_prev_vqfeat = self.basic_vae.vqidx_to_ms_vqfeat(self._streaming_prev_code_bits).detach()
        this_prev_attn_feat_content = torch.cat([self._streaming_motion_style_cond, self.vqfeat_embed(new_prev_vqfeat)], dim=1)
        
        # Slide the window for _streaming_prev_attn_feat
        self._streaming_prev_attn_feat = torch.cat([
            self._streaming_prev_attn_feat[:, this_prev_attn_feat_content.shape[1]:], 
            this_prev_attn_feat_content
        ], dim=1).detach()
        
        return split_pred_motion.detach()

    @torch.no_grad()
    def inference_streaming_start(self, batch_size, style_motion=None):
        """
        Initializes the model for streaming inference.
        Call this once before processing any audio chunks.

        Args:
            batch_size (int): The batch size (must be 1).
            style_motion (torch.Tensor, optional): Style motion tensor. Defaults to None for default style.
        """
        if self._streaming_state_initialized:
            print("Warning: Streaming state was already initialized. Re-initializing.")
            self.inference_streaming_end() # Clean up previous state first
        self._initialize_streaming_inference(batch_size, style_motion)

    @torch.no_grad()
    def inference_streaming_chunk(self, audio_chunk):
        """
        Processes a single chunk of audio and returns the predicted motion.
        `inference_streaming_start` must be called before this.
        The input `audio_chunk` should be of length `self.expected_audio_chunk_length`.
        If it's the last, possibly shorter, chunk from an audio stream, it should be
        padded to `self.expected_audio_chunk_length` by the caller.

        Args:
            audio_chunk (torch.Tensor): A chunk of audio data (B, NumSamples).
                                        Expected B=1. NumSamples should be `self.expected_audio_chunk_length`.

        Returns:
            torch.Tensor: Predicted motion for the input audio chunk.
        """
        if not self._streaming_state_initialized:
            raise RuntimeError("Streaming not initialized. Call inference_streaming_start() first.")
        if audio_chunk.shape[0] != self._streaming_batch_size:
            raise ValueError(f"Audio chunk batch size {audio_chunk.shape[0]} does not match "
                             f"initialized streaming batch size {self._streaming_batch_size}.")
        
        # Ensure audio_chunk is on the same device as the model
        audio_chunk = audio_chunk.to(self.device)
        
        return self._process_audio_chunk(audio_chunk)

    @torch.no_grad()
    def inference_streaming_end(self):
        """
        Clears the streaming state. Call this after processing all audio chunks.
        """
        self._streaming_state_initialized = False
        self._streaming_batch_size = None
        self._streaming_motion_style_cond = None
        self._streaming_prev_code_bits = None
        self._streaming_prev_attn_feat = None
        self._streaming_lvl_pos_embed = None
        self._streaming_prev_lvl_pos_embed = None
        # print("Streaming state cleared.") # Optional: for debugging

    @torch.no_grad()
    def inference(self, batch, with_gtmotion=False):
        batch_size = batch["audio"].shape[0]
        assert batch_size == 1, "Only support batch size 1 for inference."
        seq_length = math.ceil(batch["audio"].shape[-1] / 16000 * 25.0)
        if 'style_motion' in batch.keys() and batch['style_motion'] is not None:
            motion_style = self.style_encoder(batch["style_motion"]).detach()
            motion_style_cond = self.style_cond_embed(motion_style)[:, None]
            motion_style_cond = motion_style_cond * 1.1 - self.null_style_cond * 0.1
        else:
            print("No style motion provided, use default style condition.")
            motion_style_cond = self.null_style_cond.repeat(batch_size, 1, 1) # Ensure batch_size conformity
            
        lvl_pos_embed = self.lvl_embed(self.lvl_idx) + self.pos_embed
        prev_lvl_pos_embed = self.lvl_embed(self.lvl_idx).repeat(1, self.prev_ratio, 1) + self.prev_pos_embed

        # padding frames and audios
        padded_frame_length = math.ceil(seq_length / self.patch_nums[-1]) * self.patch_nums[-1]
        padded_audio_length = int(padded_frame_length / 25.0 * 16000)
        patch_audio_length = int(self.patch_nums[-1] / 25.0 * 16000)
        audio_chunks_data = batch["audio"]
        # Pad audio_chunks_data if its total length is less than padded_audio_length
        if audio_chunks_data.shape[1] < padded_audio_length:
            audio_chunks_data = torch.cat([
                    audio_chunks_data, audio_chunks_data.new_zeros(batch_size, padded_audio_length - audio_chunks_data.shape[1])
                ], dim=-1
            )
        elif audio_chunks_data.shape[1] > padded_audio_length: # If audio is longer than needed for seq_length
            audio_chunks_data = audio_chunks_data[:, :padded_audio_length]

        audio_chunks = audio_chunks_data.split(patch_audio_length, dim=-1)

        prev_motion = batch["audio"].new_zeros(batch_size, self.patch_nums[-1], self.basic_vae.motion_dim)
        prev_code_bits, _ = self.basic_vae.quant_to_vqidx(prev_motion, this_motion=None)
        prev_vqfeat = self.basic_vae.vqidx_to_ms_vqfeat(prev_code_bits)
        # prev_attn_feat structure: [style_cond, vq_feat_repeated]
        base_prev_attn_feat_content = torch.cat([motion_style_cond, self.vqfeat_embed(prev_vqfeat)], dim=1)
        prev_attn_feat = base_prev_attn_feat_content.repeat(1, self.prev_ratio, 1)

        all_pred_motions = []
        for idx in range(len(audio_chunks)):
            current_audio_chunk = audio_chunks[idx]
            # Ensure current_audio_chunk has length patch_audio_length (pad if it's the last, possibly shorter, chunk)
            if current_audio_chunk.shape[1] < patch_audio_length:
                padding_needed = patch_audio_length - current_audio_chunk.shape[1]
                current_audio_chunk = F.pad(current_audio_chunk, (0, padding_needed))
            
            split_audio_feat = self.audio_encoder(current_audio_chunk).permute(0, 2, 1) # B, L, C -> B, C, L
            split_audio_feats = [F.interpolate(split_audio_feat, size=(pn), mode='area').permute(0, 2, 1) for pn in self.patch_nums] # B, L, C
            split_audio_cond = torch.cat(split_audio_feats, dim=1).detach()
            
            next_ar_vqfeat = motion_style_cond.clone() # Clone to avoid issues if motion_style_cond is modified by reference
            for pidx, pn in enumerate(self.patch_nums):
                patch_audio_cond = split_audio_cond[:, :sum(self.patch_nums[:pidx+1])]
                
                num_current_tokens_for_mask = sum(self.patch_nums[:pidx+1])
                # Original mask is (1,1,L,L_prev_vq + L_curr_vq). L_curr_vq is sum(patch_nums)
                # We need the part corresponding to num_current_tokens_for_mask
                patch_attn_bias = self.attn_bias_for_masking[:, :, :num_current_tokens_for_mask, :num_current_tokens_for_mask + sum(self.patch_nums)*self.prev_ratio]

                attn_feat = next_ar_vqfeat + lvl_pos_embed[:, :next_ar_vqfeat.shape[1]]
                
                # Add positional embeddings to the VQ part of prev_attn_feat
                current_prev_attn_feat_with_pos = prev_attn_feat.clone()
                if current_prev_attn_feat_with_pos.shape[1] > 1: # If there's more than just style_cond
                    # prev_lvl_pos_embed is for VQ part, which is prev_attn_feat[:,1:]
                    # Ensure slicing is correct for prev_lvl_pos_embed
                    len_vq_prev_part = current_prev_attn_feat_with_pos.shape[1] - 1
                    current_prev_attn_feat_with_pos[:, 1:] = current_prev_attn_feat_with_pos[:, 1:] + prev_lvl_pos_embed[:, :len_vq_prev_part]

                for bidx in range(self.attn_depth):
                    attn_feat = self.attn_blocks[bidx](attn_feat, current_prev_attn_feat_with_pos, patch_audio_cond, attn_bias=patch_attn_bias)
                
                pred_motion_logits = self.logits_head(self.cond_logits_head(attn_feat, patch_audio_cond))
                pred_motion_bits = pred_motion_logits.view(pred_motion_logits.shape[0], pred_motion_logits.shape[1], -1, 2).argmax(dim=-1)
                if pidx < len(self.patch_nums) - 1:
                    next_ar_vqfeat_content = self.basic_vae.vqidx_to_ar_vqfeat(pidx, pred_motion_bits)
                    next_ar_vqfeat = torch.cat([motion_style_cond, self.vqfeat_embed(next_ar_vqfeat_content)], dim=1)
            
            split_prev_motion, split_pred_motion = self.basic_vae.vqidx_to_motion(prev_code_bits, pred_motion_bits)
            all_pred_motions.append(split_pred_motion)
            # set next
            prev_code_bits, _ = self.basic_vae.quant_to_vqidx(split_pred_motion, this_motion=None) # Update prev_code_bits for the next chunk's VAE call
            new_prev_vqfeat = self.basic_vae.vqidx_to_ms_vqfeat(prev_code_bits).detach()
            this_prev_attn_feat_content = torch.cat([motion_style_cond, self.vqfeat_embed(new_prev_vqfeat)], dim=1)
            # Slide the window for prev_attn_feat
            prev_attn_feat = torch.cat([prev_attn_feat[:, this_prev_attn_feat_content.shape[1]:], this_prev_attn_feat_content], dim=1)

        pred_motions = torch.cat(all_pred_motions, dim=1)[:, :seq_length]
        if with_gtmotion:
            min_length = min(batch["motion"].shape[1], pred_motions.shape[1])
            shape_code = batch["shape"].expand(-1, min_length, -1)
            return pred_motions[:, :min_length], batch["motion"][:, :min_length], shape_code
        else:
            return pred_motions

    @torch.no_grad()
    def build_attn_mask(self, patch_nums):
        L = sum(patch_nums)
        d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(patch_nums)]).view(1, L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_idx = dT[:, 0].contiguous()
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L).contiguous()
        zero_shape = list(attn_bias_for_masking.shape)
        zero_shape[-1] = patch_nums[-1]
        zero_attn_bias_for_masking = attn_bias_for_masking.new_zeros(attn_bias_for_masking.shape)
        zero_attn_bias_for_masking = zero_attn_bias_for_masking.repeat(1, 1, 1, self.prev_ratio)
        attn_bias_for_masking = torch.cat([zero_attn_bias_for_masking, attn_bias_for_masking], dim=-1)
        return attn_bias_for_masking, lvl_idx


class AdaLNBeforeHead(nn.Module):
    def __init__(self, embed_dim, cond_dim):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = embed_dim, cond_dim
        self.ln_wo_grad = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(cond_dim, 2*embed_dim))
    
    def forward(self, feat, cond_BD):
        batch_size, cond_len = feat.shape[0], cond_BD.shape[1]
        scale, shift = self.ada_lin(cond_BD).view(batch_size, cond_len, 2, -1).unbind(2)
        return self.ln_wo_grad(feat).mul(scale.add(1)).add_(shift)


def sample_with_top_k_top_p_(logits_BLV, top_k=2, top_p=0.95, num_samples=1):  # return idx, shaped (B, L)
    B, L, V = logits_BLV.shape
    if top_k > 0:
        idx_to_remove = logits_BLV < logits_BLV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BLV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BLV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BLV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BLV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=None).view(B, L, num_samples)[:, :, 0]

