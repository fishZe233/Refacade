# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from wan.modules.model import WanModel, WanAttentionBlock, sinusoidal_embedding_1d
from .mm_attention import MMModel, MMAttentionBlock
import torch.utils.checkpoint

def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            model_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output


class VaceWanAttentionBlock(MMAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            dim_ref,
            ffn_dim,
            ffn_dim_ref,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, dim_ref, ffn_dim, ffn_dim_ref, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
            self.before_proj_ref = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj_ref.weight)
            nn.init.zeros_(self.before_proj_ref.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, c_ref, x, x_ref, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            c_ref = self.before_proj_ref(c_ref) + x_ref
        c, c_ref = super().forward(c, c_ref, **kwargs)
        c_skip = self.after_proj(torch.cat([c_ref, c], dim=1))
        return c, c_ref, c_skip
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
    
    
class VaceMMModel(MMModel):
    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 ref_in_dim=None,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 dim_ref=2048,
                 ffn_dim=8192,
                 ffn_dim_ref=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, dim_ref, ffn_dim, ffn_dim_ref, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim
        self.ref_in_dim = ref_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.dim_ref, self.ffn_dim, self.ffn_dim_ref, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.ref_patch_embedding = nn.Conv3d(
            self.ref_in_dim, self.dim_ref, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward_vace(
        self,
        x,
        x_ref,
        vace_context,
        ref_context,
        seq_len,
        seq_len_ref,
        kwargs,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        # embeddings
        c = self.vace_patch_embedding(vace_context)
        c = c.flatten(2).transpose(1, 2)
        # c = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
        #               dim=1) for u in c
        # ])
        
        c_ref = self.ref_patch_embedding(ref_context)
        c_ref = c_ref.flatten(2).transpose(1, 2)
        # c_ref = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len_ref - u.size(1), u.size(2))],
        #               dim=1) for u in c_ref
        # ])

        new_kwargs = dict(x=x, x_ref=x_ref)
        new_kwargs.update(kwargs)
        hints = []
        for block in self.vace_blocks:
            c, c_ref, c_skip = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c,
                c_ref,
                **new_kwargs,
            )
            if c_skip is not None:
                hints.append(c_skip)
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        ref_context,
        context,
        seq_len,
        seq_len_ref,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    ):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
            self.freqs_ref = self.freqs_ref.to(device)

        # embeddings
        
        grid_sizes = torch.tensor([x.shape[2]/self.patch_size[0], x.shape[3]/self.patch_size[1], x.shape[4]/self.patch_size[2]]).long().unsqueeze(0).repeat(x.shape[0], 1)
        seq_lens = torch.tensor(
            (x.shape[2]//self.patch_size[0])
            * (x.shape[3]//self.patch_size[1])
            * (x.shape[4]//self.patch_size[2])
        ).repeat(x.shape[0]).long()  
        x_ref = self.patch_embedding(x[:, :, :1, :, :])     
        x_ref = x_ref.flatten(2).transpose(1, 2) 
        x_vid = self.patch_embedding(x[:, :, 1:, :, :])     
        x_vid = x_vid.flatten(2).transpose(1, 2)
        # print(x.dtype)
        grid_sizes_ref = torch.tensor([ref_context.shape[2]/self.patch_size[0], ref_context.shape[3]/self.patch_size[1], ref_context.shape[4]/self.patch_size[2]]).long().unsqueeze(0).repeat(grid_sizes.shape[0], 1)
        
        seq_lens_ref = torch.tensor(
            (ref_context.shape[2]//self.patch_size[0])
            * (ref_context.shape[3]//self.patch_size[1])
            * (ref_context.shape[4]//self.patch_size[2])
        ).repeat(x.shape[0]).long()
        assert seq_lens.max() <= seq_len
        # x = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
        #               dim=1) for u in x
        # ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            e_ref = self.time_embedding_ref(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0_ref = self.time_projection_ref(e_ref).unflatten(1, (6, self.dim))

        # text embedding
        context_lens = None
        context = self.text_embedding(context)

        kwargs = dict(
            e=e0,
            e_ref=e0_ref,
            seq_lens=seq_lens,
            seq_lens_ref=seq_lens_ref,
            grid_sizes=grid_sizes,
            grid_sizes_ref=grid_sizes_ref,
            freqs=self.freqs,
            freqs_ref=self.freqs_ref,
            context=context,
            context_lens=context_lens,
        )
        # ========== 支路 VACE ==========
        hints = self.forward_vace(
            x_vid, x_ref, vace_context, ref_context, seq_len, seq_len_ref, kwargs,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale
        kwargs.pop("e_ref")
        kwargs.pop("seq_lens_ref")
        kwargs.pop("grid_sizes_ref")
        kwargs.pop("freqs_ref")

        # ========== 主路 BLOCKS ==========
        x = torch.cat([x_ref, x_vid], dim=1)
        for block in self.blocks:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
                **kwargs,
            )

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
