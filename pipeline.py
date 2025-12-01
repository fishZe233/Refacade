from typing import Any, Callable, Dict, List, Optional, Union
import PIL.Image
import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from PIL import Image

from vae import WanVAE
from vace.models.wan.modules.model_mm import VaceMMModel
from vace.models.wan.modules.model_tr import VaceWanModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from dataclasses import dataclass


@dataclass
class RefacadePipelineOutput(BaseOutput):
    frames: torch.Tensor
    meshes: torch.Tensor
    ref_img: torch.Tensor


logger = logging.get_logger(__name__)


@torch.no_grad()
def _pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect"):
    H, W = x.shape[-2], x.shape[-1]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    pad = (0, pad_w, 0, pad_h)
    if pad_h or pad_w:
        x = F.pad(x, pad, mode=mode)
    return x, pad


@torch.no_grad()
def _unpad(x: torch.Tensor, pad):
    l, r, t, b = pad
    H, W = x.shape[-2], x.shape[-1]
    return x[..., t:H - b if b > 0 else H, l:W - r if r > 0 else W]


@torch.no_grad()
def _resize(x: torch.Tensor, size: tuple, is_mask: bool):
    mode = "nearest" if is_mask else "bilinear"
    if is_mask:
        return F.interpolate(x, size=size, mode=mode)
    else:
        return F.interpolate(x, size=size, mode=mode, align_corners=False)


@torch.no_grad()
def _center_scale_foreground_to_canvas(
    x_f: torch.Tensor,   
    m_f: torch.Tensor,   
    target_hw: tuple,    
    bg_value: float = 1.0,
):
    C, H, W = x_f.shape
    H2, W2 = target_hw
    device = x_f.device
    ys, xs = (m_f > 0.5).nonzero(as_tuple=True)
    canvas = torch.full((C, H2, W2), bg_value, dtype=x_f.dtype, device=device)
    mask_canvas = torch.zeros((1, H2, W2), dtype=x_f.dtype, device=device)
    if ys.numel() == 0:
        return canvas, mask_canvas

    y0, y1 = ys.min().item(), ys.max().item()
    x0, x1 = xs.min().item(), xs.max().item()
    crop_img = x_f[:, y0:y1 + 1, x0:x1 + 1]
    crop_msk = m_f[y0:y1 + 1, x0:x1 + 1].unsqueeze(0)
    hc, wc = crop_msk.shape[-2], crop_msk.shape[-1]
    s = min(H2 / max(1, hc), W2 / max(1, wc))
    Ht = max(1, min(H2, int(math.floor(hc * s))))
    Wt = max(1, min(W2, int(math.floor(wc * s))))
    crop_img_up = _resize(crop_img.unsqueeze(0), (Ht, Wt), is_mask=False).squeeze(0)
    crop_msk_up = _resize(crop_msk.unsqueeze(0), (Ht, Wt), is_mask=True).squeeze(0)
    crop_msk_up = (crop_msk_up > 0.5).to(crop_msk_up.dtype)

    top = (H2 - Ht) // 2
    left = (W2 - Wt) // 2
    canvas[:, top:top + Ht, left:left + Wt] = crop_img_up
    mask_canvas[:, top:top + Ht, left:left + Wt] = crop_msk_up
    return canvas, mask_canvas


@torch.no_grad()
def _sample_patch_size_from_hw(
    H: int,
    W: int,
    ratio: float = 0.2,
    min_px: int = 16,
    max_px: Optional[int] = None,
) -> int:
    r = ratio
    raw = r * min(H, W)
    if max_px is None:
        max_px = min(192, min(H, W))
    P = int(round(raw))
    P = max(min_px, min(P, max_px))
    P = int(P)
    return P


@torch.no_grad()
def _masked_patch_pack_to_center_rectangle(
    x_f: torch.Tensor,     
    m_f: torch.Tensor,     
    patch: int,
    fg_thresh: float = 0.8,
    bg_value: float = 1.0,
    min_patches: int = 4,
    flip_prob: float = 0.5,
    use_morph_erode: bool = False,
):

    C, H, W = x_f.shape
    device = x_f.device
    P = int(patch)

    x_pad, pad = _pad_to_multiple(x_f, P, mode="reflect")
    l, r, t, b = pad
    H2, W2 = x_pad.shape[-2], x_pad.shape[-1]
    m_pad = F.pad(m_f.unsqueeze(0).unsqueeze(0), (l, r, t, b), mode="constant", value=0.0).squeeze(0)

    cs_img, cs_msk = _center_scale_foreground_to_canvas(x_pad, m_pad.squeeze(0), (H2, W2), bg_value)
    if (cs_msk > 0.5).sum() == 0:
        out_img = _unpad(cs_img, pad).clamp_(-1, 1)
        out_msk = _unpad(cs_msk, pad).clamp_(0, 1)
        return out_img, out_msk, True

    m_eff = cs_msk
    if use_morph_erode:
        erode_px = int(max(1, min(6, round(P * 0.03))))
        m_eff = 1.0 - F.max_pool2d(1.0 - cs_msk, kernel_size=2 * erode_px + 1, stride=1, padding=erode_px)

    x_pad2, pad2 = _pad_to_multiple(cs_img, P, mode="reflect")
    m_pad2 = F.pad(m_eff, pad2, mode="constant", value=0.0)
    H3, W3 = x_pad2.shape[-2], x_pad2.shape[-1]

    m_pool = F.avg_pool2d(m_pad2, kernel_size=P, stride=P).view(-1)

    base_thr = float(fg_thresh)
    thr_candidates = [base_thr, max(base_thr - 0.05, 0.75), max(base_thr - 0.10, 0.60)]

    x_unf = F.unfold(x_pad2.unsqueeze(0), kernel_size=P, stride=P)
    N = x_unf.shape[-1]

    sel = None
    for thr in thr_candidates:
        idx = (m_pool >= (thr - 1e-6)).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() >= min_patches:
            sel = idx
            break
    if sel is None:
        img_fallback = _unpad(_unpad(cs_img, pad2), pad).clamp_(-1, 1)
        msk_fallback = _unpad(_unpad(cs_msk, pad2), pad).clamp_(0, 1)
        return img_fallback, msk_fallback, True

    sel = sel.to(device=device, dtype=torch.long)
    sel = sel[(sel >= 0) & (sel < N)]
    if sel.numel() == 0:
        img_fallback = _unpad(_unpad(cs_img, pad2), pad).clamp_(-1, 1)
        msk_fallback = _unpad(_unpad(cs_msk, pad2), pad).clamp_(0, 1)
        return img_fallback, msk_fallback, True

    perm = torch.randperm(sel.numel(), device=device, dtype=torch.long)
    sel = sel[perm]
    chosen_x = x_unf[:, :, sel]
    K = chosen_x.shape[-1]
    if K == 0:
        img_fallback = _unpad(_unpad(cs_img, pad2), pad).clamp_(-1, 1)
        msk_fallback = _unpad(_unpad(cs_msk, pad2), pad).clamp_(0, 1)
        return img_fallback, msk_fallback, True

    if flip_prob > 0:
        cx4 = chosen_x.view(1, C, P, P, K)
        do_flip = (torch.rand(K, device=device) < flip_prob)
        coin = (torch.rand(K, device=device) < 0.5)
        flip_h = do_flip & coin
        flip_v = do_flip & (~coin)
        if flip_h.any():
            cx4[..., flip_h] = cx4[..., flip_h].flip(dims=[3])
        if flip_v.any():
            cx4[..., flip_v] = cx4[..., flip_v].flip(dims=[2])
        chosen_x = cx4.view(1, C * P * P, K)

    max_cols = max(1, W3 // P)
    max_rows = max(1, H3 // P)
    capacity = max_rows * max_cols
    K_cap = min(K, capacity)
    cols = int(max(1, min(int(math.floor(math.sqrt(K_cap))), max_cols)))
    rows_full = min(max_rows, K_cap // cols)
    K_used = rows_full * cols
    if K_used == 0:
        img_fallback = _unpad(_unpad(cs_img, pad2), pad).clamp_(-1, 1)
        msk_fallback = _unpad(_unpad(cs_msk, pad2), pad).clamp_(0, 1)
        return img_fallback, msk_fallback, True

    chosen_x = chosen_x[:, :, :K_used]
    rect_unf = torch.full((1, C * P * P, rows_full * cols), bg_value, device=device, dtype=x_f.dtype)
    rect_unf[:, :, :K_used] = chosen_x
    rect = F.fold(rect_unf, output_size=(rows_full * P, cols * P), kernel_size=P, stride=P).squeeze(0)

    ones_patch = torch.ones((1, 1 * P * P, K_used), device=device, dtype=x_f.dtype)
    mask_rect_unf = torch.zeros((1, 1 * P * P, rows_full * cols), device=device, dtype=x_f.dtype)
    mask_rect_unf[:, :, :K_used] = ones_patch
    rect_mask = F.fold(mask_rect_unf, output_size=(rows_full * P, cols * P), kernel_size=P, stride=P).squeeze(0)

    Hr, Wr = rect.shape[-2], rect.shape[-1]
    s = min(H3 / max(1, Hr), W3 / max(1, Wr))
    Ht = min(max(1, int(math.floor(Hr * s))), H3)
    Wt = min(max(1, int(math.floor(Wr * s))), W3)

    rect_up = _resize(rect.unsqueeze(0), (Ht, Wt), is_mask=False).squeeze(0)
    rect_mask_up = _resize(rect_mask.unsqueeze(0), (Ht, Wt), is_mask=True).squeeze(0)

    canvas_x = torch.full((C, H3, W3), bg_value, device=device, dtype=x_f.dtype)
    canvas_m = torch.zeros((1, H3, W3), device=device, dtype=x_f.dtype)
    top, left = (H3 - Ht) // 2, (W3 - Wt) // 2
    canvas_x[:, top:top + Ht, left:left + Wt] = rect_up
    canvas_m[:, top:top + Ht, left:left + Wt] = rect_mask_up

    out_img = _unpad(_unpad(canvas_x, pad2), pad).clamp_(-1, 1)
    out_msk = _unpad(_unpad(canvas_m, pad2), pad).clamp_(0, 1)
    return out_img, out_msk, False


@torch.no_grad()
def _compose_centered_foreground(x_f: torch.Tensor, m_f3: torch.Tensor, target_hw: Tuple[int, int], bg_value: float = 1.0):
    m_bin = (m_f3 > 0.5).float().mean(dim=0)
    m_bin = (m_bin > 0.5).float()
    return _center_scale_foreground_to_canvas(x_f, m_bin, target_hw, bg_value)

class RefacadePipeline(DiffusionPipeline, WanLoraLoaderMixin):

    model_cpu_offload_seq = "texture_remover->transformer->vae"

    def __init__(
        self,
        vae,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: VaceMMModel = None,
        texture_remover: VaceWanModel = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            texture_remover=texture_remover,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.empty_embedding = torch.load(
            "./text_embedding/empty.pt",
            map_location="cpu"
        )
        self.negative_embedding = torch.load(
            "./text_embedding/negative.pt",
            map_location="cpu"
        )

    def vace_encode_masks(self, masks: torch.Tensor):
        masks = masks[:, :1, :, :, :]
        B, C, D, H, W = masks.shape
        patch_h, patch_w = self.vae_scale_factor_spatial, self.vae_scale_factor_spatial
        stride_t = self.vae_scale_factor_temporal
        patch_count = patch_h * patch_w
        new_D = (D + stride_t - 1) // stride_t
        new_H = 2 * (H // (patch_h * 2))
        new_W = 2 * (W // (patch_w * 2))
        masks = masks[:, 0]
        masks = masks.view(B, D, new_H, patch_h, new_W, patch_w)
        masks = masks.permute(0, 3, 5, 1, 2, 4)
        masks = masks.reshape(B, patch_count, D, new_H, new_W)
        masks = F.interpolate(
            masks,
            size=(new_D, new_H, new_W),
            mode="nearest-exact"
        )
        return masks

    def preprocess_conditions(
        self,
        video: Optional[List[PipelineImageInput]] = None,
        mask: Optional[List[PipelineImageInput]] = None,
        reference_image: Optional[PIL.Image.Image] = None,
        reference_mask: Optional[PIL.Image.Image] = None,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        reference_patch_ratio: float = 0.2,
        fg_thresh: float = 0.9,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):

        base = self.vae_scale_factor_spatial * 2
        video_height, video_width = self.video_processor.get_default_height_width(video[0])
        
        if video_height * video_width > height * width:
            scale_w = width / video_width
            scale_h = height / video_height
            video_height, video_width = int(video_height * scale_h), int(video_width * scale_w)

        if video_height % base != 0 or video_width % base != 0:
            logger.warning(
                f"Video height and width should be divisible by {base}, but got {video_height} and {video_width}. "
            )
            video_height = (video_height // base) * base
            video_width = (video_width // base) * base

        assert video_height * video_width <= height * width

        video = self.video_processor.preprocess_video(video, video_height, video_width)
        image_size = (video_height, video_width)
    
        mask = self.video_processor.preprocess_video(mask, video_height, video_width)
        mask = torch.clamp((mask + 1) / 2, min=0, max=1)

        video = video.to(dtype=dtype, device=device)
        mask = mask.to(dtype=dtype, device=device)

        if reference_image is None:
            raise ValueError("reference_image must be provided when using IMAGE_CONTROL mode.")

        if isinstance(reference_image, (list, tuple)):
            ref_img_pil = reference_image[0]
        else:
            ref_img_pil = reference_image

        if reference_mask is not None and isinstance(reference_mask, (list, tuple)):
            ref_mask_pil = reference_mask[0]
        else:
            ref_mask_pil = reference_mask

        ref_img_t = self.video_processor.preprocess(ref_img_pil, image_size[0], image_size[1])
        if ref_img_t.dim() == 4 and ref_img_t.shape[0] == 1:
            ref_img_t = ref_img_t[0]  
        if ref_img_t.shape[0] == 1:
            ref_img_t = ref_img_t.repeat(3, 1, 1)
        ref_img_t = ref_img_t.to(dtype=dtype, device=device)  

        H, W = image_size
        if ref_mask_pil is not None:
            if not isinstance(ref_mask_pil, Image.Image):
                ref_mask_pil = Image.fromarray(np.array(ref_mask_pil))
            ref_mask_pil = ref_mask_pil.convert("L")
            ref_mask_pil = ref_mask_pil.resize((W, H), Image.NEAREST)
            mask_arr = np.array(ref_mask_pil)  
            m = torch.from_numpy(mask_arr).float() / 255.0  
            m = (m > 0.5).float()  
            ref_msk3 = m.unsqueeze(0).repeat(3, 1, 1)  
        else:
            ref_msk3 = torch.ones(3, H, W, dtype=dtype)

        ref_msk3 = ref_msk3.to(dtype=dtype, device=device)

        if math.isclose(reference_patch_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            cs_img, cs_m = _compose_centered_foreground(
                x_f=ref_img_t,     
                m_f3=ref_msk3,     
                target_hw=image_size,
                bg_value=1.0,    
            )
            ref_img_out = cs_img
            ref_mask_out = cs_m  
        else:
            patch = _sample_patch_size_from_hw(
                H=image_size[0],
                W=image_size[1],
                ratio=reference_patch_ratio,
            )

            m_bin = (ref_msk3 > 0.5).float().mean(dim=0)
            m_bin = (m_bin > 0.5).float()
            reshuffled, reshuf_mask, used_fb = _masked_patch_pack_to_center_rectangle(
                x_f=ref_img_t,   
                m_f=m_bin,       
                patch=patch,
                fg_thresh=fg_thresh,
                bg_value=1.0,
                min_patches=4,
            )

            ref_img_out = reshuffled
            ref_mask_out = reshuf_mask

        B = video.shape[0]
        if batch_size is not None:
            B = batch_size

        ref_image = ref_img_out.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1, -1).contiguous()
        ref_mask = ref_mask_out.unsqueeze(0).unsqueeze(2).expand(B,  3, -1, -1, -1).contiguous()

        ref_image = ref_image.to(dtype=dtype, device=device)
        ref_mask = ref_mask.to(dtype=dtype, device=device)

        return video[:, :, :num_frames], mask[:, :, :num_frames], ref_image, ref_mask

    @torch.no_grad()
    def texture_remove(self, foreground_latent):
        sample_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1)
        text_embedding = torch.zeros(
            [256, 4096],
            device=foreground_latent.device,
            dtype=foreground_latent.dtype
        )
        context = text_embedding.unsqueeze(0).expand(
            foreground_latent.shape[0], -1, -1
        ).to(foreground_latent.device)
        sample_scheduler.set_timesteps(3, device=foreground_latent.device)
        timesteps = sample_scheduler.timesteps
        noise = torch.randn_like(
            foreground_latent,
            dtype=foreground_latent.dtype,
            device=foreground_latent.device
        )
        seq_len = math.ceil(
            noise.shape[2] * noise.shape[3] * noise.shape[4] / 4
        )
        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for _, t in enumerate(timesteps):
                timestep = torch.stack([t]).to(foreground_latent.device)
                noise_pred_cond = self.texture_remover(
                    latents,
                    t=timestep,
                    vace_context=foreground_latent,
                    vace_context_scale=1,
                    **arg_c
                )[0]
                temp_x0 = sample_scheduler.step(
                    noise_pred_cond, t, latents, return_dict=False
                )[0]
                latents = temp_x0
        return latents
    
    def dilate_mask_hw(self, mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
        B, C, F_, H, W = mask.shape
        k = 2 * radius + 1
        mask_2d = mask.permute(0, 2, 1, 3, 4).reshape(B * F_, C, H, W)
        kernel = torch.ones(
            (C, 1, k, k),
            device=mask.device,
            dtype=mask.dtype
        )
        dilated_2d = F.conv2d(
            mask_2d,
            weight=kernel,
            bias=None,
            stride=1,
            padding=radius,
            groups=C
        )
        dilated_2d = (dilated_2d > 0).to(mask.dtype)
        dilated = dilated_2d.view(B, F_, C, H, W).permute(0, 2, 1, 3, 4)
        return dilated
    
    def prepare_vace_latents(
        self,
        dilate_radius: int,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_image: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device

        vae_dtype = self.vae.dtype
        video = video.to(dtype=vae_dtype)
        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        mask_clone = mask.clone()
        mask = self.dilate_mask_hw(mask, dilate_radius)
        inactive = video * (1 - mask)
        reactive = video * mask_clone
        reactive_latent = self.vae.encode(reactive)
        mesh_latent = self.texture_remove(reactive_latent)       
        
        inactive_latent = self.vae.encode(inactive)
        ref_latent      = self.vae.encode(reference_image)
        neg_ref_latent  = self.vae.encode(torch.ones_like(reference_image))
        
        reference_mask = torch.where(reference_mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        mask = self.vace_encode_masks(mask)
        ref_mask = self.vace_encode_masks(reference_mask)

        return inactive_latent, mesh_latent, ref_latent, neg_ref_latent, mask, ref_mask


    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @torch.no_grad()
    def __call__(
        self,
        video: Optional[PipelineImageInput] = None,
        mask: Optional[PipelineImageInput] = None,
        reference_image: Optional[PipelineImageInput] = None,
        reference_mask: Optional[PipelineImageInput] = None,
        conditioning_scale: float = 1.0,
        dilate_radius: int = 3,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        num_videos_per_prompt: Optional[int] = 1,
        reference_patch_ratio: float = 0.2,
        fg_thresh: float = 0.9,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
    ):

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)


        self._guidance_scale = guidance_scale

        device = self._execution_device
        batch_size = 1

        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        video, mask, reference_image, reference_mask = self.preprocess_conditions(
            video,
            mask,
            reference_image,
            reference_mask,
            batch_size,
            height,
            width,
            num_frames,
            reference_patch_ratio,
            fg_thresh,
            torch.float16,
            device,
        )

        inactive_latent, mesh_latent, ref_latent, neg_ref_latent, mask, ref_mask = self.prepare_vace_latents(dilate_radius, video, mask, reference_image, reference_mask, device)
        c = torch.cat([inactive_latent, mesh_latent, mask], dim=1)
        c1 = torch.cat([ref_latent, ref_mask], dim=1)
        c1_negative = torch.cat(
            [neg_ref_latent, torch.zeros_like(ref_mask)],
            dim=1
        )

        num_channels_latents = 16
        noise = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float16,
            device,
            generator,
            latents,
        )
        
        latents_cond = torch.cat([ref_latent, noise], dim=2)
        latents_uncond = torch.cat([neg_ref_latent, noise], dim=2)

        seq_len = math.ceil(
            latents_cond.shape[2] *
            latents_cond.shape[3] *
            latents_cond.shape[4] / 4
        )
        seq_len_ref = math.ceil(
            ref_latent.shape[2] *
            ref_latent.shape[3] *
            ref_latent.shape[4] / 4
        )
        context = self.empty_embedding.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        context_neg = self.negative_embedding.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        arg_c = {
            "context": context,
            "seq_len": seq_len,
            "seq_len_ref": seq_len_ref
        }
        arg_c_null = {
            "context": context_neg,
            "seq_len": seq_len,
            "seq_len_ref": seq_len_ref
        }

        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                timestep = t.expand(batch_size)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    noise_pred = self.transformer(
                        latents_cond,
                        t=timestep,
                        vace_context=c,
                        ref_context=c1,
                        vace_context_scale=conditioning_scale,
                        **arg_c,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond = self.transformer(
                            latents_uncond,
                            t=timestep,
                            vace_context=c,
                            ref_context=c1_negative,
                            vace_context_scale=0,
                            **arg_c_null,
                        )[0]
                        noise_pred = (noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)).unsqueeze(0)
                temp_x0 = self.scheduler.step(noise_pred[:, :, 1:],
                    t,
                    latents_cond[:, :, 1:],
                    return_dict=False)[0]
                latents_cond = torch.cat([ref_latent, temp_x0], dim=2)
                latents_uncond = torch.cat([neg_ref_latent, temp_x0], dim=2)
                progress_bar.update()


        self._current_timestep = None

        if not output_type == "latent":
            latents = temp_x0
            latents = latents.to(vae_dtype)
            video = self.vae.decode(latents)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
            mesh = self.vae.decode(mesh_latent.to(vae_dtype))
            mesh = self.video_processor.postprocess_video(mesh, output_type=output_type)
            ref_img = reference_image.cpu().squeeze(0).squeeze(1).permute(1, 2, 0).numpy()
            ref_img = ((ref_img+1)*255/2).astype(np.uint8)
        else:
            video = temp_x0
            mesh = mesh_latent
            ref_img = ref_latent

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, mesh, ref_img)

        return RefacadePipelineOutput(frames=video, meshes=mesh, ref_img=ref_img)
