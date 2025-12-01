from pipeline import RefacadePipeline
from vace.models.wan.modules.model_mm import VaceMMModel
from vace.models.wan.modules.model_tr import VaceWanModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from wan.text2video import FlowUniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
from vae import WanVAE
import os
import torch

vae = WanVAE(
    vae_pth="./models/vae/Wan2.1_VAE.pth",
    dtype=torch.float16,
)
vae.to(torch.float16)
device="cuda:0"
texture_remover = VaceWanModel.from_config("./models/texture_remover/texture_remover.json")
ckpt = torch.load("./models/texture_remover/texture_remover.pth", map_location="cpu")
texture_remover.load_state_dict(ckpt)
texture_remover = texture_remover.to(dtype=torch.float16, device=device)
model = VaceMMModel.from_config("./models/refacade/refacade.json")
ckpt = torch.load("./models/refacade/refacade.pth", map_location="cpu")
model.load_state_dict(ckpt)
model = model.to(dtype=torch.float16, device=device)
sample_scheduler = FlowUniPCMultistepScheduler(
    num_train_timesteps=1000,
    shift=1,
)
pipe = RefacadePipeline(vae=vae, transformer=model, texture_remover=texture_remover, scheduler=sample_scheduler)

pipe.to("cuda")


ref_img = load_image("xx.png")
ref_mask = load_image("xx-mask.png")
video_path = "xx.mp4"
mask_path = "xx_mask.mp4"

height = 480
width = 832
num_frames = 81
seed = 42

output, mesh, ref = pipe(
     video=load_video(video_path),
     mask=load_video(mask_path),
     reference_image=ref_img,
     reference_mask=ref_mask,
     height=height,
     width=width,
     num_frames=num_frames,
     num_inference_steps=20,
     guidance_scale=1.5,
     reference_patch_ratio=0.1,
     generator=torch.Generator().manual_seed(seed),
     return_dict=False,
     )
export_to_video(output[0], "output.mp4", fps=16)
export_to_video(mesh[0], "mesh.mp4", fps=16)
ref = Image.fromarray(ref)
ref.save("ref.png")