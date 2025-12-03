import argparse
import os

import torch
from PIL import Image

from pipeline import RefacadePipeline
from vace.models.wan.modules.model_mm import VaceMMModel
from vace.models.wan.modules.model_tr import VaceWanModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from wan.text2video import FlowUniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image, load_video
from vae import WanVAE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RefaçadePipeline on a video with given reference texture."
    )
    parser.add_argument(
        "--ref_img",
        type=str,
        required=True,
        help="Path to the reference image (e.g. xx.png)",
    )
    parser.add_argument(
        "--ref_mask",
        type=str,
        required=True,
        help="Path to the reference mask image (e.g. xx-mask.png)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the source video (e.g. xx.mp4)",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the source video mask (e.g. xx_mask.mp4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output videos/images (default: ./outputs)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output video height (default: 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Output video width (default: 832)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames to generate (default: 81)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device to run on (default: "cuda:0")',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ====== Model & pipeline setup ======
    vae = WanVAE(
        vae_pth="./models/vae/Wan2.1_VAE.pth",
        dtype=torch.float16,
    )
    vae.to(torch.float16)

    device = args.device

    texture_remover = VaceWanModel.from_config(
        "./models/texture_remover/texture_remover.json"
    )
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

    pipe = RefacadePipeline(
        vae=vae,
        transformer=model,
        texture_remover=texture_remover,
        scheduler=sample_scheduler,
    )
    pipe.to(device)

    # ====== IO: load user-provided paths ======
    ref_img = load_image(args.ref_img)
    ref_mask = load_image(args.ref_mask)
    video_path = args.video_path
    mask_path = args.mask_path

    height = args.height
    width = args.width
    num_frames = args.num_frames
    seed = args.seed

    # ====== Inference ======
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

    # ====== Save outputs ======
    out_video_path = os.path.join(args.output_dir, "output.mp4")
    mesh_video_path = os.path.join(args.output_dir, "mesh.mp4")
    ref_img_path = os.path.join(args.output_dir, "ref.png")

    export_to_video(output[0], out_video_path, fps=16)
    export_to_video(mesh[0], mesh_video_path, fps=16)

    ref = Image.fromarray(ref)
    ref.save(ref_img_path)

    print(f"Saved edited video to: {out_video_path}")
    print(f"Saved mesh video to:   {mesh_video_path}")
    print(f"Saved ref image to:    {ref_img_path}")


if __name__ == "__main__":
    main()
