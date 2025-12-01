import os
import time
import random

import gradio as gr
import cv2
import numpy as np
from PIL import Image

os.makedirs("./sam2/SAM2-Video-Predictor/checkpoints/", exist_ok=True)

from huggingface_hub import snapshot_download

def download_sam2():
    snapshot_download(
        repo_id="facebook/sam2-hiera-large",
        local_dir="./sam2/SAM2-Video-Predictor/checkpoints/",
    )
    print("Download sam2 completed")
    
def download_refacade():
    snapshot_download(
        repo_id="fishze/Refacade",
        local_dir="./models/",
    )
    print("Download refacade completed")


# download_sam2()

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import spaces
from pipeline import RefacadePipeline
from vace.models.wan.modules.model_mm import VaceMMModel
from vace.models.wan.modules.model_tr import VaceWanModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from wan.text2video import FlowUniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image, load_video
from vae import WanVAE

COLOR_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]

video_length = 201
W = 1024
H = W
device = "cuda"

def get_pipe_image_and_video_predictor():
    vae = WanVAE(
        vae_pth="./models/vae/Wan2.1_VAE.pth",
        dtype=torch.float16,
    )

    pipe_device = "cuda"

    texture_remover = VaceWanModel.from_config(
        "./models/texture_remover/texture_remover.json"
    )
    ckpt = torch.load(
        "./models/texture_remover/texture_remover.pth",
        map_location="cpu",
    )
    texture_remover.load_state_dict(ckpt)
    texture_remover = texture_remover.to(dtype=torch.float16, device=pipe_device)

    model = VaceMMModel.from_config(
        "./models/refacade/refacade.json"
    )
    ckpt = torch.load(
        "./models/refacade/refacade.pth",
        map_location="cpu",
    )
    model.load_state_dict(ckpt)
    model = model.to(dtype=torch.float16, device=pipe_device)

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
    pipe.to(pipe_device)

    sam2_checkpoint = "./sam2/SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"
    config = "sam2_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(config, sam2_checkpoint, device="cuda")
    model_sam = build_sam2(config, sam2_checkpoint, device="cuda")
    model_sam.image_size = 1024
    image_predictor = SAM2ImagePredictor(sam_model=model_sam)

    return pipe, image_predictor, video_predictor


def get_video_info(video_path, video_state):
    video_state["input_points"] = []
    video_state["scaled_points"] = []
    video_state["input_labels"] = []
    video_state["frame_idx"] = 0

    vr = VideoReader(video_path, ctx=cpu(0))
    first_frame = vr[0].asnumpy()
    del vr

    if first_frame.shape[0] > first_frame.shape[1]:
        W_ = W
        H_ = int(W_ * first_frame.shape[0] / first_frame.shape[1])
    else:
        H_ = H
        W_ = int(H_ * first_frame.shape[1] / first_frame.shape[0])

    first_frame = cv2.resize(first_frame, (W_, H_))
    video_state["origin_images"] = np.expand_dims(first_frame, axis=0)
    video_state["inference_state"] = None
    video_state["video_path"] = video_path
    video_state["masks"] = None
    video_state["painted_images"] = None
    image = Image.fromarray(first_frame)
    return image


def segment_frame(evt: gr.SelectData, label, video_state):
    if video_state["origin_images"] is None:
        return None
    x, y = evt.index
    new_point = [x, y]
    label_value = 1 if label == "Positive" else 0

    video_state["input_points"].append(new_point)
    video_state["input_labels"].append(label_value)
    height, width = video_state["origin_images"][0].shape[0:2]
    scaled_points = []
    for pt in video_state["input_points"]:
        sx = pt[0] / width
        sy = pt[1] / height
        scaled_points.append([sx, sy])

    video_state["scaled_points"] = scaled_points

    image_predictor.set_image(video_state["origin_images"][0])
    mask, _, _ = image_predictor.predict(
        point_coords=video_state["scaled_points"],
        point_labels=video_state["input_labels"],
        multimask_output=False,
        normalize_coords=False,
    )

    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (width, height))
    mask = mask[:, :, None]

    color = (
        np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32)
        / 255.0
    )
    color = color[None, None, :]
    org_image = video_state["origin_images"][0].astype(np.float32) / 255.0
    painted_image = (1 - mask * 0.5) * org_image + mask * 0.5 * color
    painted_image = np.uint8(np.clip(painted_image * 255, 0, 255))
    video_state["painted_images"] = np.expand_dims(painted_image, axis=0)
    video_state["masks"] = np.expand_dims(mask[:, :, 0], axis=0)

    for i in range(len(video_state["input_points"])):
        point = video_state["input_points"][i]
        if video_state["input_labels"][i] == 0:
            cv2.circle(painted_image, point, radius=3, color=(0, 0, 255), thickness=-1)
        else:
            cv2.circle(painted_image, point, radius=3, color=(255, 0, 0), thickness=-1)

    return Image.fromarray(painted_image)


def clear_clicks(video_state):
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["inference_state"] = None
    video_state["masks"] = None
    video_state["painted_images"] = None
    return (
        Image.fromarray(video_state["origin_images"][0])
        if video_state["origin_images"] is not None
        else None
    )


def set_ref_image(ref_img, ref_state):
    if ref_img is None:
        return None

    if isinstance(ref_img, Image.Image):
        img_np = np.array(ref_img)
    else:
        img_np = ref_img

    ref_state["origin_image"] = img_np
    ref_state["input_points"] = []
    ref_state["input_labels"] = []
    ref_state["scaled_points"] = []
    ref_state["mask"] = None

    return Image.fromarray(img_np)


def segment_ref_frame(evt: gr.SelectData, label, ref_state):
    if ref_state["origin_image"] is None:
        return None

    x, y = evt.index
    new_point = [x, y]
    label_value = 1 if label == "Positive" else 0

    ref_state["input_points"].append(new_point)
    ref_state["input_labels"].append(label_value)

    img = ref_state["origin_image"]
    h, w = img.shape[:2]

    scaled_points = []
    for pt in ref_state["input_points"]:
        sx = pt[0] / w
        sy = pt[1] / h
        scaled_points.append([sx, sy])
    ref_state["scaled_points"] = scaled_points

    image_predictor.set_image(img)
    mask, _, _ = image_predictor.predict(
        point_coords=scaled_points,
        point_labels=ref_state["input_labels"],
        multimask_output=False,
        normalize_coords=False,
    )

    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (w, h))
    mask = mask[:, :, None]
    ref_state["mask"] = mask[:, :, 0]

    color = (
        np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32)
        / 255.0
    )
    color = color[None, None, :]
    org_image = img.astype(np.float32) / 255.0
    painted = (1 - mask * 0.5) * org_image + mask * 0.5 * color
    painted = np.uint8(np.clip(painted * 255, 0, 255))

    for i in range(len(ref_state["input_points"])):
        point = ref_state["input_points"][i]
        if ref_state["input_labels"][i] == 0:
            cv2.circle(painted, point, radius=3, color=(0, 0, 255), thickness=-1)
        else:
            cv2.circle(painted, point, radius=3, color=(255, 0, 0), thickness=-1)

    return Image.fromarray(painted)


def clear_ref_clicks(ref_state):
    ref_state["input_points"] = []
    ref_state["input_labels"] = []
    ref_state["scaled_points"] = []
    ref_state["mask"] = None
    if ref_state["origin_image"] is None:
        return None
    return Image.fromarray(ref_state["origin_image"])


@spaces.GPU(duration=40)
def track_video(n_frames, video_state):
    input_points = video_state["input_points"]
    input_labels = video_state["input_labels"]
    frame_idx = video_state["frame_idx"]
    obj_id = video_state["obj_id"]
    scaled_points = video_state["scaled_points"]

    vr = VideoReader(video_state["video_path"], ctx=cpu(0))
    height, width = vr[0].shape[0:2]
    images = [vr[i].asnumpy() for i in range(min(len(vr), n_frames))]
    del vr

    if images[0].shape[0] > images[0].shape[1]:
        W_ = W
        H_ = int(W_ * images[0].shape[0] / images[0].shape[1])
    else:
        H_ = H
        W_ = int(H_ * images[0].shape[1] / images[0].shape[0])

    images = [cv2.resize(img, (W_, H_)) for img in images]
    video_state["origin_images"] = images
    images = np.array(images)

    sam2_checkpoint = "./sam2/SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"
    config = "sam2_hiera_l.yaml"
    video_predictor_local = build_sam2_video_predictor(
        config, sam2_checkpoint, device="cuda"
    )

    inference_state = video_predictor_local.init_state(
        images=images / 255, device="cuda"
    )

    if len(torch.from_numpy(video_state["masks"][0]).shape) == 3:
        mask = torch.from_numpy(video_state["masks"][0])[:, :, 0]
    else:
        mask = torch.from_numpy(video_state["masks"][0])

    video_predictor_local.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        mask=mask,
    )

    output_frames = []
    mask_frames = []
    color = (
        np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32)
        / 255.0
    )
    color = color[None, None, :]
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor_local.propagate_in_video(
        inference_state
    ):
        frame = images[out_frame_idx].astype(np.float32) / 255.0
        mask = np.zeros((H, W, 3), dtype=np.float32)
        for i, logit in enumerate(out_mask_logits):
            out_mask = logit.cpu().squeeze().detach().numpy()
            out_mask = (out_mask[:, :, None] > 0).astype(np.float32)
            mask += out_mask
        mask = np.clip(mask, 0, 1)
        mask = cv2.resize(mask, (W_, H_))
        mask_frames.append(mask)
        painted = (1 - mask * 0.5) * frame + mask * 0.5 * color
        painted = np.uint8(np.clip(painted * 255, 0, 255))
        output_frames.append(painted)

    video_state["masks"] = mask_frames
    video_file = f"/tmp/{time.time()}-{random.random()}-tracked_output.mp4"
    clip = ImageSequenceClip(output_frames, fps=15)
    clip.write_videofile(
        video_file, codec="libx264", audio=False, verbose=False, logger=None
    )
    print("Tracking done")
    return video_file, video_state


@spaces.GPU(duration=50)
def inference_and_return_video(
    dilate_radius,
    num_inference_steps,
    guidance_scale,
    ref_patch_ratio,
    fg_threshold,
    seed,
    video_state,
    ref_state,
):
    if video_state["origin_images"] is None or video_state["masks"] is None:
        print("No video frames or video masks.")
        return None, None, None

    if ref_state["origin_image"] is None or ref_state["mask"] is None:
        print("Reference image or reference mask missing.")
        return None, None, None

    images = video_state["origin_images"]
    masks = video_state["masks"]

    video_frames = []
    mask_frames = []
    for img, msk in zip(images, masks):
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        img_pil = Image.fromarray(img.astype(np.uint8))

        if isinstance(msk, np.ndarray):
            if msk.ndim == 3:
                m2 = msk[..., 0]
            else:
                m2 = msk
        else:
            m2 = np.asarray(msk)

        m2 = (m2 > 0.5).astype(np.uint8) * 255
        msk_pil = Image.fromarray(m2, mode="L")

        video_frames.append(img_pil)
        mask_frames.append(msk_pil)

    num_frames = len(video_frames)

    h0, w0 = images[0].shape[:2]
    if h0 > w0:
        height = 832
        width = 480
    else:
        height = 480
        width = 832

    ref_img_np = ref_state["origin_image"]
    ref_mask_np = ref_state["mask"]

    ref_img_pil = Image.fromarray(ref_img_np.astype(np.uint8))
    ref_mask_bin = (ref_mask_np > 0.5).astype(np.uint8) * 255
    ref_mask_pil = Image.fromarray(ref_mask_bin, mode="L")

    pipe.to("cuda")
    with torch.no_grad():
        retex_frames, mesh_frames, ref_img_out = pipe(
            video=video_frames,
            mask=mask_frames,
            reference_image=ref_img_pil,
            reference_mask=ref_mask_pil,
            conditioning_scale=1.0,
            height=height,
            width=width,
            num_frames=num_frames,
            dilate_radius=int(dilate_radius),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            reference_patch_ratio=float(ref_patch_ratio),
            fg_thresh=float(fg_threshold),
            generator=torch.Generator(device="cuda").manual_seed(seed),
            return_dict=False,
        )

        retex_frames_uint8 = (np.clip(retex_frames[0], 0.0, 1.0) * 255).astype(np.uint8)

        mesh_frames_uint8 = (np.clip(mesh_frames[0], 0.0, 1.0) * 255).astype(np.uint8)


        retex_output_frames = [frame for frame in retex_frames_uint8]
        mesh_output_frames = [frame for frame in mesh_frames_uint8]

        if ref_img_out.dtype != np.uint8:
            ref_img_out = (np.clip(ref_img_out, 0.0, 1.0) * 255).astype(np.uint8)

    retex_video_file = f"/tmp/{time.time()}-{random.random()}-refacade_output.mp4"
    retex_clip = ImageSequenceClip(retex_output_frames, fps=16)
    retex_clip.write_videofile(
        retex_video_file, codec="libx264", audio=False, verbose=False, logger=None
    )

    mesh_video_file = f"/tmp/{time.time()}-{random.random()}-mesh_output.mp4"
    mesh_clip = ImageSequenceClip(mesh_output_frames, fps=16)
    mesh_clip.write_videofile(
        mesh_video_file, codec="libx264", audio=False, verbose=False, logger=None
    )

    ref_image_to_show = ref_img_out

    return retex_video_file, mesh_video_file, ref_image_to_show


text = """
<div style='text-align:center; font-size:32px; font-family: Arial, Helvetica, sans-serif;'>
  Refaçade Video Retexture Demo
</div>
<div style='text-align:center; font-size:14px; color: #888; margin-top: 5px; font-family: Arial, Helvetica, sans-serif;'>
  Video mask from SAM2, Reference mask from SAM2 image clicks, RefacadePipeline for object retexture task.
</div>
"""

pipe, image_predictor, video_predictor = get_pipe_image_and_video_predictor()

with gr.Blocks() as demo:
    video_state = gr.State(
        {
            "origin_images": None,
            "inference_state": None,
            "masks": None,
            "painted_images": None,
            "video_path": None,
            "input_points": [],
            "scaled_points": [],
            "input_labels": [],
            "frame_idx": 0,
            "obj_id": 1,
        }
    )

    ref_state = gr.State(
        {
            "origin_image": None,
            "input_points": [],
            "input_labels": [],
            "scaled_points": [],
            "mask": None,
        }
    )

    gr.Markdown(f"<div style='text-align:center;'>{text}</div>")

    with gr.Column():
        video_input = gr.Video(label="Upload Video", elem_id="my-video1")
        get_info_btn = gr.Button("Extract First Frame", elem_id="my-btn")
        
        gr.Examples(
            examples=[
                ["./examples/1.mp4"],
                ["./examples/2.mp4"],
                ["./examples/3.mp4"],
                ["./examples/4.mp4"],
                ["./examples/5.mp4"],
                ["./examples/6.mp4"],
            ],
            inputs=[video_input],
            label="You can upload or choose a source video below to retexture.",
            elem_id="my-btn2"
        )

        image_output = gr.Image(
            label="First Frame Segmentation",
            interactive=True,
            elem_id="my-video",
        )

        demo.css = """
        #my-btn {
           width: 60% !important;
           margin: 0 auto;
        }
        #my-video1 {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #my-video {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #my-md {
           margin: 0 auto;
        }
        #my-btn2 {
            width: 60% !important;
            margin: 0 auto;
        }
        #my-btn2 button {
            width: 120px !important;
            max-width: 120px !important;
            min-width: 120px !important;
            height: 70px !important;
            max-height: 70px !important;
            min-height: 70px !important;
            margin: 8px !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            white-space: normal !important;
        }
        #my-btn3 {
            width: 60% !important;
            margin: 0 auto;
        }
        #ref_title {
            text-align: center;
        }
        #ref-image {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #ref-mask {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #mesh-row {
           width: 60% !important;
           margin: 0 auto;
        }
        """

        with gr.Row(elem_id="my-btn"):
            point_prompt = gr.Radio(
                ["Positive", "Negative"], label="Click Type", value="Positive"
            )
            clear_btn = gr.Button("Clear All Clicks")

        with gr.Row(elem_id="my-btn"):
            n_frames_slider = gr.Slider(
                minimum=1, maximum=201, value=81, step=1, label="Tracking Frames (4N+1)"
            )
            track_btn = gr.Button("Tracking")
        video_output = gr.Video(label="Tracking Result", elem_id="my-video")

        gr.Markdown("Reference Image & Mask (SAM2 Points)", elem_id="ref_title")

        ref_image_input = gr.Image(
            label="Upload Reference Image", elem_id="ref-image", interactive=True
        )
        gr.Examples(
            examples=[
                ["./examples/reference_image/1.png"],
                ["./examples/reference_image/2.png"],
                ["./examples/reference_image/3.png"],
                ["./examples/reference_image/4.png"],
                ["./examples/reference_image/5.png"],
                ["./examples/reference_image/6.png"],
                ["./examples/reference_image/7.png"],
                ["./examples/reference_image/8.png"],
                ["./examples/reference_image/9.png"],
            ],
            inputs=[ref_image_input],
            label="You can upload or choose a reference image below to retexture.",
            elem_id="my-btn3"
        )
        ref_image_display = gr.Image(
            label="Reference Mask Segmentation",
            elem_id="ref-mask",
            interactive=True,
        )

        with gr.Row(elem_id="my-btn"):
            ref_point_prompt = gr.Radio(
                ["Positive", "Negative"], label="Ref Click Type", value="Positive"
            )
            ref_clear_btn = gr.Button("Clear Ref Clicks")

        with gr.Column(elem_id="my-btn"):

            dilate_radius_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Mask Dilation Radius",
            )
            inference_steps_slider = gr.Slider(
                minimum=1,
                maximum=50,
                value=20,
                step=1,
                label="Num Inference Steps",
            )
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=3.0,
                value=1.5,
                step=0.1,
                label="Guidance Scale",
            )
            ref_patch_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.1,
                step=0.05,
                label="Reference Patch Ratio",
            )
            fg_threshold_slider = gr.Slider(
                minimum=0.7,
                maximum=1.0,
                value=0.8,
                step=0.01,
                label="Jigsaw Patches' Foreground Coverage Threshold",
            )
            seed_slider = gr.Slider(
                minimum=0,
                maximum=2147483647,
                value=42,
                step=1,
                label="Seed",                
            )

        remove_btn = gr.Button("Retexture", elem_id="my-btn")

        with gr.Row(elem_id="mesh-row"):
            mesh_video = gr.Video(label="Untextured Object")
            ref_image_final = gr.Image(
                label="Jigsawed Reference Image",
                interactive=False,
            )

        remove_video = gr.Video(label="Retexture Results", elem_id="my-video")

        remove_btn.click(
            inference_and_return_video,
            inputs=[
                dilate_radius_slider,
                inference_steps_slider,
                guidance_slider,
                ref_patch_slider,
                fg_threshold_slider,
                seed_slider,
                video_state,
                ref_state,
            ],
            outputs=[remove_video, mesh_video, ref_image_final],
        )

        get_info_btn.click(
            get_video_info,
            inputs=[video_input, video_state],
            outputs=image_output,
        )

        image_output.select(
            fn=segment_frame,
            inputs=[point_prompt, video_state],
            outputs=image_output,
        )

        clear_btn.click(clear_clicks, inputs=video_state, outputs=image_output)

        track_btn.click(
            track_video,
            inputs=[n_frames_slider, video_state],
            outputs=[video_output, video_state],
        )

        ref_image_input.change(
            set_ref_image,
            inputs=[ref_image_input, ref_state],
            outputs=ref_image_display,
        )
        ref_image_display.select(
            fn=segment_ref_frame,
            inputs=[ref_point_prompt, ref_state],
            outputs=ref_image_display,
        )
        ref_clear_btn.click(
            clear_ref_clicks, inputs=ref_state, outputs=ref_image_display
        )

demo.launch()

