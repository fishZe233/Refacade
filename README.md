<h1 align="center">
  <span style="color:#2196f3;"><b>Refaçade</b></span>: Editing Object with Given Reference Texture
</h1>

<p align="center">
    Youze Huang<sup>1,*</sup> 
    Penghui Ruan<sup>2,*</sup> 
    Bojia Zi<sup>3,*</sup> 
    Xianbiao Qi<sup>4,†</sup> 
    Jianan Wang<sup>5</sup> 
    Rong Xiao<sup>4</sup> <br>
  <sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.
</p>
<p align="center">
    <span><sup>1</sup> University of Electronic Science and Technology of China</span>&emsp;
    <span><sup>2</sup> The Hong Kong Polytechnic University</span><br>
    <span><sup>3</sup> The Chinese University of Hong Kong</span>&emsp;
    <span><sup>4</sup> IntelliFusion Inc.</span>&emsp;
    <span><sup>5</sup> Astribot Inc.</span>
</p>

<p align="center">
  <a href="https://huggingface.co/fishze/Refacade"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/fishZe233/Refacade"><img alt="Github" src="https://img.shields.io/badge/Refaçade-github-black"></a>
  <a href="https://huggingface.co/spaces/Ryan-PR/Refacade"><img alt="Huggingface Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Space-1e90ff"></a>
  <a href="https://refacade.github.io/"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</p>

https://github.com/user-attachments/assets/e1e53908-2c78-4433-947d-11d124a4dd32

## 🚀 Overview

**Refaçade** is a unified image–video retexturing model built upon the Wan2.1-based VACE framework. It edits the surface material of specified objects in a video using user-provided reference textures, while preserving the original geometry and background. We use **Jigsaw Permutation** to decouple structural information in the reference image and a **Texture Remover** to disentangle the original object’s appearance. This functionality enables users to explore diverse possibilities effectively.

<p align="center">
  <img src="assets/pipe.png" alt="Refaçade illustration" width="90%">
</p>

---

## 🛠️ Installation

Our project is built upon [Wan2.1-based VACE](https://github.com/ali-vilab/VACE).

```bash
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1
```

---

## 🏃‍♂️ Gradio Demo

You can use this gradio demo to retexture objects. Note that you don't need to compile the SAM2.
```bash
python app.py
```

---

## 📂 Download

```shell
huggingface-cli download --resume-download fishze/Refacade --local-dir models
```

We recommend to organize local directories as:
```angular2html
Refacade
├── ...
├── examples
├── models
│   ├── refacade
│   │   └── ...
│   ├── texture-remover
│   │   └── ...
│   └── vae
│       └── ...
├── sam2
└── ...
```

---

## ⚡ Quick Start

### Minimal Example

```bash
python test_pipe.py \
  --ref_img    ./assets/single_example/1.png \
  --ref_mask   ./assets/single_example/mask.png \
  --video_path ./assets/single_example/1.mp4 \
  --mask_path  ./assets/single_example/mask.mp4 \
  --output_dir ./outputs
```

---

