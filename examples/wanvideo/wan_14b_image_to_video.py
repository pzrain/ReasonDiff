import torch
from diffsynth import ModelManager, WanVideoPipeline
from diffsynth.data import save_video
from PIL import Image
import os

lora_path = ''
rintermod_path = ''
assets = ['candle']

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)

model_manager.load_lora(lora_path, lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.rintermod.load_state_dict(
    { k[10:]:v for k, v in torch.load(rintermod_path).items() if 'rintermod' in k },
    strict=False
)
pipe.rintermod.to(pipe.device, dtype=pipe.torch_dtype)
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

for asset_name in assets:
    image_path = f"assets/{asset_name}.png"
    image = Image.open(image_path).convert('RGB')
    prompt = []
    cond_idx = None
    with open(f"assets/{asset_name}.txt") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("cond_idx:"):
                cond_idx = int(line.strip().split(":")[1])
            else:
                prompt.append(line.strip())
    num_frame = len(prompt)
    # Image-to-video
    video = pipe(
        prompt=prompt,
        negative_prompt="overbright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression artifact, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        input_image=image,
        num_inference_steps=50,
        tiled=True,
        num_frames=num_frame,
        cond_index=cond_idx,
        width=832,
        height=480,
        use_diff_reasoner=True,
    )
    output_path = f"output"
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f"video-{asset_name}.mp4")
    save_video(video, output_path, fps=15, quality=9)
    