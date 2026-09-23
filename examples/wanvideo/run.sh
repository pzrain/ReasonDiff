python examples/wanvideo/train_wan.py \
  --task data_process \
  --dataset_path <dataset_path> \
  --output_path ./models \
  --text_encoder_path "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
  --image_encoder_path "models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --vae_path "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 33 \
  --height 480 \
  --width 832

python examples/wanvideo/train_wan.py \
  --task train \
  --train_architecture lora \
  --dataset_path <dataset_path> \
  --output_path output-ckpt/ \
  --dit_path "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 2000 \
  --max_epochs 10 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing