import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import os
from tqdm import tqdm

# 配置
base_model = "runwayml/stable-diffusion-v1-5"
controlnet_path = "./checkpoints/cityscapes_controlnet" # 训练好的路径
label_dir = "/media/HDD0/wzl/mmcls/DLandCV-course/homework2/pytorch-CycleGAN-and-pix2pix-master/datasets/cityscapes_unpaired/valA"       # 验证集 Label
output_dir = "./results/cityscapes_exp5_controlnet"
os.makedirs(output_dir, exist_ok=True)

# 1. 加载模型
print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")

# 2. 推理
print("Generating images...")
for filename in tqdm(os.listdir(label_dir)):
    if not (filename.endswith(".jpg") or filename.endswith(".png")): continue
    
    # 读取 Label
    label_path = os.path.join(label_dir, filename)
    image = Image.open(label_path).convert("RGB").resize((512, 512))
    
    # 生成
    # Seed 很重要，固定 Seed 可以复现
    generator = torch.manual_seed(42)
    output = pipe(
        "high quality realistic city street view, driving scene, 4k",
        negative_prompt="blur, low quality, cartoon, painting, watermark",
        image=image,
        num_inference_steps=20,
        generator=generator,
    ).images[0]
    
    # 保存
    output.save(os.path.join(output_dir, filename.replace(".jpg", "_fake_B.png")))

print("Done!")