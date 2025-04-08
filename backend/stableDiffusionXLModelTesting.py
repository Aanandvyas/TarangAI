from diffusers import DiffusionPipeline
import torch

path = r"F:\TarangAI - 23BCE11755\models\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\462165984030d82259a11f4367a4eed129e94a7b"

pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
prompt = "Mim Jong Un shanking hands with Modi"
# prompt = "Elon Musk and meloni handshake while modi is jelous behind"
images = pipe(prompt=prompt).images[0]
images.show()