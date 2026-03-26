from pathlib import Path
import torch
import yaml
from PIL import Image

CONFIG_PATH = "configs/z_image.yaml"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def resolve_dtype(dtype_str: str):
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype_map[dtype_str]


def load_pipeline(cfg: dict):
    model_id = cfg["model_id"]
    torch_dtype = resolve_dtype(cfg.get("dtype", "bfloat16"))

    prefer_specialized = cfg.get("prefer_specialized_pipeline", True)
    allow_fallback = cfg.get("allow_remote_code_fallback", True)

    if prefer_specialized:
        try:
            from diffusers import ZImagePipeline

            pipe = ZImagePipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
            )
            return pipe, "ZImagePipeline"
        except Exception as e:
            print(f"[WARN] ZImagePipeline load failed: {e}")

    if allow_fallback:
        try:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            return pipe, "DiffusionPipeline"
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pipeline with both ZImagePipeline and DiffusionPipeline: {e}"
            ) from e

    raise RuntimeError("No available pipeline loader configuration.")


def generate_image(pipe, cfg: dict, prompt: str) -> Image.Image:
    device = cfg.get("device", "cuda")
    negative_prompt = cfg.get("negative_prompt", "")
    height = int(cfg.get("height", 1024))
    width = int(cfg.get("width", 1024))
    num_inference_steps = int(cfg.get("num_inference_steps", 32))
    guidance_scale = float(cfg.get("guidance_scale", 4.0))
    seed = int(cfg.get("seed", 42))

    pipe = pipe.to(device)
    # seed 固定済みの乱数生成器
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    return result.images[0]


def save_image(image: Image.Image, output_path: str):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main():
    cfg = load_config(CONFIG_PATH)
    prompt = load_prompt(cfg["prompt_file"])

    pipe, pipeline_name = load_pipeline(cfg)
    print(f"[INFO] Loaded with: {pipeline_name}")

    image = generate_image(pipe, cfg, prompt)
    save_image(image, cfg["output_path"])


if __name__ == "__main__":
    main()