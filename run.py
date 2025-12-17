import argparse
import os
from typing import Tuple

import torch
from PIL import Image

from nets.funiegan import GeneratorFunieGAN as FunieGenerator


def load_generator(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = FunieGenerator()
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def read_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np
    arr = np.array(img)
    arr = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return arr.unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).clamp(0, 1)
    t = (t * 255.0).byte()
    t = t.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(t, mode="RGB")


def enhance_image(input_path: str, output_path: str, weights_path: str) -> Tuple[str, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(weights_path, device)
    img = read_image(input_path)
    
    # Ensure dimensions are divisible by 32 for U-Net architecture
    w, h = img.size
    new_w = ((w + 31) // 32) * 32
    new_h = ((h + 31) // 32) * 32
    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
        original_size = (w, h)
    else:
        original_size = None
    
    x = pil_to_tensor(img).to(device)
    with torch.no_grad():
        y = model(x)
    out = tensor_to_pil(y)
    
    # Resize back to original if needed
    if original_size:
        out = out.resize(original_size, Image.LANCZOS)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.save(output_path)
    return input_path, output_path


def parse_args():
    p = argparse.ArgumentParser(description="Enhance underwater image with FUnIE-GAN (PyTorch)")
    p.add_argument("input", help="Path to input image")
    p.add_argument("output", nargs="?", default=None, help="Path to save enhanced image (default: output/<input_filename>)")
    p.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "models", "funie_generator.pth"), help="Path to generator weights (.pth)")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output
    if output_path is None:
        input_basename = os.path.basename(args.input)
        output_path = os.path.join("output", input_basename)
    enhance_image(args.input, output_path, args.weights)
    print(f"Enhanced image saved to: {output_path}")


if __name__ == "__main__":
    main()
