
FUnIE-GAN (PyTorch Inference Only)

This subpackage is a minimal, inference-only version of FUnIE-GAN in PyTorch. It bundles pretrained weights and provides a simple CLI to enhance a single image from a path.

Usage

- Install dependencies: `pip install -r requirements.txt`
- Run enhancement: `python run.py /path/to/input.jpg [output_path]`
  - If no output path is specified, saves to `output/<input_filename>`

Notes

- The script defaults to using `models/funie_generator.pth`. You can override via `--weights`.
- CUDA will be used automatically if available; otherwise CPU.
