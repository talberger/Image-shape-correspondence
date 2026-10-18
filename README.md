# DISC

Dense image–shape correspondence: mapping every pixel of an object in a photo to a point on a 3D shape.
Work in progress (Tel Aviv University). Development branch: `analysis`.

Started from [BSB — Best Segmentation Buddies](https://github.com/threedle/bsb) (Lang et al., CVPR 2026); see `LICENSE`.

## Setup
- Env files: `env/requirements_disc_gpu.txt` (Linux + CUDA 11.3), `env/requirements_disc_mac.txt` (Apple Silicon, CPU only).
- Every session: `source env/activate.sh` (sets `DISC_ROOT` / `DISC_WS` and activates the `disc` env).
