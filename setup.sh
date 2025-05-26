(
  echo "Downloading SAM 2 checkpoints..."
  cd checkpoints || exit 1
  bash download_ckpts.sh
)

(
  echo "Downloading DINO checkpoints..."
  cd gdino_checkpoints || exit 1
  bash download_ckpts.sh
)

pip install torch torchvision torchaudio transformers opencv-python supervision

pip install -e .
pip install --no-build-isolation -e grounding_dino


