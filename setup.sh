(
  echo "Downloading SAM 2 checkpoints..."
  cd checkpoints
  ./download_checkpoints.sh
)

(
  echo "Downloading DINO checkpoints..."
  cd gdino_checkpoints
  ./download_checkpoints.sh
)

#pip install torch torchvision torchaudio transformers

pip install -e .
pip install --no-build-isolation -e grounding_dino


