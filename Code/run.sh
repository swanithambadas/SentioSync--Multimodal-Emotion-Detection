#!/usr/bin/env bash
set -e

echo "Installing requirements…"
pip install --upgrade pip
pip install -r requirements.txt


echo "adding different notebooks where different models are trained"
NOTEBOOKS=(
  emotic_data_extraction.ipynb
  face_and_pose_embeddings.ipynb
  data_cleaning.ipynb
  individual_vit.ipynb
  Simple_VIT.ipynb
  Chunked_VIT.ipynb
  complex_chunked_vit.ipynb
  cross_attention.ipynb
  pytorch_models.ipynb
)

echo "running notebooks and all the outputs are located in place"
for nb in "${NOTEBOOKS[@]}"; do
  echo "Running $nb …"
  jupyter nbconvert \
    --to notebook \
    --execute "$nb" \
    --inplace \
    
    --ExecutePreprocessor.timeout=-1
done

echo "Deploying the models locally using streamlit"
streamlit run app.py

echo "All done!"