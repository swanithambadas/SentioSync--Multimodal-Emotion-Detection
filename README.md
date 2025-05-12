<p align="center">
  <img src="assets/demo.gif" alt="SentioSync Demo" width="80%"/>
</p>

<h1 align="center">💡 SentioSync – Multimodal Emotion Detection</h1>
<p align="center"><strong>Realtime emotion analysis with body pose & facial embeddings in Vision Transformers</strong></p>

<p align="center">
  <a href="#notebooks--scripts">📓 Notebooks & Scripts</a> • 
  <a href="#techniques">🔬 Techniques</a> • 
  <a href="#technologies">📚 Technologies</a> • 
  <a href="#project-structure">📂 Project Structure</a> • 
  <a href="#contributors">👥 Contributors</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white"/> 
  <img src="https://img.shields.io/badge/Streamlit-1.x-orange?logo=streamlit&logoColor=white"/> 
  <img src="https://img.shields.io/badge/PyTorch-1.10-red?logo=pytorch&logoColor=white"/> 
  <img src="https://img.shields.io/badge/Transformers-4.x-purple?logo=transformers&logoColor=white"/> 
  <img src="https://img.shields.io/badge/OpenCV-4.x-yellow?logo=opencv&logoColor=white"/>
</p>

---

## 📓 Notebooks & Scripts

- **Streamlit App**: [`Code/app.py`](./Code/app.py) – Image upload, model selector & inference loop  
- **Model Definitions**: [`Code/models.py`](./Code/models.py) – Chunked, cross‐attention & multi‐stage ViT variants  
- **Quick Start**: [`Code/run.sh`](./Code/run.sh) – Env setup & demo launch  
- **Experiments** (in `Code/`):  
  - `data_cleaning.ipynb` – Dataset exploration & cleaning  
  - `emotic_data_extraction.ipynb` – Emotic dataset processing  
  - `face_and_pose_embeddings.ipynb` – CLIP & Keypoint‐RCNN feature extraction  
  - `cross_attention.ipynb`, `Chunked_VIT.ipynb`, `Simple_VIT.ipynb`, `complex_chunked_vit.ipynb`, `individual_vit.ipynb` – Architecture variants  
  - `pytorch_models.ipynb` – Prototyping pipelines  
- **Annotations**: `Code/annotations/Annotations.mat` – Raw annotation data  
- **Pretrained Weights**: `Code/models/*.pt` – Checkpoint files  

---

## 🔬 Techniques

- **Person Detection** with TorchVision’s [Faster R-CNN](https://pytorch.org/vision/stable/models.html#torchvision.models.detection.fasterrcnn_resnet50_fpn)  
- **Facial Embeddings** via OpenCV Haar cascades ([docs](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)) + OpenAI’s [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)  
- **Pose Estimation** using [Keypoint-RCNN](https://pytorch.org/vision/stable/models.html#keypointrcnn_resnet50_fpn)  
- **Vision Transformers**  
  - Chunked & multi‐stage pipelines for large inputs  
  - Cross‐attention fusion of face & pose features  
- **Realtime UI** with [Streamlit](https://docs.streamlit.io/) for fast iteration  

---

## 📚 Technologies

- **Python 3.8+**  
- **Streamlit** – Interactive demos ([docs](https://docs.streamlit.io/))  
- **PyTorch & TorchVision** – Deep learning & detection ([docs](https://pytorch.org/))  
- **Transformers (Hugging Face)** – CLIP model ([docs](https://huggingface.co/docs/transformers/))  
- **OpenCV** – Image I/O & Haar cascades ([docs](https://docs.opencv.org/4.x/))  
- **Pillow** – Image handling ([docs](https://pillow.readthedocs.io/))  
- **NumPy & SciPy** – Numerical ops ([NumPy](https://numpy.org/doc/), [SciPy](https://scipy.org/docs/))  

---

## 📂 Project Structure

```bash
.
├── Code/
│   ├── annotations/              # Raw Emotic .mat annotation files
│   │   └── Annotations.mat
│   ├── models/                   # Pretrained .pt checkpoints
│   │   └── *.pt
│   ├── app.py                    # Streamlit entry point
│   ├── models.py                 # ViT & fusion architecture definitions
│   ├── requirements.txt          # Python dependencies
│   └── run.sh                    # Setup & launch script
└── Report/
    └── swanitha_manishbi_final_report.pdf  # Detailed write-up & results
