import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from transformers import CLIPImageProcessor, CLIPModel
from PIL import Image, ImageDraw
import cv2
import os
import tempfile

# ─── CONFIG ──────────────────────────────────────────
DET_THRESH   = 0.5
HAAR_XML     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EMO_CLASSES  = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "contempt"]
MODEL_DIR    = "models"
# ─────────────────────────────────────────────────────

st.set_page_config(page_title="Emotion-VIT Predictor", layout="wide")
st.title("🔍 SentioSync")

# — Sidebar: choose pretrained ViT weights —
pt_files       = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
selected_model = st.sidebar.selectbox("Choose The Model", ["(none)"] + pt_files)

# — Image uploader —
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to get started.")
    st.stop()

# Save upload for OpenCV
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
tfile.write(uploaded.read())
img_path = tfile.name

# Show the uploaded image
pil_img = Image.open(img_path).convert("RGB")
st.image(pil_img, caption="Uploaded Image", use_container_width=False, width=200)

# — Single Predict button —
if st.button("Predict Emotions"):

    if selected_model == "(none)":
        st.error("Please select a ViT model from the sidebar before predicting.")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Person detection (Faster R-CNN)
    det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).eval()
    tensor_img = T.ToTensor()(pil_img).unsqueeze(0)
    with torch.no_grad():
        det_out = det_model(tensor_img)[0]

    boxes  = det_out["boxes"]
    labels = det_out["labels"]
    scores = det_out["scores"]
    mask   = (labels == 1) & (scores >= DET_THRESH)
    person_boxes = boxes[mask]

    # Draw detected persons
    debug = pil_img.copy()
    dr    = ImageDraw.Draw(debug)
    for b in person_boxes:
        x1, y1, x2, y2 = map(int, b.tolist())
        dr.rectangle([x1, y1, x2, y2], outline="red", width=3)
    #st.subheader(f"Detected {len(person_boxes)} Person(s)")
    #st.image(debug, use_column_width=True)

    # Prepare records
    records = [{"id": idx, "bbox": list(map(int, b.tolist()))}
               for idx, b in enumerate(person_boxes)]

    # 2. Face embeddings via Haar + CLIP
    face_cascade = cv2.CascadeClassifier(HAAR_XML)
    proc         = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_mdl     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    cv_img       = cv2.imread(img_path)

    for rec in records:
        x1, y1, x2, y2 = rec["bbox"]
        crop = cv2.cvtColor(cv_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            face_img = crop[fy:fy+fh, fx:fx+fw]
        else:
            face_img = crop

        face_pil = Image.fromarray(face_img)
        inputs   = proc(images=face_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_mdl.get_image_features(**inputs).squeeze(0).cpu()
        rec["face_emb"] = emb

    # 3. Pose embeddings via Keypoint-RCNN
    kp_model  = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    to_tensor = T.ToTensor()
    for rec in records:
        x1, y1, x2, y2 = rec["bbox"]
        crop = cv_img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            pose_vec = torch.zeros(34)
        else:
            tc = to_tensor(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                kp_out = kp_model(tc)[0]
            keep = (kp_out["scores"] >= DET_THRESH) & (kp_out["labels"] == 1)
            if keep.any():
                i   = torch.nonzero(keep).squeeze(1)[0].item()
                kps = kp_out["keypoints"][i][:, :2]
                pose_vec = kps.flatten()
            else:
                pose_vec = torch.zeros(34)
        rec["pose_emb"] = pose_vec

    # 4. Load ViT and predict emotions
    from models import (
        ChunkedMultiStageViT,
        ChunkedMultiStageViT_fine,
        ChunkedCrossAttnViT,
        ChunkedCrossAttnViT_tuned,
    )

    if selected_model == "ChunkedMultiStageVit.pt":
        vit = ChunkedMultiStageViT()
    elif selected_model == "ChunkedMultiStageVit_fine.pt":
        vit = ChunkedMultiStageViT_fine()
    elif selected_model == "ChunkedCrossAttnVit.pt":
        vit = ChunkedCrossAttnViT()
    elif selected_model == "ChunkedCrossAttnVit_tuned.pt":
        vit = ChunkedCrossAttnViT_tuned()
    else:
        st.error("Unrecognized model file.")
        st.stop()

    ckpt = torch.load(os.path.join(MODEL_DIR, selected_model), map_location=device)
    try:
        vit.load_state_dict(ckpt)
    except RuntimeError:
        st.warning("Key mismatch detected: loading with strict=False.")
        load_res = vit.load_state_dict(ckpt, strict=False)
        if load_res.missing_keys:
            st.write("Missing keys (using defaults):", load_res.missing_keys)
        if load_res.unexpected_keys:
            st.write("Unexpected keys (ignored):", load_res.unexpected_keys)
    vit.to(device).eval()

    st.subheader("Predicted Emotions")
    for rec in records:
        f_vec = rec["face_emb"].unsqueeze(0).to(device)
        p_vec = rec["pose_emb"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = vit(f_vec, p_vec)  # (1,7)
        mh = (out > 0.5).int().squeeze(0).cpu().tolist()
        labels = [c for c, m in zip(EMO_CLASSES, mh) if m]
        result = ", ".join(labels) if labels else "none detected"
        st.markdown(f"**Person {rec['id']}** → {result}")
