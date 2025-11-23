# MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization (ICCV 2025)

Official PyTorch implementation of **“MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization”**, accepted at **ICCV 2025**.

> **Hyung Kyu Kim¹, Sangmin Lee², Hak Gu Kim¹**  
> ¹Chung-Ang University, ²Korea University

---

## 📝 Abstract

Speech-driven 3D facial animation aims to synthesize realistic facial motion sequences from given audio, matching the speaker’s speaking style.  
However, previous works often require priors such as class labels or additional 3D facial meshes at inference time.

To address these issues, we propose **MemoryTalker**, which enables realistic and accurate 3D facial motion synthesis **using only audio input**.  
Our framework consists of two training stages:

1. **Memorizing**: Storing and retrieving general facial motion in a motion memory.  
2. **Animating**: Synthesizing personalized facial motions by stylizing the retrieved memory with audio-guided style features.

---

## 🛠️ Environment Setup

Tested with:
- **Python 3.8+**
- **PyTorch 1.12+**

```bash
git clone https://github.com/cau-irislab/MemoryTalker.git
cd MemoryTalker
```

---

## 📂 Data Preparation

We use **VOCASET** for training and evaluation.  
Download the dataset and organize the directory as follows:

```
MemoryTalker/
└── vocaset/
    ├── wav/                   # Audio files (.wav)
    ├── vertices_npy/          # 3D facial motion files (.npy)
    ├── templates.pkl          # Neutral face templates
    └── range/
        └── lips_coordinates.npy  # Lip indices for LVE metric
```

---

## 🚀 Training Pipeline

Training consists of **two stages**.  
You must train **Stage 1 first**, then use the pretrained model to train **Stage 2**.

---

### 1) Stage 1: Memorizing General Motion

The model learns generic lip synchronization and stores it in the motion memory.

```bash
python train_VOCA_1stage.py \
    --stage 1 \
    --root_path ./ \
    --dataset vocaset \
    --wav_path wav \
    --vertices_path vertices_npy \
    --template_file templates.pkl \
    --exp_name baseline_1stage \
    --max_epoch 100 \
    --device cuda
```

**Checkpoint path**
```
vocaset/save_baseline_1stage_stage1/
```

---

### 2) Stage 2: Audio-Guided Stylization

We freeze the general motion parameters and train the **Speaking Style Encoder** to stylize memory based on input audio.

> Requires the best checkpoint from Stage 1 (e.g., `4_model.pth` or `best_model.pth`).

```bash
python train_VOCA_2stage.py \
    --root_path ./ \
    --dataset vocaset \
    --pretrained_model_path ./vocaset/save_baseline_1stage_stage1/4_model.pth \
    --exp_name baseline_2stage \
    --lr 0.00005 \
    --max_epoch 100 \
    --triplet_margin 5.0
```

---

## 📊 Inference

Generate personalized 3D facial animations using the trained Stage 2 model:

```bash
# Automatically runs after Stage 2 training
# or can be executed manually
python train_VOCA_2stage.py \
    --evaluate_only \
    --pretrained_model_path <PATH_TO_STAGE2_CHECKPOINT>
```

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Kim_2025_ICCV,
    author    = {Kim, Hyung Kyu and Lee, Sangmin and Kim, Hak Gu},
    title     = {MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}
```

---

## ✅ Acknowledgements

This project is based on VOCASET and built with PyTorch.  
We thank the community for open-source tools and datasets that supported this research.
