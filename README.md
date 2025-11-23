MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization (ICCV 2025)

This repository contains the official PyTorch implementation of the paper:
"MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization", accepted at ICCV 2025.

<b>Hyung Kyu Kim¹, Sangmin Lee², Hak Gu Kim¹</b>
¹Chung-Ang University, ²Korea University
</div>

📝 Abstract

Speech-driven 3D facial animation aims to synthesize realistic facial motion sequences from given audio, matching the speaker's speaking style. However, previous works often require priors such as class labels or additional 3D facial meshes at inference.
To address these issues, we propose MemoryTalker, which enables realistic and accurate 3D facial motion synthesis by reflecting speaking style only with audio input. Our framework consists of two training stages:

Memorizing: Storing and retrieving general motion.

Animating: Performing personalized facial motion synthesis with motion memory stylized by audio-guided style features.

🛠️ Environment Setup

The code has been tested with Python 3.8+ and PyTorch 1.12+.

git clone [https://github.com/cau-irislab/MemoryTalker.git](https://github.com/cau-irislab/MemoryTalker.git)
cd MemoryTalker
pip install -r requirements.txt


📂 Data Preparation

We use VOCASET for training and evaluation. Please download the dataset and organize the directory as follows:

MemoryTalker/
└── vocaset/
    ├── wav/                # Audio files (.wav)
    ├── vertices_npy/       # 3D facial motion files (.npy)
    ├── templates.pkl       # Neutral face templates
    └── range/
        └── lips_coordinates.npy  # Lip indices for LVE metric


🚀 Training Pipeline

Our training consists of two stages. You must train Stage 1 first, then use the pretrained model to train Stage 2.

1. Stage 1: Memorizing General Motion

In this stage, the model learns generic lip synchronizations and stores them in the Motion Memory.

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


Note: The checkpoints will be saved in vocaset/save_baseline_1stage_stage1/.

2. Stage 2: Audio-Guided Stylization

In this stage, we freeze the general motion parameters and train the Speaking Style Encoder to stylize the memory based on the input audio.

Requires: Path to the best checkpoint from Stage 1 (e.g., 4_model.pth or best_model.pth).

python train_VOCA_2stage.py \
    --root_path ./ \
    --dataset vocaset \
    --pretrained_model_path ./vocaset/save_baseline_1stage_stage1/4_model.pth \
    --exp_name baseline_2stage \
    --lr 0.00005 \
    --max_epoch 100 \
    --triplet_margin 5.0


📊 Inference

To generate 3D facial animations using the trained Stage 2 model:

# This is automatically run after training, or can be run manually via test script
python train_VOCA_2stage.py --evaluate_only --pretrained_model_path ... 


📜 Citation

If you find this code or paper useful, please cite:

@InProceedings{Kim_2025_ICCV,
    author    = {Kim, Hyung Kyu and Lee, Sangmin and Kim, Hak Gu},
    title     = {MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}
