# README — MER2023 Evaluation Guide

This repository contains a pruned and lightweight implementation of a **multimodal emotion reasoning model** designed for evaluation on the **MER2023 dataset**.

---

## 🚀 Environment Setup

- Recommended version: **Python 3.9**
- Install dependencies using:
```
pip install -r requirements.txt
```

---

## 🎯 Evaluation on the MER2023 Dataset

### 1. Download Model Weights

####  Base Model
Download `LLaMA-2 7B Chat` https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Set model path inside `eval_emotion.yaml`:
```yaml
llama_model: "/mnt/bear3/users/jungji/ckpt/Llama-2-7b-chat-hf"
```

####  Finetuned Checkpoint
Download `checkpoint_best.pth`  https://drive.google.com/file/d/1NoPZDj5_392zBtVK1IHO8bepA4910iI_/view

Set model path inside `eval_emotion.yaml`:
```yaml
ckpt: "/path/to/checkpoint_best.pth"
```

---

### 2. Download MER2023 Data

#### Feature Files Required
- `features_of_MER2023-SEMI.zip`  
  https://drive.google.com/file/d/1DJJ8wP3g4yLT0ZFZ_-H4izJHAGJx2AJ_/view

- `relative_test3_NCEV.txt`  
  https://drive.google.com/file/d/1YyoWabWtAJuFI6ylMM220i_kh_0Nagv9/view

After extraction, directory structure must be:

```
mer2023/
│── features_of_MER2023-SEMI/
      └── first_frames_MER2023-SEMI/
      └── relative_test3_NCEV.txt   ← must be placed here
```

Update config file `eval_emotion.yaml` accordingly:
```yaml
feature_face_caption:
    eval_file_path: /mnt/lynx2/datasets/mer2023/features_of_MER2023-SEMI/relative_test3_NCEV.txt
    img_path: /mnt/lynx2/datasets/mer2023/features_of_MER2023-SEMI/first_frames_MER2023-SEMI
```

---

### 3. Run Evaluation

To enable pruning mode → set `prune: true` in `eval_emotion.yaml`  
Otherwise → use `false`.

Run evaluation via:
```
torchrun --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset feature_face_caption
```

---
## 🎯 Simple Inference without Pruning

### 1. Download Model Weights

####  Hubert Checkpoint(Audio Encoder)
Download `chinese-hubert-large` https://huggingface.co/TencentGameMate/chinese-hubert-large


####  Base Model
Download `LLaMA-2 7B Chat` https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Set model path inside `eval_emotion.yaml`:
```yaml
llama_model: "/mnt/bear3/users/jungji/ckpt/Llama-2-7b-chat-hf"
```

####  Finetuned Checkpoint
Download `checkpoint_best.pth`  https://drive.google.com/file/d/1NoPZDj5_392zBtVK1IHO8bepA4910iI_/view

Set model path inside `eval_emotion.yaml`:
```yaml
ckpt: "/path/to/checkpoint_best.pth"
```

### 2. Run Inference

--video-path "video_path" --hubert-model-path "/mnt/bear3/users/jungji/ckpt/chinese-hubert-large"

---
### 📚 Reference

Original Emotion-LLaMA Repository:  
https://github.com/ZebangCheng/Emotion-LLaMA
