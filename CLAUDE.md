# Task 4: Satellite Land-Use Classifier (VoidMain Internship)

## Goal
Brief: "Select an image/text/tabular problem. Design/use a suitable neural-network architecture, train/fine-tune, evaluate it and test on unseen data. Explain architecture, training decisions, results and limitations."
**Keep it SIMPLE.** Only build what's below - minimal code, no extras. It just needs to work and be easy to show in a video.

## Setup
- Windows, Python 3.14, project in OneDrive. Keep `.venv` and `data/` out of git. Empty `screenshots/.gitkeep`.
- Install PyTorch + torchvision (use CUDA if `nvidia-smi` works and a matching wheel exists, otherwise CPU - don't spend long on this). Pin versions in `requirements.txt`.

## What to build
1. **Data:** EuroSAT RGB via `torchvision.datasets.EuroSAT` (27,000 images, 10 classes). Stratified split 70/15/15; test set used only at the end. Resize/normalize; random flips on train.
2. **Model:** pretrained MobileNetV3-Small from torchvision, new 10-class head. Train the head first (frozen backbone), then fine-tune all layers with a lower LR. A few epochs, AdamW, early stopping, save best checkpoint.
3. **Evaluate on the unseen test set:** accuracy, macro F1, classification report, confusion matrix. Save 2 figures: training curves + confusion matrix.
4. **Export** the model to ONNX (small file) for the demo.
5. **Demo:** one static page `web/index.html` (inline CSS/JS) using onnxruntime-web from CDN: upload an image or click a few sample tiles -> top-3 classes with animated probability bars. Below it: test accuracy, the 2 figures, and a short "How it works" + "Limitations" section.
   - Look: dark navy `#0B1020` background, gradient hero (electric blue `#4F7CFF` -> neon pink `#FF3D81`), marigold `#FFB020` and mint `#22D3B6` accents, rounded glassy cards. CSS only, responsive.
6. **README:** problem, dataset credit (Helber et al. 2019), architecture, training decisions, results, limitations, how to run, live link.

## Structure
```
train.py      # data -> train -> evaluate -> figures -> ONNX -> copies files into web/
web/          # index.html + model.onnx + figures + sample images
reports/      # figures + metrics.json
README.md, requirements.txt, .gitignore, screenshots/.gitkeep
```

## Rules
- One command: `python train.py`. Clean, commented code; no test suite needed.
- Do everything in one go without asking. Pick the safer/simpler option when unsure.
- Open the page locally once to confirm predictions work.
- Finish with a SHORT summary: test accuracy/F1, how to run, Render Static Site settings (publish dir `web`, no build command). Then stop.
