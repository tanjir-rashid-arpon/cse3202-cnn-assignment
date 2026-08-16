# CSE-3202: CNN Image Classification with PyTorch

**Assignment 8 | FashionMNIST + Custom Phone Images**

---

## 📁 Repository Structure

```
cse3202-cnn-assignment/
├── dataset/              ← 10 custom smartphone photos
│   ├── tshirt.jpg
│   ├── sneaker.jpg
│   ├── bag.jpg
│   ├── trouser.jpg
│   ├── pullover.jpg
│   ├── dress.jpg
│   ├── coat.jpg
│   ├── sandal.jpg
│   ├── shirt.jpg
│   └── ankleboot.jpg
├── model/
│   └── 190110.pth        ← Saved model state dict (after training)
├── 190110.ipynb          ← Google Colab notebook
└── README.md
```

---

## 🧠 Model Architecture

A custom CNN built with `nn.Module` containing:

| Layer Block | Details |
|-------------|---------|
| Conv Block 1 | Conv2d(1→32) × 2, ReLU, MaxPool2d, Dropout(0.25) |
| Conv Block 2 | Conv2d(32→64) × 2, ReLU, MaxPool2d, Dropout(0.25) |
| Conv Block 3 | Conv2d(64→128), ReLU, MaxPool2d, Dropout(0.25) |
| FC Layer 1   | Linear(1152→512), ReLU, Dropout(0.5) |
| FC Layer 2   | Linear(512→256), ReLU, Dropout(0.3) |
| Output       | Linear(256→10) |

- **Loss Function**: `nn.CrossEntropyLoss`
- **Optimizer**: `torch.optim.Adam` (lr=0.001, weight_decay=1e-4)
- **Scheduler**: StepLR (step=5, gamma=0.5)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dataset | FashionMNIST (60K train / 10K test) |
| Epochs | 15 |
| Best Validation Accuracy | ~92%+ |

### Training History
![Training History](training_history.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Custom Image Predictions
![Custom Predictions](custom_predictions.png)

### Error Analysis
![Error Analysis](error_analysis.png)

---

## 🏷️ FashionMNIST Classes

| Label | Class | Custom Photo |
|-------|-------|-------------|
| 0 | T-shirt/top | `tshirt.jpg` |
| 1 | Trouser | `trouser.jpg` |
| 2 | Pullover | `pullover.jpg` |
| 3 | Dress | `dress.jpg` |
| 4 | Coat | `coat.jpg` |
| 5 | Sandal | `sandal.jpg` |
| 6 | Shirt | `shirt.jpg` |
| 7 | Sneaker | `sneaker.jpg` |
| 8 | Bag | `bag.jpg` |
| 9 | Ankle boot | `ankleboot.jpg` |

---

## 🚀 How to Run

1. Open `190110.ipynb` in Google Colab
2. Set Runtime → **T4 GPU**
3. Click **Runtime → Run All**

The notebook will automatically:
1. Clone this repo (pulls custom images)
2. Download FashionMNIST via `torchvision.datasets`
3. Train the CNN for 15 epochs
4. Save model weights to `model/190110.pth`
5. Predict on all 10 custom phone images
6. Display all plots and visuals

> ⚠️ **Penalty Note**: No manual file uploads required. Everything is automated via `git clone`.

---

## 📋 Submission Checklist

- [x] GitHub repo with `dataset/`, `model/`, `.ipynb`, `README.md`
- [x] CNN class with `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Linear`
- [x] `forward(self, x)` method defined
- [x] `nn.CrossEntropyLoss` + `torch.optim.Adam`
- [x] Training loop: `zero_grad()` → `backward()` → `step()`
- [x] Model saved with `torch.save(model.state_dict(), 'ID.pth')`
- [x] Custom images loaded with PIL, same transforms as training
- [x] `torch.softmax` applied for confidence scores
- [x] Training History plots (Loss & Accuracy vs Epochs)
- [x] Confusion Matrix heatmap
- [x] Custom Prediction Gallery (10 images, class + confidence)
- [x] Visual Error Analysis (3 misclassified examples)
- [x] Run All works without any manual uploads
