#  **VLG Dataset Image Classification Project**

## 🌟 **Overview**
🗂️ This repository contains resources and code for classifying images into 50 📚 categories using 🏋️‍♂️ ResNet50. It details steps such as data preprocessing, model fine-tuning, training, and 📈 making predictions, providing a robust framework for image classifications.

---

## 📁 **Repository Structure**

- 📝 **`notebook.ipynb`**: 📘 Jupyter notebook with the complete 🚀 implementation pipeline.
- 💾 **`best_model.pth`**: 🔒 Checkpoint for the trained ResNet50 model.
- 📊 **`predictions.csv`**: 📑 File containing submission-ready 📝 predictions for the test dataset.
- 📄 **`README.md`**: 📜 File explaining 📚 project details and instructions.

---

## ⚙️ **Prerequisites**

🔧 Install these 🛠️ tools to set up the environment:

- 🐍 Python 
- 🔥 PyTorch 
- 📸 torchvision 
- 📐 scikit-learn
- 🎨 matplotlib
- 📊 pandas
- 📓 Jupyter Notebook

---

## 🗂️ **Datasets**

📂 The dataset is divided into 🏋️‍♂️ training, 📊 validation, and 📁 test sets. It facilitates essential tasks:

- 📥 Loading and preprocessing 🖼️ data.
- 🔀 Splitting data into training and validation subsets.
- 🛠️ Generating predictions for unseen test data.

---

## 🔑 **Key Steps in Implementation**

### 1️⃣ **📥 Data Loading & Preprocessing**

- **🔎 Description**: The dataset is read, resized to 224x224 pixels, normalized for ResNet50, and converted into tensor format. Each image undergoes transformation to maintain consistency with ResNet50 input requirements.
- **🧑‍💻 Code Highlights**:
  ```python
  transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  ```
- **Rationale**: These transformations ensure pixel intensities are scaled, and the model input format is maintained.

### 2️⃣ **🛠️ Model Fine-Tuning**

- **🔎 Description**: The ResNet50 architecture is modified to support 50 🏷️ output categories by adding a custom fully connected layer. A dropout layer is included to reduce overfitting.
- **🧑‍💻 Code Highlights**:
  ```python
  model.fc = nn.Sequential(
      nn.Linear(num_features, 512),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(512, 50)
  )
  ```
- **Rationale**: Transfer learning leverages pretrained weights, while the fully connected layer adapts the model to our specific classification task.

### 3️⃣ **🏋️‍♂️ Training & Validation**

- **🔎 Description**: The model is trained using the Adam optimizer with weight decay for regularization, and a StepLR scheduler dynamically adjusts the learning rate to enhance convergence.
- **🧑‍💻 Code Highlights**:
  ```python
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
  torch.save(model.state_dict(), "best_model.pth")
  ```
- **Rationale**: The combination of optimization techniques ensures efficient training and prevents overfitting.

### 4️⃣ **🔍 Evaluation**

- **🔎 Description**: The model is evaluated on validation data using metrics such as accuracy and loss, with trends plotted to monitor training effectiveness.
- **Key Metrics**: Validation accuracy of ~98.01% highlights the model’s strong generalization.

### 5️⃣ **📤 Prediction & Submission**

- **🔎 Description**: Predictions are generated for the test dataset, and results are formatted into a CSV file for submission.
- **🧑‍💻 Code Highlights**:
  ```python
  test_preds = []
  for images in test_loader:
      outputs = model(images.to(device))
      _, preds = torch.max(outputs, 1)
      test_preds.extend(preds.cpu().numpy())
  ```
- **Rationale**: This process ensures compatibility with competition requirements.

---

## 🏆 **Results**

- **🎯 Training Accuracy**: Achieved an impressive 📈 ~95.63%.
- **🎯 Validation Accuracy**: Reached 📊 ~98.01%, demonstrating robust model generalization.



## 🌟 **Vital Components**

1️⃣ **🏋️‍♂️ Pretrained ResNet50**: Utilizes transfer learning for high 🎯 accuracy.
2️⃣ **🧠 Custom Fully Connected Layer**: Adapts ResNet50 for the 50-category 🏷️ classification task.
3️⃣ **🎚️ Learning Rate Scheduler**: Dynamically adjusts learning rates for efficient training.
4️⃣ **⛔ Dropout Layers**: Reduces 📉 overfitting, ensuring better generalization on unseen datasets.
5️⃣ **📥 Data Preprocessing**: Rescales and normalizes images to align with model input requirements.
6️⃣ **🛠️ Custom Training Loop**: Integrates evaluation and scheduler updates within each epoch for streamlined performance monitoring.

---

## 💬 **Comments & Documentation**

💡 The code is extensively commented to provide clear explanations for:

- 🔧 Data preprocessing steps.
- 🛠️ Model 🎛️ architecture modifications.
- 🔄 Training and validation loops.
- 📊 Test data prediction and result generation.

---

## 🌟 **Conclusion**
This project showcases a robust approach to image classification using a ResNet50 backbone, custom fully connected layer, and a learning rate scheduler. It provides a solid foundation for further exploration and fine-tuning for optimal performance in image classification tasks.

