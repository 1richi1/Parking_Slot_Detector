
# Parking Slot Detector and Classifier

![Python Version](https://img.shields.io/badge/python-3.x-blue)
![Status](https://img.shields.io/badge/status-Completed-green)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains the final project for **Parking Slot Detection and Classification**. The goal of this project is to detect and classify parking slots in images as **available** or **occupied** using a combination of computer vision techniques and machine learning models.

---

## 🚀 Features

- **Modified SIFT Algorithm**: Efficient feature extraction with reduced complexity.
- **Multilayer Perceptron (MLP)**: Pre-trained model included for immediate testing.
- **Multiple Region Proposal Methods**:
  - Classical approach with edge detection.
  - Machine learning-based selective search.
  - Connected component labeling.
- **Dataset Compatibility**: Designed for CNRPark+EXT and PKLot datasets.
- **Robust Testing**: Handles diverse weather conditions and image perspectives.

---

## 📁 Project Structure

```plaintext
Parking_Slot_Detector/
├── data/                  # Folder for input test images.
├── results/               # Folder for output results.
├── CNR-EXT/               # Dataset folder (to be added by the user).
├── CNRPark/               # Dataset folder (to be added by the user).
├── mlp_model.pkl          # Pre-trained MLP model.
├── main.py                # Entry point for running the project.
├── neural_network.py      # MLP model training and configuration.
├── requirements.txt       # List of dependencies.
└── README.md              # Project documentation.
```

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Parking_Slot_Detector.git
   cd Parking_Slot_Detector
   ```

2. Install dependencies, see main.py as reference

3. Download the datasets:
   - **CNRPark+EXT Dataset**: [Download Here](https://cnrpark.it)
   - (Optional) **PKLot Dataset**: [Download Here](https://web.inf.ufpr.br/vri/databases/pklot/)

4. Place the dataset folders (`CNR-EXT`, `CNRPark`, or `PKLot`) in the root directory. Ensure file paths match the structure in the `.csv` files.

---

## 🛠️ Usage

### Running the Project

1. Add test images to the `data/` folder.
2. Update the `filepath` variable in `main.py` with the path of your test image.
3. Run the script:
   ```bash
   python main.py
   ```

By default, the script runs in **test mode** using the pre-trained MLP model.

### Training the Model

To train or fine-tune the MLP model:
1. Uncomment the training lines in `main.py`.
2. Adjust configurations in `neural_network.py`.
3. Run the script:
   ```bash
   python main.py
   ```

---

## 📊 Results

- **Modified SIFT Descriptors**:
  - Improved accuracy and reduced training time compared to traditional SIFT.
- **Region Proposal Methods**:
  - Classical methods perform better on frontal views.
  - Machine learning-based selective search excels in general scenarios.
- **Performance Under Varying Conditions**:
  - Consistent results across sunny, rainy, foggy, and cloudy weather.

### Example Output

**Input Image:**
![Input Example](path/to/input_example.jpg)

**Detected Parking Slots:**
![Output Example](path/to/output_example.jpg)

---

## 🔬 Technical Details

### Region Proposal Methods
1. **Classical Approach**: Edge detection (Canny) + Contour finding.
2. **Selective Search**: Machine learning-based region proposals.
3. **Connected Components**: Groups pixels into connected regions.

### Classification
- **Modified SIFT**: Custom descriptor to standardize features.
- **MLP Model**:
  - Logistic activation.
  - Stochastic Gradient Descent (SGD) solver.

---

## 📚 References

1. Uijlings, J. R. R., et al. "Selective Search for Object Recognition."
2. Felzenszwalb, P. F., and Huttenlocher, D. P. "Efficient Graph-Based Image Segmentation."
3. Wu, K., et al. "Optimizing Connected Component Labeling Algorithms."

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---


Ready to detect parking slots? 🚗✨ Download the datasets, run the code, and start exploring!
