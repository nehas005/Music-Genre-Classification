# Music Genre Classification

Music genre classification using the GTZAN dataset, mel-spectrogram features, traditional machine learning models, and a CNN.

## Dataset

The project uses the GTZAN music genre dataset.

The dataset contains 10 genres:

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

One corrupted audio file (`jazz.00054.wav`) was excluded because it could not be decoded during feature extraction.

## Person 1 - Traditional Machine Learning

Person 1 implemented:

- Audio loading and preprocessing
- Mel-spectrogram extraction
- Feature preparation
- Train/validation/test splitting
- Feature scaling
- SVM classification
- k-NN classification
- Baseline evaluation
- Saved traditional ML models

## Feature Extraction

Each audio file is converted into a log-scaled mel-spectrogram.

For traditional ML, each spectrogram is converted into a fixed-length feature vector using:

- Mean across time for each mel-frequency band
- Standard deviation across time for each mel-frequency band

This produces 256 features per audio file.

## Dataset Split

The dataset is split into:

- 70% training
- 15% validation
- 15% testing

A fixed random seed of 42 is used.

## Traditional ML Results

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| SVM | 68.00% | **68.67%** |
| k-NN | 56.67% | **50.00%** |

SVM performed better than k-NN on the test set.

## Saved Models

The following models are generated:

- `models/svm_model.pkl`
- `models/knn_model.pkl`
- `models/scaler.pkl`

## Results

The project generates:

- `results/sample_spectrogram.png`
- `results/accuracy_comparison.png`

Additional feature and test arrays are stored in the `results/` directory.

## Running the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt