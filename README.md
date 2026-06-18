# Image Captioning with Custom Encoder–Decoder and BLIP + LoRA

A deep learning project for automatic image caption generation using two complementary approaches:

1. **Custom image captioning model from scratch**
   A vision-language architecture built with an image encoder and a Transformer-based text decoder.

2. **BLIP fine-tuning with LoRA**
   A parameter-efficient fine-tuning pipeline for `Salesforce/blip-image-captioning-base` using LoRA adapters on the COCO 2014 dataset.

This repository demonstrates both a fundamental implementation of image captioning and a modern transfer-learning approach using a pretrained vision-language model.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Main Features](#main-features)
* [Repository Structure](#repository-structure)
* [Approach 1: Custom Encoder–Decoder Model](#approach-1-custom-encoderdecoder-model)
* [Approach 2: BLIP Fine-Tuning with LoRA](#approach-2-blip-fine-tuning-with-lora)
* [Dataset](#dataset)
* [Installation](#installation)
* [How to Run](#how-to-run)
* [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [Model Checkpoints](#model-checkpoints)
* [Future Improvements](#future-improvements)
* [License](#license)
* [Author](#author)

---

## Project Overview

Image captioning is a multimodal deep learning task that combines computer vision and natural language processing. The goal is to generate a meaningful textual description for a given image.

This project explores image captioning through two different modeling strategies:

### 1. From-Scratch Architecture

The first approach builds an image captioning model manually using:

* A CNN-based image encoder
* A Transformer-based text decoder
* Custom vocabulary construction
* COCO caption preprocessing
* Greedy decoding and beam search generation

This part is useful for understanding how image captioning systems work internally.

### 2. BLIP + LoRA Fine-Tuning

The second approach fine-tunes a pretrained BLIP model using LoRA, a parameter-efficient fine-tuning technique. Instead of updating all model parameters, LoRA trains a small number of adapter weights, making the process more memory-efficient and suitable for limited GPU environments such as Google Colab.

---

## Main Features

* End-to-end image captioning pipeline
* Custom encoder–decoder model implementation
* Transformer decoder for caption generation
* EfficientNet-based image feature extraction
* BLIP fine-tuning using LoRA adapters
* COCO 2014 dataset support
* Caption generation with greedy decoding and beam search
* Evaluation using standard captioning metrics
* Notebook-based workflow suitable for Google Colab
* Checkpoint saving and loading
* Visualization of generated captions

---

## Repository Structure

```text
Image_captioning/
│
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
│
├── data acquisition.ipynb
├── Training-5k.ipynb
├── Traning-100k.ipynb
├── blip_lora_coco2014.ipynb
│
└── checkpoints.zip
```

### File Descriptions

| File                       | Description                                                       |
| -------------------------- | ----------------------------------------------------------------- |
| `data acquisition.ipynb`   | Notebook for downloading or preparing the dataset                 |
| `Training-5k.ipynb`        | Training notebook for the custom model using a smaller subset     |
| `Traning-100k.ipynb`       | Training notebook for the custom model using a larger COCO subset |
| `blip_lora_coco2014.ipynb` | BLIP fine-tuning notebook using LoRA                              |
| `checkpoints.zip`          | Saved model checkpoints                                           |
| `LICENSE`                  | MIT license                                                       |

> Note: The file name `Traning-100k.ipynb` appears to contain a spelling typo. You may optionally rename it to `Training-100k.ipynb` for clarity.

---

## Approach 1: Custom Encoder–Decoder Model

The custom model is designed to show the core components of an image captioning system.

### Architecture

The model consists of two main parts:

```text
Image → CNN Encoder → Visual Features → Transformer Decoder → Caption
```

### Image Encoder

The encoder extracts visual features from images using an EfficientNet-based backbone. The extracted features are projected into an embedding space and passed to the decoder.

Main components:

* EfficientNet feature extractor
* Linear projection layer
* Layer normalization
* Positional embeddings
* Optional fine-tuning of deeper encoder blocks

### Text Decoder

The decoder is implemented using a Transformer decoder architecture.

Main components:

* Token embedding layer
* Positional embedding layer
* Multi-head attention
* Cross-attention over image features
* Feed-forward layers
* Layer normalization
* Linear vocabulary prediction head

### Caption Generation

The custom model supports:

* Greedy decoding
* Beam search decoding

Greedy decoding selects the most probable next token at each step, while beam search keeps multiple candidate captions and generally produces more fluent results.

---

## Approach 2: BLIP Fine-Tuning with LoRA

The second approach fine-tunes the pretrained BLIP image captioning model:

```text
Salesforce/blip-image-captioning-base
```

Instead of full fine-tuning, this project uses **LoRA**, which injects low-rank trainable adapters into selected attention layers.

### Why LoRA?

LoRA is useful because it:

* Reduces the number of trainable parameters
* Requires less GPU memory
* Makes fine-tuning faster
* Preserves the pretrained model knowledge
* Works well in Colab-style environments

### BLIP + LoRA Pipeline

The BLIP notebook includes:

1. Dependency installation
2. Imports and configuration
3. COCO dataset loading
4. Dataset class definition
5. Data collator and DataLoader creation
6. BLIP model initialization
7. LoRA configuration
8. Training loop
9. Validation and generation
10. Evaluation metrics
11. Visualization
12. Inference
13. Save and load functions
14. Final results report

---

## Dataset

This project uses the **COCO 2014 image captioning dataset**.

Expected dataset structure:

```text
dataset/
├── train2014/
├── val2014/
├── test2014/
└── annotations/
    ├── captions_train2014.json
    ├── captions_val2014.json
    └── image_info_test2014.json
```

Each image in COCO usually has multiple human-written captions. During training, the model can learn from these different descriptions to generate more natural captions.

---

## Installation

The notebooks are designed to run in Google Colab. A GPU runtime is recommended.

### Basic Dependencies

Install the required packages inside the notebooks or manually with:

```bash
pip install torch torchvision transformers datasets peft accelerate
pip install nltk rouge-score pycocoevalcap safetensors einops
```

For the BLIP + LoRA notebook, pinned versions may be used for reproducibility:

```bash
pip install transformers==4.44.2 datasets==2.21.0 peft==0.13.2 accelerate==0.34.2
pip install nltk==3.9.1 rouge-score==0.1.2 pycocoevalcap==1.2 safetensors
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/AliSedghiye/Image_captioning.git
cd Image_captioning
```

### 2. Open the Notebooks

You can run the notebooks in:

* Google Colab
* Jupyter Notebook
* JupyterLab

Recommended order:

```text
1. data acquisition.ipynb
2. Training-5k.ipynb
3. Traning-100k.ipynb
4. blip_lora_coco2014.ipynb
```

### 3. Prepare the Dataset

Download and arrange COCO 2014 in the expected structure:

```text
DeepLearning/
├── train2014/
├── val2014/
├── test2014/
└── annotations/
    ├── captions_train2014.json
    ├── captions_val2014.json
    └── image_info_test2014.json
```

If you are using Google Drive, update the dataset path inside the notebooks, for example:

```python
BASE_DIR = "/content/drive/MyDrive/DeepLearning"
```

### 4. Train the Custom Model

Use:

```text
Training-5k.ipynb
```

for a smaller training experiment, or:

```text
Traning-100k.ipynb
```

for a larger training run.

The custom training notebook includes:

* Vocabulary building
* Image transformations
* Dataset loading
* Encoder–decoder model definition
* Training and validation loops
* Checkpoint saving
* Caption generation

### 5. Fine-Tune BLIP with LoRA

Use:

```text
blip_lora_coco2014.ipynb
```

This notebook fine-tunes BLIP using LoRA adapters and evaluates the generated captions using multiple metrics.

---

## Evaluation Metrics

The project uses several metrics to evaluate caption quality.

| Metric       | Purpose                                           |
| ------------ | ------------------------------------------------- |
| BLEU-1       | Measures unigram overlap with reference captions  |
| BLEU-2       | Measures bigram overlap                           |
| BLEU-3       | Measures trigram overlap                          |
| BLEU-4       | Measures 4-gram overlap                           |
| ROUGE-L      | Measures longest common subsequence similarity    |
| CIDEr        | Captioning-specific consensus metric              |
| CLIPScore    | Measures image-text alignment                     |
| RefCLIPScore | Combines image grounding and reference similarity |

Using both text-overlap and image-aware metrics gives a more complete evaluation of caption quality.

---

## Results

### BLIP + LoRA Results

The BLIP + LoRA experiment reports strong performance while training only a small percentage of the full model parameters.

| Metric               |       Value |
| -------------------- | ----------: |
| Total Parameters     | 249,183,548 |
| Trainable Parameters |   1,769,472 |
| Trainable Percentage |      0.710% |
| Best Validation Loss |      2.2763 |
| BLEU-1               |      0.7600 |
| BLEU-2               |      0.6027 |
| BLEU-3               |      0.4697 |
| BLEU-4               |      0.3654 |
| ROUGE-L              |      0.5800 |
| CIDEr                |      1.3512 |
| CLIPScore            |      0.7521 |
| RefCLIPScore         |      0.8066 |

These results show that LoRA can adapt a pretrained vision-language model efficiently while updating less than 1% of the model parameters.

---

## Model Checkpoints

Saved model checkpoints are included in:

```text
checkpoints.zip
```

Depending on the notebook, checkpoints may include:

* Best model weights
* Last model weights
* LoRA adapter weights
* Training configuration
* Validation loss records

To use a checkpoint, extract the zip file and update the checkpoint path inside the relevant notebook.

---

## Example Inference

A typical inference flow is:

```python
image = Image.open("sample.jpg").convert("RGB")
caption = generate_caption(model, image)
print(caption)
```

For the BLIP + LoRA model, the inference pipeline uses the BLIP processor and the fine-tuned LoRA adapter to generate captions from input images.

---

## Future Improvements

Possible improvements for this project include:

* Add a clean Python package structure instead of notebook-only execution
* Add `requirements.txt`
* Add sample input images and generated captions
* Add training logs and loss curves to the README
* Add a comparison table between the custom model and BLIP + LoRA
* Add Gradio or Streamlit demo interface
* Add Hugging Face model card
* Add support for other datasets such as Flickr8k or Flickr30k
* Add attention visualization for generated captions
* Add automated evaluation scripts

---

## License

This project is licensed under the MIT License.

---

## Author

**Ali Sedghiye**

GitHub: [AliSedghiye](https://github.com/AliSedghiye)

---

## Acknowledgements

This project uses open-source tools and models from the deep learning ecosystem, including:

* PyTorch
* TorchVision
* Hugging Face Transformers
* PEFT
* Salesforce BLIP
* COCO Dataset
* NLTK
* ROUGE Score
* pycocoevalcap

Special thanks to the open-source community for providing the tools and datasets that make multimodal deep learning research accessible.
