# Meme Analyser

A multi-modal AI system for comprehensive analysis and virality potential assessment of internet memes — combining computer vision, OCR, sentiment analysis, and object detection into a single real-time pipeline.

![Gradio Interface](assets/interface_screenshot.png)

## Overview

Internet memes communicate meaning through a tight interplay of image and text, making them notoriously hard to analyze with a single model. Meme Analyser tackles this by decomposing the problem into five specialized, interconnected modules whose outputs are synthesized into one cohesive, human-readable analysis — delivered through an interactive Gradio dashboard in 3–5 seconds per image.

## Features

- **Meme Classification** — A fine-tuned Vision Transformer (`vit-base-patch16-224-in21k`) classifies overall emotional tone (`very_positive`, `positive`, `neutral`, `negative`, `very_negative`), trained on 6,992 labeled meme images.
- **Advanced OCR Pipeline** — OpenCV-based pre-processing (grayscale conversion, 2x upscaling, Otsu binarization, median blur) significantly boosts Tesseract's accuracy on stylized, low-resolution meme text.
- **Sentiment Analysis** — Extracted text is passed through DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) for fast, reliable sentiment scoring.
- **Object Detection** — Facebook's DETR (`detr-resnet-50`) identifies people and objects within the image using end-to-end transformer-based detection.
- **Virality Potential Heuristic** — A custom rule-based scoring engine synthesizes outputs from all four modules into a 0–100 "Virality Potential" score with a transparent, rule-by-rule breakdown.

## System Architecture

An uploaded image is processed in parallel by three primary pipelines — the ViT classifier, DETR object detector, and the OpenCV/Tesseract OCR pipeline. OCR output feeds into the DistilBERT sentiment model, and all four resulting signals (classification, objects, text, sentiment) feed into the heuristic virality engine. The interface renders all seven resulting data points in real time.

```
                              Uploaded Meme
   │
   ├──▶ ViT Classifier ─────────────────▶ Emotional tone + confidence ──┐
   │                                                                     │
   ├──▶ DETR Object Detection ──────────▶ Detected objects/people ──────┤
   │                                                                     ├──▶ Virality Heuristic Engine ──▶ Gradio Dashboard
   └──▶ OpenCV + Tesseract OCR ──▶ Extracted text                       │
                                       │                                 │
                                       └──▶ DistilBERT Sentiment ────────┘
```

## Output Dashboard

Each analysis populates seven read-only fields:

| Field | Description | Example |
|---|---|---|
| Virality Potential | Final heuristic score with qualitative label | `High Potential (Score: 80/100)` |
| Virality Analysis | Rule-by-rule rationale for the score | `Positive Content (+30), Text is Present (+20)...` |
| Meme Classification | Predicted emotional tone label | `positive` |
| Classification Confidence | Softmax confidence of the classifier | `92.45%` |
| Sentiment Analysis | Text-only sentiment result | `POSITIVE (99.8%)` |
| Extracted Text (OCR) | Cleaned, processed text from the meme | — |
| Identified Objects | Detected objects with confidence scores | `person (99%), tie (88%)` |

## Virality Scoring Rules

The heuristic score is a weighted sum of positive and negative factors, clamped between 0–100:

| Rule | Points |
|---|---|
| `very_positive` classification | +50 |
| `positive` classification | +30 |
| Presence of OCR-detected text | +20 |
| Object list includes "person" | +10 |
| Negative text sentiment | −10 |

## Tech Stack

- **Modeling**: PyTorch, Hugging Face `transformers`
- **Vision**: Vision Transformer (ViT), DETR (ResNet-50 backbone)
- **NLP**: DistilBERT
- **OCR**: Tesseract, OpenCV
- **Data**: pandas
- **Interface**: Gradio
- **Training Environment**: Google Colab (T4 GPU)

## Dataset

Trained on the [6992 Labeled Meme Images Dataset](https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset) (Kaggle), annotated across five sentiment-based classes.

## Repository Structure

```
meme-analysis-ds/
├── train.ipynb              # ViT fine-tuning pipeline
├── app.ipynb                # Gradio inference application
├── my-awesome-meme-classifier/  # Fine-tuned model checkpoints
├── images/                  # Dataset images
├── labels.csv                # Image-label mapping
└── labels/                   # Label metadata
```

## Getting Started

1. Clone the repository and open `train.ipynb` in Google Colab (or a local Jupyter environment with GPU support).
2. Run the training notebook to fine-tune the ViT classifier on the labeled dataset, or download pre-trained weights into `my-awesome-meme-classifier/`.
3. Launch `app.ipynb` to start the Gradio interface.
4. Upload a meme image and click **Analyze Meme** to view the full seven-part analysis.

## Limitations & Future Work

The Virality Potential score is a rule-based heuristic, not a data-driven prediction, so its accuracy remains unvalidated against real engagement metrics. All models are English-centric and may underperform on multi-lingual or culturally-specific memes. Planned improvements include:

- Replacing the heuristic with a data-driven model trained on scraped social engagement metrics
- Support for video-based memes (GIFs, short clips)
- Irony and sarcasm detection where text and image sentiment diverge
- Meme template identification and evolution tracking over time

## License

MIT License © 2026 Likhitha K M
