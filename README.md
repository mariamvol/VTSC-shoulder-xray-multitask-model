# VTSC-shoulder-xray-multitask-model
Inference-only pipeline for automated analysis of shoulder X-ray studies: projection routing, object detection, segmentation, fracture classification, foreign body detection and neck-shaft angle estimation.

The system processes a ZIP archive containing X-ray images and produces:

- projection classification;
- bone and implant/object detection;
- ROI-based bone segmentation;
- foreign body / metal construction detection;
- humerus fracture classification;
- greater tubercle fracture classification;
- neck-shaft angle estimation;
- human-readable report;
- downloadable ZIP report.

> This project is intended for research and educational use only. It is not a medical device and must not be used as a standalone diagnostic system.

## Pipeline

The unified inference pipeline uses a single bundled checkpoint:

```text
VTSC_unified_bundle.pt
```

## Internally, the bundle contains multiple specialized models:

- projection router;
- D/S foreign body classifiers;
- D/S bone detectors;
- D/S ROI segmentation models;
- D/S humerus fracture classifiers;
- ROI detector;
- greater tubercle fracture classifier;
- neck-shaft angle regressor.
- Input format

## Upload a .zip archive with the following structure:
```text
101.zip
└── 101/
    ├── 1/
    │   ├── IMG-0001-00001.jpg
    │   └── IMG-0001-00002.jpg
    └── 2/
        └── IMG-0003-00001.jpg
```

## Installation
```bash
pip install -r requirements.txt
```
## Model weights

Download VTSC_unified_bundle.pt from the Releases page and place it here:
```text
weights/VTSC_unified_bundle.pt
```

## Run Gradio app
```bash
python app.py
```
Then open the local URL printed in the terminal.

## Outputs

The application generates:

- human-readable conclusion;
- image-level table;
- colored detection and segmentation visualizations;
- JSON summary;
- downloadable ZIP report.

## License
This project is released under the MIT License.

## Disclaimer

This software is provided for research and educational purposes only. The automatic output is not a medical diagnosis. Final interpretation must be performed by a qualified medical specialist.
