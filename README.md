# PrototypeSampling

This repository provides source code for the MICCAI2024 paper "Leveraging Image Captions for Selective Whole Slide Image Annotation" [[`arXiv`
](https://arxiv.org/abs/2407.06363)] [[`blogpost`](https://deepmicroscopy.org/leveraging-image-captions-for-streamlining-histopathology-image-annotation-miccai-2024-paper/)].

## Usage
* Class prototype identificaition via [keyword_search.py](code/Class_Prototypes_Identification/keyword_search.py) or [text_to_image_retrieval.py](code/Class_Prototypes_Identification/text_to_image_retrieval.py).
* You may run [experiments.py](code/Class_Prototypes_Identification/experiments.py) to reproduce [examples](examples).

## Reference Code for downstream tasks
* Breast tumor segmentation on CAMELYON16: https://github.com/DeepMicroscopy/AdaptiveRegionSelection
* Mitotic figure detection on MITOS_WSI_CMC: https://github.com/DeepMicroscopy/MITOS_WSI_CMC

## Reference Code for region selection methods (standard/adaptive)
https://github.com/DeepMicroscopy/AdaptiveRegionSelection
