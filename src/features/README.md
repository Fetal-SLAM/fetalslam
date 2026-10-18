# XFeat And LighterGlue Inference

This directory contains an adapted inference subset and bundled model weights
from [XFeat / accelerated_features](https://github.com/verlab/accelerated_features).
Training scripts, evaluation datasets, upstream demo programs and figures are
not included in this release.

Authors of XFeat: Guilherme Potje, Felipe Cadar, Andre Araujo, Renato Martins,
and Erickson R. Nascimento. See the retained [upstream license](LICENSE).
LightGlue inference is provided by Kornia; its upstream licenses and authorship
remain applicable to that dependency.

The SLAM adapter extracts XFeat descriptors and caches the original image size
as `(width, height)`. It uses XFeat-trained LighterGlue weights for matching,
then applies homography verification. The upstream MNN utilities remain in
`modules/xfeat.py`, but the maintained SLAM adapter does not call them.

Local adaptations include serialized inference, relative package imports,
image-size metadata, local checkpoint loading, and explicit checkpoint-key
compatibility handling. Missing learned parameters are rejected rather than
silently left randomly initialized. The deterministic Kornia early-stop buffer
is initialized from the model configuration when absent in the checkpoint.

```bibtex
@INPROCEEDINGS{potje2024cvpr,
  author={Guilherme Potje and Felipe Cadar and Andre Araujo and Renato Martins and Erickson R. Nascimento},
  booktitle={2024 IEEE / CVF Computer Vision and Pattern Recognition (CVPR)},
  title={XFeat: Accelerated Features for Lightweight Image Matching},
  year={2024}
}
```
