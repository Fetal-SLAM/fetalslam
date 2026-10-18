
from kornia.feature.lightglue import LightGlue
from torch import nn
import torch
import os

class LighterGlue(nn.Module):
    """
        Lighter version of LightGlue :)
    """

    default_conf_xfeat = {
    "name": "xfeat",  # just for interfacing
    "input_dim": 64,  # input descriptor dimension (autoselected from weights)
    "descriptor_dim": 96,
    "add_scale_ori": False,
    "add_laf": False,  # for KeyNetAffNetHardNet
    "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
    "n_layers": 6,
    "num_heads": 1,
    "flash": True,  # enable FlashAttention if available.
    "mp": False,  # enable mixed precision
    "depth_confidence": -1,  # early stopping, disable with -1
    "width_confidence": 0.95,  # point pruning, disable with -1
    "filter_threshold": 0.1,  # match threshold
    "weights": None,
    }

    def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat-lighterglue.pt'):
        super().__init__()
        self.net = LightGlue(features=None, **self.default_conf_xfeat)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(weights):
            state_dict = torch.load(weights, map_location=self.dev, weights_only=True)
        else:
            raise FileNotFoundError(f'Missing bundled LighterGlue weights: {weights}')

        # Training checkpoints bundle the extractor, which XFeat loads separately.
        state_dict = {key: value for key, value in state_dict.items()
                      if not key.startswith('extractor.')}
        # rename old state dict entries
        for i in range(self.net.conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}

        # This deterministic early-stop buffer is registered by newer Kornia.
        if 'confidence_thresholds' in self.net.state_dict():
            state_dict.setdefault('confidence_thresholds', self.net.confidence_thresholds)
        self.net.load_state_dict(state_dict, strict=True)
        self.net.to(self.dev).eval()

    @torch.inference_mode()
    def forward(self, data, min_conf = 0.1):
        self.net.conf.filter_threshold = min_conf
        result = self.net( {   'image0': {'keypoints': data['keypoints0'], 'descriptors': data['descriptors0'], 'image_size': data['image_size0']},
                               'image1': {'keypoints': data['keypoints1'], 'descriptors': data['descriptors1'], 'image_size': data['image_size1']}  
                           } )
        return result
