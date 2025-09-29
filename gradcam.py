import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: pytorch model
        target_layer: the conv layer module to hook (ex: model.features[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: 1xC x H x W tensor on device
        """
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        loss = out[0, class_idx]
        loss.backward(retain_graph=True)
        grads = self.gradients[0].cpu().numpy()  # C x H x W
        acts = self.activations[0].cpu().numpy() # C x H x W
        weights = np.mean(grads, axis=(1,2))  # C
        cam = np.zeros(acts.shape[1:], dtype=np.float32)  # H x W
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max()-cam.min() + 1e-8)
        return cam

def overlay_cam_on_pil(pil_img, cam, alpha=0.4):
    img = np.array(pil_img).astype(np.float32)/255.0
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)[:,:,::-1]/255.0
    over = img*(1-alpha) + heatmap*alpha
    over = np.clip(over, 0, 1)
    return Image.fromarray((over*255).astype('uint8'))
