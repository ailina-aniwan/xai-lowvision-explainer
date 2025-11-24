from typing import Tuple

import numpy as np
import torch
from PIL import Image
from matplotlib import cm
from torchvision import models
from torchvision.models import ResNet18_Weights


def load_model_and_transform():
    """
    Load a pretrained ResNet18 model, its associated transforms, and class labels.

    Returns:
        model (torch.nn.Module): ResNet18 in eval mode (CPU).
        transform (callable): Preprocessing transform for PIL images.
        class_names (list[str]): ImageNet class names.
    """
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    model.to("cpu")

    # Use the standard preprocessing associated with the weights
    transform = weights.transforms()
    class_names = weights.meta["categories"]

    return model, transform, class_names


class GradCAM:
    """
    Minimal Grad-CAM implementation for ResNet-like models.
    Hooks into a given target layer to capture activations and gradients.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Save feature maps from the forward pass
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Gradients with respect to the feature maps
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM for the top predicted class.

        Args:
            input_tensor (torch.Tensor): Shape [1, 3, H, W].

        Returns:
            cam (np.ndarray): 2D Grad-CAM heatmap (values in [0, 1]).
            class_idx (int): Index of the predicted class.
            prob (float): Confidence for that class (0–1).
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # [1, num_classes]
        probs = torch.softmax(output, dim=1)

        class_idx = int(torch.argmax(probs, dim=1).item())
        score = output[0, class_idx]

        # Backpropagate from the chosen class score
        score.backward()

        # activations: [1, C, H, W], gradients: [1, C, H, W]
        activations = self.activations[0]  # [C, H, W]
        gradients = self.gradients[0]  # [C, H, W]

        # Global average pooling of gradients over spatial dimensions
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted sum over channels
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_np = cam.cpu().numpy()
        prob = float(probs[0, class_idx].item())

        return cam_np, class_idx, prob


def overlay_heatmap_on_image(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on top of the original image.

    Args:
        image (PIL.Image.Image): Original RGB image.
        cam (np.ndarray): 2D heatmap with values in [0, 1].
        alpha (float): Blending factor for heatmap overlay.

    Returns:
        overlayed (PIL.Image.Image): Image with Grad-CAM heatmap overlaid.
    """
    image = image.convert("RGB")
    width, height = image.size

    # Resize cam to match the image size
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (width, height),
        resample=Image.BILINEAR,
    )

    cam_resized_np = np.array(cam_resized).astype(np.float32) / 255.0

    # Apply a colormap (e.g. "jet")
    colormap = cm.get_cmap("jet")
    colored_cam = colormap(cam_resized_np)  # [H, W, 4] RGBA in [0,1]
    colored_cam = (colored_cam[..., :3] * 255).astype(np.uint8)  # drop alpha

    heatmap_img = Image.fromarray(colored_cam)

    # Blend heatmap with original image
    overlayed = Image.blend(image, heatmap_img, alpha)
    return overlayed


def get_spatial_description(cam: np.ndarray) -> str:
    """
    Describe the approximate location of the most important region
    in the image based on the Grad-CAM heatmap.

    We compute a weighted center of mass and map it to a 3x3 grid.

    Args:
        cam (np.ndarray): 2D heatmap (values in [0, 1]).

    Returns:
        description (str): e.g., "top-left", "center-right", "bottom-center", etc.
    """
    if cam.ndim != 2:
        cam = cam.squeeze()

    h, w = cam.shape
    if h == 0 or w == 0:
        return "center"

    # Add small epsilon to avoid divide-by-zero
    eps = 1e-8
    weights = cam.astype(np.float64) + eps

    y_indices, x_indices = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )

    # Weighted average of coordinates
    y_mean = (y_indices * weights).sum() / weights.sum()
    x_mean = (x_indices * weights).sum() / weights.sum()

    # Normalize [0,1]
    y_norm = y_mean / max(h - 1, 1)
    x_norm = x_mean / max(w - 1, 1)

    # Map to top/center/bottom and left/center/right
    if y_norm < 1.0 / 3.0:
        vertical = "top"
    elif y_norm < 2.0 / 3.0:
        vertical = "center"
    else:
        vertical = "bottom"

    if x_norm < 1.0 / 3.0:
        horizontal = "left"
    elif x_norm < 2.0 / 3.0:
        horizontal = "center"
    else:
        horizontal = "right"

    if vertical == "center" and horizontal == "center":
        return "center"
    return f"{vertical}-{horizontal}"
