import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms

from src.cnn.resnet_model import PlantDiseaseResNet
from src.cnn.class_names import class_mapping


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

model = PlantDiseaseResNet(num_classes=38)

model.load_state_dict(
    torch.load(
        "models/cnn/plant_disease_resnet18_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(image_tensor, class_id):

    # --------------------------------------------------------
    # We need gradients with respect to the final
    # convolutional feature maps.
    # --------------------------------------------------------

    feature_maps = None

    def capture_features(module, inputs, output):
        nonlocal feature_maps
        feature_maps = output

    # Target layer
    target_layer = model.model.layer4[-1].conv2

    # Register forward hook
    hook = target_layer.register_forward_hook(
        capture_features
    )

    # Make sure gradients are enabled
    image_tensor = image_tensor.clone().detach()
    image_tensor.requires_grad_(True)

    # Forward pass
    output = model(image_tensor)

    # Remove hook
    hook.remove()

    if feature_maps is None:
        raise RuntimeError(
            "Could not capture feature maps."
        )

    # --------------------------------------------------------
    # Get score for selected class
    # --------------------------------------------------------

    score = output[0, class_id]

    # --------------------------------------------------------
    # Calculate gradient of class score with respect
    # to feature maps
    # --------------------------------------------------------

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=feature_maps,
        retain_graph=False,
        create_graph=False,
        allow_unused=False
    )[0]

    if gradients is None:
        raise RuntimeError(
            "Could not calculate Grad-CAM gradients."
        )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    activations = feature_maps[0]
    gradients = gradients[0]

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    weights = gradients.mean(
        dim=(1, 2)
    )

    # --------------------------------------------------------
    # Weighted combination of feature maps
    # --------------------------------------------------------

    cam = torch.zeros(
        activations.shape[1:],
        dtype=activations.dtype,
        device=activations.device
    )

    for i, weight in enumerate(weights):

        cam += weight * activations[i]

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = torch.relu(cam)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    return cam.detach().cpu().numpy()


# ============================================================
# CREATE GRAD-CAM IMAGE
# ============================================================

def create_gradcam(image_path, output_path):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    image_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


    class_id = predicted_class.item()

    confidence_value = confidence.item()

    disease_name = class_mapping[class_id]


    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    cam = generate_gradcam(
        image_tensor,
        class_id
    )


    # --------------------------------------------------------
    # Resize CAM
    # --------------------------------------------------------

    cam = cv2.resize(
        cam,
        original_image.size
    )


    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # --------------------------------------------------------
    # Convert original image
    # --------------------------------------------------------

    original = np.array(
        original_image
    )

    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        output_path,
        overlay
    )


    return disease_name, confidence_value


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "Enter the path to a leaf image: "
    ).strip()


    output_path = (
        "models/cnn/gradcam_result.jpg"
    )


    disease, confidence = create_gradcam(
        image_path,
        output_path
    )


    print("\n==============================")
    print("GRAD-CAM EXPLANATION")
    print("==============================")


    print(
        "Predicted Disease:",
        disease
    )


    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


    print(
        "\nGrad-CAM saved to:"
    )


    print(
        output_path
    )
