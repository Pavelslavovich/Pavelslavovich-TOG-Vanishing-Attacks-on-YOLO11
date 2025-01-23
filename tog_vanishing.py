from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import os
from typing import List, Tuple

# Path to images for testing
data = [
    "/home/pavel/isp2025/val2017/images_1/1.jpg",
    "/home/pavel/isp2025/val2017/images_1/2.jpg",
    "/home/pavel/isp2025/val2017/images_1/3.jpg",
    "/home/pavel/isp2025/val2017/images_1/4.jpg",
]

class AdversarialYOLO(YOLO):
    def __init__(self, model_path: str, epsilon: float = 4 / 255, step_size: float = 1 / 255, num_steps: int = 10):
        super().__init__(model_path)
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps

    def attack(self, image: torch.Tensor) -> torch.Tensor:
        """Perform adversarial attack on the input image."""
        image = image.clone().detach().requires_grad_(True).to(self.device)

        for _ in range(self.num_steps):
            predictions = self.model(image)

            # Handle different prediction formats
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            elif hasattr(predictions, "boxes"):
                predictions = predictions.boxes
            else:
                raise TypeError(f"Unexpected predictions type: {type(predictions)}")

            # Calculate losses
            objectness = predictions[:, 4]  # Confidence in object detection
            class_confidence = predictions[:, 5:].max(1)[0]  # Max class confidence
            total_loss = objectness.sum() + class_confidence.sum()

            # Backpropagate
            self.model.zero_grad()
            total_loss.backward()

            # Apply perturbation
            with torch.no_grad():
                perturbation = self.step_size * image.grad.sign()
                image = torch.clamp(image - perturbation, 0, 1).detach().requires_grad_(True)

        return image

def resize_with_padding(img: Image.Image, target_size: Tuple[int, int] = (640, 640)) -> Image.Image:
    """Resize the image to fit within the target size, adding black padding if necessary."""
    original_width, original_height = img.size
    target_width, target_height = target_size

    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    img_resized = img.resize((new_width, new_height))
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(img_resized, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    return new_img

def process_image(img: Image.Image) -> torch.Tensor:
    """Resize the image with padding and convert it to a tensor."""
    img_resized = resize_with_padding(img, target_size=(640, 640))
    img_tensor = torch.tensor(
        np.array(img_resized).transpose(2, 0, 1) / 255.0, dtype=torch.float32
    ).unsqueeze(0)
    return img_tensor

def visualize_and_save_attacks(
    original_images: List[Image.Image],
    attacked_images: List[torch.Tensor],
    model_path: str,
    save_dir: str = "output",
):
    """Visualize and save results of adversarial attacks."""
    os.makedirs(save_dir, exist_ok=True)
    attack_dir = os.path.join(save_dir, "attacked_images")
    os.makedirs(attack_dir, exist_ok=True)

    model = YOLO(model_path)

    for i, (orig, adv) in enumerate(zip(original_images, attacked_images)):
        orig_resized = resize_with_padding(orig, target_size=(640, 640))

        result_orig = model.predict(orig_resized, imgsz=(640, 640))[0]
        result_adv = model.predict(adv, imgsz=(640, 640))[0]

        img_orig = result_orig.plot(line_width=2)
        img_adv = result_adv.plot(line_width=2)

        adv_img_np = adv.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255
        adv_img_np = adv_img_np.astype(np.uint8)

        attacked_image_path = os.path.join(attack_dir, f"attacked_{i}.png")
        Image.fromarray(adv_img_np).save(attacked_image_path)

        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        axs[0].imshow(orig_resized)
        axs[0].set_title("Original Image", fontsize=12)
        axs[0].axis("off")

        axs[1].imshow(img_orig[..., ::-1])  
        axs[1].set_title("Original Image with Detection", fontsize=12)
        axs[1].axis("off")

        axs[2].imshow(img_adv)  
        axs[2].set_title("Attacked Image with Detection", fontsize=12)
        axs[2].axis("off")

        visualization_path = os.path.join(save_dir, f"visualization_{i}.png")
        plt.tight_layout()
        plt.savefig(visualization_path)
        plt.close()

        print(f"Visualization saved to {visualization_path}")
        print(f"Attacked image saved to {attacked_image_path}")

def main():
    model_path = "yolo11n.pt"
    adversarial_model = AdversarialYOLO(model_path)

    original_images = [Image.open(img_path) for img_path in data]
    attacked_images = [
        adversarial_model.attack(process_image(img)) for img in original_images
    ]

    visualize_and_save_attacks(original_images, attacked_images, model_path)

if __name__ == "__main__":
    main()
