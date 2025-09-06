#!/usr/bin/env python3
"""
CLIP-based Glasses Detection with Multiple Prompt Strategies
"""

import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Tuple, Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlassesDetector:
    """
    Glasses detection using CLIP with multiple prompt strategies for improved accuracy.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = None,
        image_size: int = 384
    ):
        """
        Initialize the glasses detector.
        
        Args:
            model_name: CLIP model to use
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            image_size: Maximum image size for processing
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        logger.info(f"Initializing {model_name} on {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Define prompt strategies
        self.prompt_strategies = [
            # Strategy 1: Direct glasses description
            {
                "glasses": [
                    "a photo of a person wearing glasses",
                    "a photo of a person wearing eyeglasses",
                    "a photo of a person with spectacles on their face",
                    "a photo of a person wearing prescription glasses"
                ],
                "no_glasses": [
                    "a photo of a person with no glasses",
                    "a photo of a person without any eyewear",
                    "a photo of a person with bare face",
                    "a photo of a person not wearing glasses"
                ]
            },
            # Strategy 2: Focus on eye area
            {
                "glasses": [
                    "eyes behind glass lenses",
                    "eyes with glasses frames",
                    "eyes visible through eyeglasses",
                    "glasses on face"
                ],
                "no_glasses": [
                    "bare eyes without glasses",
                    "natural eyes with no eyewear",
                    "unobstructed view of eyes",
                    "eyes without any glasses"
                ]
            },
            # Strategy 3: Fashion/accessory angle
            {
                "glasses": [
                    "face with eyewear accessory",
                    "person wearing vision correction glasses",
                    "face with optical frames",
                    "wearing spectacles"
                ],
                "no_glasses": [
                    "face without any accessories",
                    "natural face appearance",
                    "unadorned facial features",
                    "no eyewear on face"
                ]
            }
        ]
    
    def detect(self, image_path: str) -> Tuple[Optional[bool], float]:
        """
        Detect if a person in the image is wearing glasses.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (has_glasses, confidence)
            - has_glasses: True if glasses detected, False if not, None if error
            - confidence: Confidence score between 0 and 1
        """
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return None, 0.0
            
            # Run detection with all strategies
            all_scores = []
            
            for strategy in self.prompt_strategies:
                score = self._evaluate_strategy(image, strategy)
                if score is not None:
                    all_scores.append(score)
            
            if not all_scores:
                return None, 0.0
            
            # Combine scores from all strategies
            final_score = np.mean(all_scores)
            has_glasses = final_score > 0.5
            confidence = final_score if has_glasses else (1 - final_score)
            
            return has_glasses, confidence
            
        except Exception as e:
            logger.warning(f"Error processing {image_path}: {e}")
            return None, 0.0
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess an image."""
        try:
            if not os.path.exists(image_path):
                return None
            
            image = Image.open(image_path).convert("RGB")
            
            # Resize if needed
            if max(image.size) > self.image_size:
                ratio = self.image_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def _evaluate_strategy(
        self,
        image: Image.Image,
        strategy: Dict[str, List[str]]
    ) -> Optional[float]:
        """Evaluate a single prompt strategy."""
        try:
            # Combine all prompts
            all_prompts = strategy["glasses"] + strategy["no_glasses"]
            
            # Process with CLIP
            inputs = self.processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            # Calculate glasses probability
            n_glasses_prompts = len(strategy["glasses"])
            glasses_prob = np.max(probs[:n_glasses_prompts])
            no_glasses_prob = np.max(probs[n_glasses_prompts:])
            
            # Normalize
            total = glasses_prob + no_glasses_prob
            if total > 0:
                return glasses_prob / total
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Strategy evaluation failed: {e}")
            return None
    
    def detect_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Tuple[Optional[bool], float]]:
        """
        Detect glasses in multiple images.
        
        Args:
            image_paths: List of image file paths
            show_progress: Whether to show progress bar
            
        Returns:
            List of (has_glasses, confidence) tuples
        """
        results = []
        
        iterator = image_paths
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="Detecting glasses")
        
        for image_path in iterator:
            has_glasses, confidence = self.detect(image_path)
            results.append((has_glasses, confidence))
            
            # Clear cache periodically
            if len(results) % 100 == 0 and self.device == "cuda":
                torch.cuda.empty_cache()
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect glasses in images")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14",
                       help="CLIP model to use")
    parser.add_argument("--device", choices=["cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = GlassesDetector(model_name=args.model, device=args.device)
    
    # Detect glasses
    has_glasses, confidence = detector.detect(args.image)
    
    if has_glasses is not None:
        result = "WEARING GLASSES" if has_glasses else "NO GLASSES"
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.1%}")
    else:
        print("Detection failed")


if __name__ == "__main__":
    main()