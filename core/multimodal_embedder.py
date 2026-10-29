from pathlib import Path
from typing import Union, List
import numpy as np
import torch
from PIL import Image
import cv2

class MultimodalEmbedder:
    """
    Multimodal semantic encoder for text, images, and video frames using CLIP / SigLIP.
    Returns unit-normalized (L2) 512D or 768D embeddings suitable for cosine similarity.
    """
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._embedding_dim = 512

    def _init_model(self):
        if self._model is not None:
            return

        try:
            import open_clip
            print(f"[MultimodalEmbedder] Loading OpenCLIP {self.model_name} ({self.pretrained}) on {self.device}...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model = model.eval()
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._embedding_dim = getattr(model, "visual", None) and getattr(model.visual, "output_dim", 512) or 512
            print(f"[MultimodalEmbedder] Model ready. Output dimension: {self._embedding_dim}")
        except Exception as e:
            print(f"[MultimodalEmbedder] OpenCLIP load warning: {e}. Trying SentenceTransformers / Transformers...")
            try:
                from transformers import CLIPProcessor, CLIPModel
                self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
                self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._tokenizer = None
                self._embedding_dim = 512
            except Exception as te:
                print(f"[MultimodalEmbedder] Fallback warning: {te}. Using deterministic mock encoder.")
                self._model = "mock"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds a single string or list of text strings into normalized vectors.
        """
        self._init_model()
        is_single = isinstance(text, str)
        text_list = [text] if is_single else text

        if self._model == "mock":
            # Deterministic pseudo-embedding for testing / fallback
            np.random.seed(abs(hash(" ".join(text_list))) % (2**32))
            arr = np.random.randn(len(text_list), self._embedding_dim).astype(np.float32)
            arr /= np.linalg.norm(arr, axis=-1, keepdims=True)
            return arr[0] if is_single else arr

        with torch.no_grad():
            if self._tokenizer is not None:
                tokens = self._tokenizer(text_list).to(self.device)
                features = self._model.encode_text(tokens)
            else:
                inputs = self._preprocess(text=text_list, return_tensors="pt", padding=True).to(self.device)
                features = self._model.get_text_features(**inputs)

            features /= features.norm(dim=-1, keepdim=True)
            arr = features.cpu().numpy().astype(np.float32)
            return arr[0] if is_single else arr

    def embed_image(self, image_input: Union[Path, str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Embeds an image (Path, PIL Image, or OpenCV BGR frame) into a normalized vector.
        """
        self._init_model()

        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Convert BGR (OpenCV) to RGB
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        if self._model == "mock":
            np.random.seed(abs(hash(img.tobytes()[:100])) % (2**32))
            arr = np.random.randn(self._embedding_dim).astype(np.float32)
            return arr / np.linalg.norm(arr)

        with torch.no_grad():
            if self._tokenizer is not None:
                tensor = self._preprocess(img).unsqueeze(0).to(self.device)
                features = self._model.encode_image(tensor)
            else:
                inputs = self._preprocess(images=img, return_tensors="pt").to(self.device)
                features = self._model.get_image_features(**inputs)

            features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype(np.float32)[0]

    def embed_video_preview(self, video_path: Union[Path, str], num_frames: int = 4) -> np.ndarray:
        """
        Samples representative frames from a video file, embeds them,
        and aggregates into a single representative aesthetic/visual embedding vector.
        """
        v_path = Path(video_path).resolve()
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        cap = cv2.VideoCapture(str(v_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            # Fallback to single image or random vector
            return self.embed_text(v_path.stem)

        # Sample evenly spaced frame indices
        step = max(1, total_frames // (num_frames + 1))
        indices = [step * (i + 1) for i in range(num_frames)]

        frame_embeddings = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
            ret, frame = cap.read()
            if ret and frame is not None:
                emb = self.embed_image(frame)
                frame_embeddings.append(emb)

        cap.release()

        if not frame_embeddings:
            return self.embed_text(v_path.stem)

        # Mean pooled normalized embedding
        mean_vec = np.mean(frame_embeddings, axis=0)
        norm = np.linalg.norm(mean_vec)
        return (mean_vec / norm).astype(np.float32) if norm > 1e-6 else mean_vec

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates cosine similarity between two 1D vectors.
        """
        a = vec_a.flatten()
        b = vec_b.flatten()
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-7 or norm_b < 1e-7:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))
