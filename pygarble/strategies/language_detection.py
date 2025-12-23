import os
import urllib.request
from typing import Any, Optional

from .base import BaseStrategy


class LanguageDetectionStrategy(BaseStrategy):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._model: Optional[Any] = None
        self._model_path: str = self._get_model_path()

    def _get_model_path(self) -> str:
        custom_path = self.kwargs.get("model_path")
        if custom_path:
            return custom_path

        model_dir = os.path.expanduser("~/.pygarble/models")
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, "lid.176.bin")

    def _download_model(self) -> None:
        if os.path.exists(self._model_path):
            return

        custom_path = self.kwargs.get("model_path")
        if custom_path:
            raise FileNotFoundError(
                f"Model file not found at custom path: {self._model_path}"
            )

        model_url = (
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/"
            "lid.176.bin"
        )
        print(
            f"Downloading FastText language detection model to "
            f"{self._model_path}..."
        )
        urllib.request.urlretrieve(model_url, self._model_path)
        print("Model download completed.")

    def _load_model(self) -> None:
        if self._model is None:
            try:
                import fasttext
                import numpy as np

                self._download_model()
                self._model = fasttext.load_model(self._model_path)

                # Monkey patch to fix NumPy 2.0 compatibility issue
                original_predict = self._model.predict

                def patched_predict(text, k=1):
                    labels, probs = original_predict(text, k)
                    # Convert to list to avoid NumPy 2.0 copy=False issue
                    if isinstance(probs, np.ndarray):
                        probs = probs.tolist()
                    return labels, probs

                self._model.predict = patched_predict

            except ImportError:
                raise ImportError(
                    "fasttext is required for LanguageDetectionStrategy. "
                    "Install it with: pip install fasttext-wheel"
                )

    def _predict_impl(self, text: str) -> bool:
        self._load_model()

        predictions = self._model.predict(text, k=1)
        labels, scores = predictions

        if not labels or not scores:
            return True

        label = labels[0]
        score = float(scores[0])

        target_lang = self.kwargs.get("target_language", "en")

        if label == f"__label__{target_lang}":
            return score < 0.5
        else:
            return True

    def _predict_proba_impl(self, text: str) -> float:
        self._load_model()

        predictions = self._model.predict(text, k=1)
        labels, scores = predictions

        if not labels or not scores:
            return 1.0

        label = labels[0]
        score = float(scores[0])

        target_lang = self.kwargs.get("target_language", "en")

        if label == f"__label__{target_lang}":
            return float(1.0 - score)
        else:
            return 1.0
