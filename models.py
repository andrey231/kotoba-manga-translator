import logging
import os
import sys
from functools import wraps
from threading import RLock

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

from image_io import clip_box

logger = logging.getLogger(__name__)


_MODEL_LOCK = RLock()
_CUDA_DLL_HANDLE = None


def _locked(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _MODEL_LOCK:
            return function(*args, **kwargs)

    return wrapped


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


BUBBLE_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"


BUBBLE_CLASSES = {0: "bubble", 1: "text_bubble", 2: "text_free"}


LAMA_REPOS = [
    ("deckyfx/anime-big-lama", "anime-manga-big-lama.pt"),
    ("df1412/anime-big-lama", "anime-manga-big-lama.pt"),
]


def load_inpainting_model():
    last_error = None
    for repo_id, filename in LAMA_REPOS:
        try:
            logger.debug(f"[lama] Loading {repo_id}/{filename}...")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model = torch.jit.load(model_path, map_location=DEVICE)
            model.eval()
            logger.debug(f"[lama] Loaded from {repo_id} ({DEVICE})")
            return model
        except Exception as e:
            logger.warning(f"[lama] Failed with {repo_id}: {e}")
            last_error = e
    logger.warning("[lama] ⚠ All sources unavailable, inpainting will fall back to cv2.inpaint")
    logger.warning(f"       Last error: {last_error}")
    return None


def load_detector():
    processor = AutoImageProcessor.from_pretrained(
        BUBBLE_MODEL_ID,
        size={"height": 960, "width": 960},
    )
    model = RTDetrV2ForObjectDetection.from_pretrained(BUBBLE_MODEL_ID)
    model.to(DEVICE).eval()
    return processor, model


_inpaint_model = None


_inpaint_loaded = False


_detector_processor = None


_detector_model = None


@_locked
def get_inpaint_model():
    global _inpaint_model, _inpaint_loaded
    if not _inpaint_loaded:
        _inpaint_model = load_inpainting_model()
        _inpaint_loaded = True
    if _inpaint_model is not None:
        _inpaint_model.to(DEVICE)
    return _inpaint_model


@_locked
def get_detector():
    global _detector_processor, _detector_model
    if _detector_model is None:
        _detector_processor, _detector_model = load_detector()
    _detector_model.to(DEVICE)
    return _detector_processor, _detector_model


OCR_HF_ID = "JustANormalTinkerer/hayai-ocr-v2.5-nova"


OCR_VISION_ID = "google/siglip2-base-patch16-naflex"


_ocr_processor = None


_ocr_model = None


_ocr_tokenizer = None


@_locked
def get_ocr_model():
    global _ocr_processor, _ocr_model, _ocr_tokenizer
    if _ocr_model is None:
        from transformers import AutoModel, PreTrainedTokenizerFast

        logger.debug("[ocr] Loading %s", OCR_HF_ID)
        processor = AutoImageProcessor.from_pretrained(OCR_VISION_ID)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(OCR_HF_ID)
        model = AutoModel.from_pretrained(OCR_HF_ID, trust_remote_code=True).to(DEVICE).eval()
        _ocr_processor, _ocr_model, _ocr_tokenizer = processor, model, tokenizer
        logger.debug("[ocr] Loaded on %s", DEVICE)
    _ocr_model.to(DEVICE)
    return _ocr_processor, _ocr_model, _ocr_tokenizer


@_locked
def release_gpu_memory():
    global _ctd_session
    if DEVICE != "cuda":
        return
    if _ocr_model is not None:
        decoder = getattr(_ocr_model, "decoder", None)
        mask_cache = getattr(decoder, "_mask_cache", None)
        if isinstance(mask_cache, dict):
            mask_cache.clear()
        module = sys.modules.get(type(_ocr_model).__module__)
        for name in ("_get_2d_visual_freqs", "_get_1d_text_freqs"):
            cached = getattr(module, name, None)
            if cached is not None and hasattr(cached, "cache_clear"):
                cached.cache_clear()
    for model in (_detector_model, _ocr_model, _inpaint_model):
        if model is not None:
            model.to("cpu")
    _ctd_session = None
    torch.cuda.empty_cache()


_ctd_session = None


_CTD_MODEL_ID = "mayocream/comic-text-detector-onnx"


_CTD_MODEL_FILE = "comic-text-detector.onnx"


_CTD_INPUT_SIZE = 1024


def _add_torch_cuda_dll_dir():
    global _CUDA_DLL_HANDLE
    if DEVICE != "cuda" or os.name != "nt":
        return
    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            if _CUDA_DLL_HANDLE is None:
                _CUDA_DLL_HANDLE = os.add_dll_directory(torch_lib)
    except Exception:
        pass


@_locked
def _get_ctd_session():
    global _ctd_session
    if _ctd_session is not None:
        return _ctd_session if _ctd_session is not False else None

    _add_torch_cuda_dll_dir()
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("[ctd] ⚠ Install: pip install onnxruntime-gpu  (или onnxruntime для CPU)")
        _ctd_session = False
        return None

    try:
        ckpt = hf_hub_download(_CTD_MODEL_ID, _CTD_MODEL_FILE)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if DEVICE == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(ckpt, providers=providers)
        _ctd_session = sess
        used = sess.get_providers()[0].replace("ExecutionProvider", "")
        logger.debug(f"[ctd] Loaded Comic Text Detector ({used})")
        return sess
    except Exception as e:
        logger.warning(f"[ctd] ⚠ Ошибка загрузки модели: {e}")
        _ctd_session = False
        return None


def _ctd_page_mask(img_cv: np.ndarray) -> np.ndarray | None:
    sess = _get_ctd_session()
    if sess is None:
        return None

    try:
        input_info = sess.get_inputs()[0]
        input_height, input_width = input_info.shape[-2:]
        input_height = input_height if isinstance(input_height, int) else _CTD_INPUT_SIZE
        input_width = input_width if isinstance(input_width, int) else _CTD_INPUT_SIZE
        tensor, content_size = prepare_ctd_input(img_cv, input_height, input_width)
        if input_info.type == "tensor(float16)":
            tensor = tensor.astype(np.float16)
        outputs = sess.run(None, {input_info.name: tensor})
        candidates = [out for out in outputs if out.ndim == 4 and out.shape[:2] == (1, 1)]
        if len(candidates) != 1:
            raise ValueError("CTD must return exactly one single-channel segmentation mask")
        return restore_ctd_mask(
            candidates[0], content_size, (input_height, input_width), img_cv.shape[:2]
        )
    except Exception as e:
        logger.warning(f"[ctd] inference error: {e}")
        return None


def prepare_ctd_input(
    image: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, tuple[int, int]]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("CTD expects a non-empty uint8 BGR image")
    scale = min(height / image.shape[0], width / image.shape[1])
    content_height = max(1, min(height, round(image.shape[0] * scale)))
    content_width = max(1, min(width, round(image.shape[1] * scale)))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (content_width, content_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:content_height, :content_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, (content_height, content_width)


def restore_ctd_mask(
    mask: np.ndarray,
    content_size: tuple[int, int],
    input_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    probabilities = mask[0, 0]
    if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("CTD mask must contain probabilities between zero and one")
    crop_height = max(
        1,
        min(
            probabilities.shape[0], round(content_size[0] * probabilities.shape[0] / input_size[0])
        ),
    )
    crop_width = max(
        1,
        min(
            probabilities.shape[1], round(content_size[1] * probabilities.shape[1] / input_size[1])
        ),
    )
    restored = cv2.resize(
        probabilities[:crop_height, :crop_width],
        (output_size[1], output_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return (restored > 0.5).astype(np.uint8) * 255


def detect_bubbles(image_pil: Image.Image, threshold: float = 0.5) -> list[dict]:
    processor, model = get_detector()
    w, h = image_pil.size
    inputs = processor(images=image_pil.convert("RGB"), return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[h, w]], device=model.device),
        threshold=threshold,
    )[0]

    bubbles = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if int(label) not in BUBBLE_CLASSES or not torch.isfinite(box).all():
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1, x2, y2 = clip_box(x1, y1, x2 - x1, y2 - y1, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        bubbles.append(
            {
                "class": BUBBLE_CLASSES[int(label)],
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(score),
                "speaker": "unknown",
                "gender": "unknown",
            }
        )
    return bubbles
