import asyncio
import base64
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests
import torch
from fastapi.testclient import TestClient
from PIL import Image

import bubbles
import image_io
import llm
import models
import ocr
import rendering
import translation
import web
from settings import Settings, ollama_endpoint, settings, use_settings


class ImageTests(unittest.TestCase):
    def test_crop_uses_original_intersection_and_handles_empty_region(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertEqual(image_io.clip_box(-5, -3, 7, 5, 10, 10), (0, 0, 2, 2))
        self.assertIsNone(image_io.prepare_ocr_crop(image, 20, 0, 5, 5))
        self.assertIsNone(image_io.prepare_ocr_crop(image, 0, 0, -1, 5))

    def test_large_and_thin_ocr_crops_are_bounded_rgb(self):
        for height, width in ((2000, 900), (1, 5000)):
            image = np.full((height, width, 3), (10, 20, 230), dtype=np.uint8)
            for enhanced in (True, False):
                crop = image_io.prepare_ocr_crop(image, 0, 0, width, height, enhanced)
                self.assertEqual(crop.mode, "RGB")
                self.assertLessEqual(max(crop.size), image_io.OCR_MAX_SIDE)
                decoded = Image.open(io.BytesIO(base64.b64decode(image_io.encode_image(crop))))
                self.assertEqual(decoded.size, crop.size)
                if not enhanced:
                    self.assertEqual(
                        crop.getpixel((crop.width // 2, crop.height // 2)), (230, 20, 10)
                    )

    def test_unicode_paths_transparency_and_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "страница.png")
            Image.new("RGBA", (4, 5), (255, 0, 0, 0)).save(path)
            decoded = image_io.read_image(path)
            self.assertTrue(np.all(decoded == 255))
            image_io.write_image(path, decoded)
            np.testing.assert_array_equal(image_io.read_image(path), decoded)
            image = Image.new("RGB", (4, 7))
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(image_io.read_image(path).shape, (4, 7, 3))


class ModelContractTests(unittest.TestCase):
    def test_ctd_receives_rgb_nchw_normalized_contiguous_input(self):
        image = np.full((2, 4, 3), (10, 20, 255), dtype=np.uint8)
        tensor, content = models.prepare_ctd_input(image, 8, 8)
        self.assertEqual(tensor.shape, (1, 3, 8, 8))
        self.assertEqual(content, (4, 8))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 0, 0], [1, 20 / 255, 10 / 255])
        self.assertTrue(np.all(tensor[:, :, 4:] == 0))

    def test_ctd_selects_segmentation_and_maps_lower_resolution_output(self):
        session = Mock()
        session.get_inputs.return_value = [
            SimpleNamespace(name="images", shape=[1, 3, 8, 8], type="tensor(float)")
        ]
        mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
        mask[:, :, :2] = 1
        session.run.return_value = [np.zeros((1, 2, 4, 4)), np.zeros((1, 100, 7)), mask]
        with patch.object(models, "_get_ctd_session", return_value=session):
            result = models._ctd_page_mask(np.zeros((20, 40, 3), dtype=np.uint8))
        self.assertEqual(result.shape, (20, 40))
        self.assertTrue(np.all(result == 255))

    def test_detector_clamps_boxes_and_drops_empty_or_unknown_classes(self):
        processor = Mock()
        processor.return_value.to.return_value = {}
        processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.9, 0.8, 0.7]),
                "labels": torch.tensor([1, 1, 99]),
                "boxes": torch.tensor([[-5, -5, 7, 8], [12, 12, 14, 14], [0, 0, 5, 5]]),
            }
        ]
        model = Mock(device="cpu")
        with patch.object(models, "get_detector", return_value=(processor, model)):
            result = models.detect_bubbles(Image.new("L", (10, 10)))
        self.assertEqual(len(result), 1)
        self.assertEqual([result[0][key] for key in ("x", "y", "width", "height")], [0, 0, 7, 8])
        self.assertEqual(processor.call_args.kwargs["images"].mode, "RGB")

    def test_lama_padding_mask_and_unmasked_pixels(self):
        image = np.full((13, 17, 3), (10, 20, 30), dtype=np.uint8)
        mask = np.zeros((13, 17), dtype=np.uint8)
        mask[3:8, 4:9] = 255

        def model(image_tensor, mask_tensor):
            self.assertEqual(tuple(image_tensor.shape), (1, 3, 16, 24))
            self.assertEqual(tuple(mask_tensor.shape), (1, 1, 16, 24))
            self.assertEqual(set(mask_tensor.unique().tolist()), {0, 1})
            np.testing.assert_allclose(
                image_tensor[0, :, 0, 0].numpy(), [30 / 255, 20 / 255, 10 / 255]
            )
            return torch.ones_like(image_tensor)

        with (
            patch.object(rendering, "DEVICE", "cpu"),
            patch.object(rendering, "build_inpaint_mask", return_value=mask),
            patch.object(rendering, "get_inpaint_model", return_value=model),
        ):
            result = rendering.inpaint_page(image, [])
        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
        self.assertTrue(np.all(result[mask > 0] == 255))

    def test_masks_use_crop_storage_and_preserve_manual_angle(self):
        image = np.full((300, 500, 3), 255, dtype=np.uint8)
        bubble = {
            "x": 10,
            "y": 20,
            "width": 40,
            "height": 30,
            "translation": "Hello",
            "text_angle": 12,
        }
        with patch.object(
            rendering, "_ctd_page_mask", return_value=np.full((300, 500), 255, dtype=np.uint8)
        ):
            rendering._compute_text_masks(image, [bubble])
        self.assertEqual(bubble["_text_mask"].shape, (30, 40))
        self.assertEqual(bubble["text_angle"], 12)
        mask = rendering.build_inpaint_mask(image, [bubble])
        self.assertEqual(mask.shape, (300, 500))
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[25, 15], 255)

    def test_ocr_works_without_creating_crop_files(self):
        with (
            use_settings(Settings()),
            patch.object(ocr, "_ocr_call", return_value="Hello") as infer,
        ):
            text = ocr.ocr_region(np.zeros((20, 20, 3), dtype=np.uint8), 0, 0, 20, 20, 1, 1)
        self.assertEqual(text, "Hello")
        self.assertIsInstance(infer.call_args.args[0], Image.Image)

    def test_ocr_error_does_not_return_partial_text(self):
        response = Mock()
        response.iter_lines.return_value = [
            b'{"response":"partial\\n"}',
            b'{"error":"model crashed"}',
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(ocr.requests, "post", return_value=response):
            self.assertEqual(ocr._ocr_stream(Image.new("RGB", (4, 4))), "")


class TranslationTests(unittest.TestCase):
    def test_translation_batches_respect_size_and_keep_offsets(self):
        items = [(i, {"text": "文" * 400}) for i in range(4)]
        chunks = list(
            translation._translation_chunks(items, "", translation.MangaContext(), "English", {})
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual([index for _, chunk in chunks for index, _ in chunk], list(range(4)))
        for offset, chunk in chunks:
            self.assertEqual(offset, chunk[0][0])

    def test_oversized_single_translation_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            list(
                translation._translation_chunks(
                    [(0, {"text": "x" * 10000})], "", translation.MangaContext(), "English", {}
                )
            )

    def test_response_ids_never_fall_back_to_array_positions(self):
        chunk = [(0, {"text": "A"}), (1, {"text": "B"})]
        missing = []
        translation._apply_chunk_results([{"id": 2, "translation": "second"}], chunk, 0, missing)
        self.assertNotIn("translation", chunk[0][1])
        self.assertEqual(chunk[1][1]["translation"], "second")
        self.assertEqual(missing, [0])

    def test_duplicate_and_invalid_translations_are_retried(self):
        for results in (
            [{"id": 1, "translation": "a"}, {"id": 1, "translation": "b"}],
            [{"id": 1, "translation": ["bad"]}],
            [{"id": True, "translation": "bad"}],
        ):
            chunk = [(0, {"text": "original"})]
            missing = []
            translation._apply_chunk_results(results, chunk, 0, missing)
            self.assertEqual(missing, [0])

    def test_ollama_http_errors_are_not_empty_successes(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("model missing")
        with patch.object(llm.requests, "post", return_value=response):
            with self.assertRaises(requests.HTTPError):
                llm.ollama("missing", "Translate")


class SettingsTests(unittest.TestCase):
    def test_concurrent_jobs_have_independent_settings(self):
        async def worker(name):
            with use_settings(Settings(llm_model=name)):
                await asyncio.sleep(0)
                return await asyncio.to_thread(lambda: settings().llm_model)

        async def run():
            return await asyncio.gather(worker("one"), worker("two"))

        self.assertEqual(asyncio.run(run()), ["one", "two"])
        self.assertEqual(settings().llm_model, "")

    def test_invalid_settings_fail_early(self):
        for values in (
            {"chunk_size": 0},
            {"detect_threshold": float("nan")},
            {"translate_retries": -1},
        ):
            with self.assertRaises(ValueError):
                Settings(**values)
        self.assertEqual(ollama_endpoint("localhost:11434"), "http://localhost:11434/api/generate")
        self.assertEqual(
            ollama_endpoint("https://host/prefix/api/generate/"), "https://host/prefix/api/generate"
        )


class BubbleTests(unittest.TestCase):
    def test_moving_text_does_not_move_the_inpaint_region(self):
        bubble = {"x": 10, "y": 20, "width": 30, "height": 30, "translation": "Hello"}
        bubbles.update_bubble(bubble, {"box_cx": 150, "box_cy": 150})
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        mask = rendering.build_inpaint_mask(image, [bubble])
        self.assertEqual(mask[25, 15], 255)
        self.assertEqual(mask[150, 150], 0)

    def test_old_jobs_migrate_without_duplicate_aliases(self):
        bubble = {"w": 40, "h": 30, "orig_w": 40, "orig_h": 30, "_text_angle": 12}
        bubbles.normalize_bubble(bubble)
        self.assertEqual(bubble, {"width": 40, "height": 30, "text_angle": 12})

    def test_moving_resized_box_preserves_scale(self):
        bubble = {"x": 0, "y": 0, "width": 40, "height": 30}
        bubbles.update_bubble(bubble, {"box_sw": 2})
        bubbles.update_bubble(bubble, {"box_cx": 200})
        self.assertEqual(bubble["width"], 80)
        self.assertEqual(bubble["source_box"], [0, 0, 40, 30])
        self.assertEqual(bubble["x"], 160)

    def test_nonfinite_editor_values_are_rejected(self):
        with self.assertRaises(ValueError):
            bubbles.update_bubble({"x": 0, "y": 0, "width": 40, "height": 30}, {"box_cx": "nan"})


class WebTests(unittest.TestCase):
    def test_frontend_assets_are_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/static/app.js", response.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/app.css").status_code, 200)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        for name, relative in (
            ("BASE_DIR", ""),
            ("UPLOADS_DIR", "uploads"),
            ("RESULTS_DIR", "results"),
            ("JOBS_DIR", "jobs"),
        ):
            directory = root / relative
            directory.mkdir(exist_ok=True)
            self.patch(name, directory)
        self.patch("GLOSSARY_FILE", root / "glossary.json")
        self.patch("JOBS", {})
        self.patch("MODEL_GATE", asyncio.Lock())
        self.client = TestClient(web.app)
        image = io.BytesIO()
        Image.new("RGB", (40, 50), "white").save(image, format="PNG")
        self.png = image.getvalue()

    def patch(self, name, value):
        patcher = patch.object(web, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def upload(self, files=None):
        response = self.client.post(
            "/api/upload",
            files=files or [("files", ("page.png", self.png, "image/png"))],
            data={"llm_model": "test-model"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["job_id"]

    def test_upload_sanitizes_paths_and_prevents_duplicate_output_stems(self):
        job_id = self.upload(
            [
                ("files", ("../page.png", self.png, "image/png")),
                ("files", ("page.jpg", self.png, "image/jpeg")),
            ]
        )
        names = sorted(path.name for path in (web.UPLOADS_DIR / job_id).iterdir())
        self.assertEqual(names, ["page.png", "page_2.jpg"])
        self.assertFalse((web.UPLOADS_DIR / "page.png").exists())

    def test_archive_duplicate_names_are_preserved(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("chapter1/page.png", self.png)
            archive.writestr("chapter2/page.png", self.png)
        job_id = self.upload([("files", ("pages.zip", buffer.getvalue(), "application/zip"))])
        self.assertEqual(len(list((web.UPLOADS_DIR / job_id).iterdir())), 2)

    def test_rerender_returns_canonical_server_data_and_restores_job_settings(self):
        job_id = self.upload()
        web._save_pages(
            job_id,
            [
                {
                    "page": 1,
                    "filename": "page.png",
                    "bubbles": [
                        {"idx": 1, "x": 1, "y": 1, "w": 20, "h": 30, "translation": "hello"}
                    ],
                }
            ],
        )

        def draw(image, bubble_list, **kwargs):
            self.assertEqual(settings().llm_model, "test-model")
            self.assertEqual(bubble_list[0]["width"], 20)
            return image

        with patch.object(web, "draw_results", side_effect=draw):
            response = self.client.post(
                f"/api/job/{job_id}/page/1/render",
                json={"bubbles": [{"idx": 1, "text_color": "#ff0080", "box_cx": 20}]},
            )
        self.assertEqual(response.status_code, 200, response.text)
        bubble = response.json()["bubbles"][0]
        self.assertEqual(bubble["text_color"], [255, 0, 128])
        self.assertNotIn("w", bubble)
        self.assertEqual(web._load_pages(job_id)[0]["bubbles"][0], bubble)

    def test_worker_failures_reach_websocket(self):
        job_id = self.upload()
        with patch.object(web.mt, "process_directory", side_effect=ValueError("test failure")):
            with self.client.websocket_connect(f"/ws/{job_id}") as socket:
                socket.send_json({"action": "start"})
                event = socket.receive_json()
        self.assertEqual(event, {"type": "error", "message": "test failure"})
        self.assertEqual(web.JOBS[job_id]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
