import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import requests

import llm
import manga_translator as mt
from settings import Settings, use_settings


class ChapterTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / "input"
        self.source.mkdir()
        for name in ("10.png", "2.png"):
            (self.source / name).touch()
        self.events = []
        self.render_page = mt._render_page
        self.cancel = threading.Event()
        self.stack.enter_context(patch.object(mt, "release_gpu_memory"))
        self.stack.enter_context(
            patch.object(mt, "CharacterArchive", return_value=Mock(characters={}))
        )
        self.stack.enter_context(
            patch.object(
                mt, "unload_model", side_effect=lambda model: self.events.append("unload_initial")
            )
        )
        self.stack.enter_context(
            patch.object(
                llm, "release_gpu_memory", side_effect=lambda: self.events.append("release_small")
            )
        )
        self.unload = self.stack.enter_context(
            patch.object(
                llm, "unload_model", side_effect=lambda model: self.events.append("unload_large")
            )
        )
        self.prepare = self.stack.enter_context(
            patch.object(mt, "_prepare_page", side_effect=self._prepare)
        )
        self.translate = self.stack.enter_context(
            patch.object(mt, "_translate_page", side_effect=self._translate)
        )
        self.render = self.stack.enter_context(
            patch.object(mt, "_render_page", side_effect=self._render)
        )
        self.done = Mock()

    def _prepare(self, image_path, page_idx, errors, on_stage):
        self.events.append(f"prepare_{page_idx}")
        return [{"text": Path(image_path).name}]

    def _translate(self, image_path, page_idx, bubbles, manga_ctx, *args):
        self.events.append(f"translate_{page_idx}")
        self.assertEqual(bubbles[0]["text"], Path(image_path).name)
        bubbles[0]["translation"] = f"translated {page_idx}"
        return False

    def _render(self, image_path, page_idx, bubbles, output_path, debug, unchanged, on_stage):
        self.events.append(f"render_{page_idx}")
        self.assertIn("unload_large", self.events)
        if not unchanged:
            self.assertEqual(bubbles[0]["translation"], f"translated {page_idx}")

    def _run(self):
        return mt.process_directory(
            input_dir=str(self.source),
            output_dir=str(self.root / "output"),
            error_log_path=str(self.root / "errors.log"),
            llm_model="qwen",
            on_page_done=self.done,
            cancel_event=self.cancel,
        )

    def test_all_pages_finish_each_phase_before_the_next_phase(self):
        result = self._run()
        self.assertEqual(
            self.events,
            [
                "unload_initial",
                "prepare_1",
                "prepare_2",
                "release_small",
                "translate_1",
                "translate_2",
                "unload_large",
                "render_1",
                "render_2",
            ],
        )
        self.assertEqual(result["processed"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertEqual([call.args[2] for call in self.done.call_args_list], ["2.png", "10.png"])
        self.assertIs(
            self.translate.call_args_list[0].args[3], self.translate.call_args_list[1].args[3]
        )

    def test_cancellation_during_translation_unloads_model_and_skips_rendering(self):
        def translate(*args):
            self.cancel.set()
            return False

        self.translate.side_effect = translate
        result = self._run()
        self.translate.assert_called_once()
        self.unload.assert_called_once_with("qwen")
        self.render.assert_not_called()
        self.done.assert_not_called()
        self.assertEqual(result["processed"], 0)

    def test_cancellation_during_preparation_never_starts_large_model(self):
        def prepare(*args):
            self.cancel.set()
            return []

        self.prepare.side_effect = prepare
        self._run()
        self.translate.assert_not_called()
        self.unload.assert_not_called()
        self.render.assert_not_called()

    def test_failed_translation_preserves_source_and_other_pages_continue(self):
        def translate(*args):
            if args[1] == 1:
                raise requests.ReadTimeout("timeout")
            return self._translate(*args)

        self.translate.side_effect = translate
        result = self._run()
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["processed"], 1)
        self.assertTrue(self.render.call_args_list[0].args[5])
        self.assertEqual(self.done.call_args_list[0].args[4], [])

    def test_unload_failure_prevents_inpainting_with_large_model_resident(self):
        self.unload.side_effect = requests.ConnectionError("unload failed")
        with self.assertRaises(requests.ConnectionError):
            self._run()
        self.render.assert_not_called()

    def test_preparation_failure_is_not_sent_to_large_model(self):
        def prepare(*args):
            if args[1] == 1:
                raise ValueError("invalid image")
            return self._prepare(*args)

        self.prepare.side_effect = prepare
        result = self._run()
        self.translate.assert_called_once()
        self.assertEqual(self.translate.call_args.args[1], 2)
        self.assertEqual(result["failed"], 1)
        self.assertTrue(self.render.call_args_list[0].args[5])

    def test_cancellation_during_rendering_keeps_completed_page(self):
        def render(*args):
            self._render(*args)
            self.cancel.set()

        self.render.side_effect = render
        result = self._run()
        self.render.assert_called_once()
        self.done.assert_called_once()
        self.assertEqual(result["processed"], 1)

    def test_preserving_source_does_not_load_inpainting_models(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with (
            patch.object(mt, "read_image", return_value=image),
            patch.object(mt, "write_image") as write,
            patch.object(mt, "draw_results") as draw,
        ):
            self.render_page("input.png", 1, [], "output.png", False, True, None)
        draw.assert_not_called()
        self.assertIs(write.call_args.args[1], image)


class ModelPhaseTests(unittest.TestCase):
    def test_unload_does_not_send_a_generation_prompt(self):
        response = Mock()
        response.json.return_value = {"done": True, "done_reason": "unload"}
        with patch.object(llm.requests, "post", return_value=response) as post:
            llm.unload_model("qwen")
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "model": "qwen",
                "keep_alive": 0,
                "stream": False,
            },
        )
        response.close.assert_called_once()

    def test_multiple_requests_release_small_models_once_and_unload_at_end(self):
        response = Mock()
        response.json.return_value = {"response": "ok", "done_reason": "stop"}
        with (
            use_settings(Settings(llm_model="qwen")),
            patch.object(llm, "release_gpu_memory") as release,
            patch.object(llm, "unload_model") as unload,
            patch.object(llm.requests, "post", return_value=response) as post,
        ):
            with llm.model_phase():
                llm.ollama("qwen", "first")
                llm.ollama("qwen", "second")
                unload.assert_not_called()
            release.assert_called_once()
            unload.assert_called_once_with("qwen")
            self.assertTrue(
                all(call.kwargs["json"]["keep_alive"] == -1 for call in post.call_args_list)
            )
            self.assertFalse(llm._PHASE_ACTIVE.get())

    def test_phase_exception_still_unloads_and_resets_phase(self):
        with (
            use_settings(Settings(llm_model="qwen")),
            patch.object(llm, "release_gpu_memory"),
            patch.object(llm, "unload_model") as unload,
        ):
            with self.assertRaises(ValueError), llm.model_phase():
                raise ValueError("bad response")
            unload.assert_called_once_with("qwen")
            self.assertFalse(llm._PHASE_ACTIVE.get())
