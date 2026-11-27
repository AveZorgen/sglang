"""Compare eager BF16 prefill and decode with and without paging on the same eight NPUs."""

import os
import unittest

import requests
import torch

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.device_count() >= 8, "requires eight NPUs"
)
class TestPagedExpertsTP8(unittest.TestCase):
    def score(self, extra_args):
        process = popen_launch_server(
            os.environ.get("QWEN3_30B_A3B_MODEL_PATH", "Qwen/Qwen3-30B-A3B"),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="npu",
            env={"SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1"},
            other_args=[
                "--device",
                "npu",
                "--tp-size",
                "8",
                "--ep-size",
                "1",
                "--dtype",
                "bfloat16",
                "--moe-a2a-backend",
                "none",
                "--disable-decode-cuda-graph",
                "--disable-prefill-cuda-graph",
                "--disable-shared-experts-fusion",
                "--disable-radix-cache",
                "--context-length",
                "512",
                "--max-total-tokens",
                "1024",
                *extra_args,
            ],
        )
        try:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "Berlin is the capital of Germany. Rome is the capital of Italy. Paris is the capital of",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 2,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json()["meta_info"]
        finally:
            terminate_and_kill_process_tree(process)

    def test_matches_unpaged(self):
        reference = self.score([])
        # K = top_k so a single decode token fits, and prefill exercises evictions.
        paged = self.score(
            ["--enable-paged-experts", "--paged-experts-num-resident", "8"]
        )
        for key in ("input_token_logprobs", "output_token_logprobs"):
            expected, actual = reference[key], paged[key]
            if key == "input_token_logprobs":
                expected, actual = expected[1:], actual[1:]  # first token has no score
            self.assertGreaterEqual(len(expected), 2)
            self.assertEqual([t[1] for t in expected], [t[1] for t in actual])
            torch.testing.assert_close(
                torch.tensor([t[0] for t in actual]),
                torch.tensor([t[0] for t in expected]),
                rtol=0,
                atol=0.25,
            )


if __name__ == "__main__":
    unittest.main()
