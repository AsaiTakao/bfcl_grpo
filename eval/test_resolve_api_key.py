# -*- coding: utf-8 -*-
"""
_resolve_api_key のユニットテスト(TDD)。

なぜ: GEMINI_API_KEY を HF_TOKEN と同じく「ファイル渡し(Docker/BuildKit secret)」で
渡せるようにする。優先順位(ファイル > 環境変数 > リテラル)と、
改行除去・欠損時フォールバックの挙動を固定する。

実行: python -m unittest eval.test_resolve_api_key  (依存: pyyaml のみ)
"""
from __future__ import annotations

import os
import tempfile
import unittest

from run_tau_eval import _resolve_api_key


class ResolveApiKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        # テスト間で環境変数が漏れないよう、関連キーを退避・除去する。
        self._saved = {
            k: os.environ.pop(k, None)
            for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_FILE")
        }

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_env_var_direct(self) -> None:
        """従来どおり環境変数から読める(後方互換)。"""
        os.environ["GEMINI_API_KEY"] = "key-from-env"
        cfg = {"api_key_env": "GEMINI_API_KEY"}
        self.assertEqual(_resolve_api_key(cfg), "key-from-env")

    def test_file_via_file_env(self) -> None:
        """*_FILE 環境変数が指すファイルの中身を読む(改行は除去)。"""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write("key-from-file\n")
            path = f.name
        try:
            os.environ["GEMINI_API_KEY_FILE"] = path
            cfg = {
                "api_key_env": "GEMINI_API_KEY",
                "api_key_file_env": "GEMINI_API_KEY_FILE",
            }
            self.assertEqual(_resolve_api_key(cfg), "key-from-file")
        finally:
            os.unlink(path)

    def test_file_via_default_path(self) -> None:
        """api_key_file の既定パス(HF_TOKEN_FILE 相当)から読む。"""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write("key-default-path")
            path = f.name
        try:
            cfg = {"api_key_env": "GEMINI_API_KEY", "api_key_file": path}
            self.assertEqual(_resolve_api_key(cfg), "key-default-path")
        finally:
            os.unlink(path)

    def test_file_takes_priority_over_env(self) -> None:
        """ファイルが読めれば環境変数より優先される。"""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write("key-from-file")
            path = f.name
        try:
            os.environ["GEMINI_API_KEY"] = "key-from-env"
            os.environ["GEMINI_API_KEY_FILE"] = path
            cfg = {
                "api_key_env": "GEMINI_API_KEY",
                "api_key_file_env": "GEMINI_API_KEY_FILE",
            }
            self.assertEqual(_resolve_api_key(cfg), "key-from-file")
        finally:
            os.unlink(path)

    def test_falls_back_to_env_when_file_missing(self) -> None:
        """ファイルパスが存在しない場合は環境変数へフォールバックする。"""
        os.environ["GEMINI_API_KEY"] = "key-from-env"
        os.environ["GEMINI_API_KEY_FILE"] = "/nonexistent/path/does/not/exist"
        cfg = {
            "api_key_env": "GEMINI_API_KEY",
            "api_key_file_env": "GEMINI_API_KEY_FILE",
        }
        self.assertEqual(_resolve_api_key(cfg), "key-from-env")

    def test_literal_fallback(self) -> None:
        """ファイルも環境変数指定も無ければリテラル(既定 EMPTY)。"""
        self.assertEqual(_resolve_api_key({}), "EMPTY")
        self.assertEqual(_resolve_api_key({"api_key": "lit"}), "lit")


if __name__ == "__main__":
    unittest.main()
