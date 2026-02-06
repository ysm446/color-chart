"""モデル管理モジュール - Qwen3-VLモデルのダウンロード・管理"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import snapshot_download

AVAILABLE_MODELS = {
    "Qwen3-VL-4B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "size_gb": 8,
        "vram_gb": 10,
        "description": "軽量・高速モデル",
        "speed": "高速",
        "accuracy": "良好",
        "recommended_use": "プロトタイプ/リアルタイム",
    },
    "Qwen3-VL-8B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "size_gb": 16,
        "vram_gb": 16,
        "description": "高精度モデル（推奨）",
        "speed": "中速",
        "accuracy": "優秀",
        "recommended_use": "本番使用/詳細分析",
    },
}

DEFAULT_CACHE_DIR = "./models"
CONFIG_FILE = "config.json"


class ModelManager:
    """Qwen3-VLモデルのダウンロード・管理を行うクラス"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("COLOR_CHART_MODEL_DIR")
            or self._load_config_cache_dir()
            or DEFAULT_CACHE_DIR
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()

    def _load_config_cache_dir(self) -> Optional[str]:
        """config.jsonからcache_dirを読み込む"""
        config_path = Path(CONFIG_FILE)
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                return config.get("model_cache_dir")
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _load_metadata(self) -> dict:
        """モデルメタデータを読み込む"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_metadata(self):
        """モデルメタデータを保存する"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    def _update_last_used(self, model_name: str):
        """モデルの最終使用日時を更新する"""
        self._metadata.setdefault(model_name, {})
        self._metadata[model_name]["last_used"] = datetime.now().isoformat()
        self._save_metadata()

    def get_available_models(self) -> list[dict]:
        """利用可能なモデル一覧を取得"""
        models = []
        for name, info in AVAILABLE_MODELS.items():
            downloaded = self.is_downloaded(name)
            models.append({
                "name": name,
                "repo_id": info["repo_id"],
                "size_gb": info["size_gb"],
                "vram_gb": info["vram_gb"],
                "description": info["description"],
                "speed": info["speed"],
                "accuracy": info["accuracy"],
                "recommended_use": info["recommended_use"],
                "downloaded": downloaded,
            })
        return models

    def is_downloaded(self, model_name: str) -> bool:
        """モデルがダウンロード済みかチェック"""
        model_dir = self.cache_dir / model_name
        if not model_dir.exists():
            return False
        # config.jsonの存在で判定（Transformersモデルの必須ファイル）
        return (model_dir / "config.json").exists()

    def get_model_path(self, model_name: str) -> Optional[str]:
        """モデルのローカルパスを取得"""
        if not self.is_downloaded(model_name):
            return None
        self._update_last_used(model_name)
        return str(self.cache_dir / model_name)

    def list_downloaded_models(self) -> list[dict]:
        """ダウンロード済みモデル一覧を取得"""
        models = []
        for name in AVAILABLE_MODELS:
            if not self.is_downloaded(name):
                continue
            model_dir = self.cache_dir / name
            size_bytes = sum(
                f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
            )
            size_gb = size_bytes / (1024 ** 3)
            last_used = self._metadata.get(name, {}).get("last_used", "未使用")
            models.append({
                "name": name,
                "size_gb": size_gb,
                "last_used": last_used,
                "path": str(model_dir),
            })
        return models

    def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """モデルをダウンロードする

        Args:
            model_name: AVAILABLE_MODELSのキー名またはrepo_id
            progress_callback: 進捗コールバック(percent, speed_mbps, eta_seconds)

        Returns:
            ダウンロード先のパス
        """
        # repo_idで指定された場合、model_nameに変換
        resolved_name = model_name
        repo_id = model_name
        for name, info in AVAILABLE_MODELS.items():
            if model_name in (name, info["repo_id"]):
                resolved_name = name
                repo_id = info["repo_id"]
                break
        else:
            raise ValueError(
                f"不明なモデル: {model_name}。"
                f"利用可能: {list(AVAILABLE_MODELS.keys())}"
            )

        local_dir = self.cache_dir / resolved_name

        if progress_callback:
            progress_callback(0, 0, 0)

        start_time = time.time()

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                resume_download=True,
            )
        except Exception as e:
            raise RuntimeError(f"モデルのダウンロードに失敗しました: {e}") from e

        elapsed = time.time() - start_time
        if progress_callback:
            progress_callback(100, 0, 0)

        # メタデータ更新
        self._metadata[resolved_name] = {
            "downloaded_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "download_time_seconds": round(elapsed, 1),
        }
        self._save_metadata()

        return str(local_dir)

    def delete_model(self, model_name: str) -> bool:
        """モデルを削除する"""
        model_dir = self.cache_dir / model_name
        if not model_dir.exists():
            return False
        shutil.rmtree(model_dir)
        self._metadata.pop(model_name, None)
        self._save_metadata()
        return True

    def verify_model(self, model_name: str) -> dict:
        """モデルの整合性をチェックする"""
        model_dir = self.cache_dir / model_name
        result = {"name": model_name, "valid": False, "issues": []}

        if not model_dir.exists():
            result["issues"].append("モデルディレクトリが存在しません")
            return result

        required_files = ["config.json"]
        for fname in required_files:
            if not (model_dir / fname).exists():
                result["issues"].append(f"必須ファイルが見つかりません: {fname}")

        # safetensorsまたはpytorch_model.binの存在チェック
        has_weights = (
            list(model_dir.glob("*.safetensors"))
            or list(model_dir.glob("pytorch_model*.bin"))
        )
        if not has_weights:
            result["issues"].append("モデルの重みファイルが見つかりません")

        result["valid"] = len(result["issues"]) == 0
        return result

    def get_cache_info(self) -> dict:
        """キャッシュ（ストレージ）情報を取得"""
        total_size = 0
        model_sizes = {}
        for name in AVAILABLE_MODELS:
            model_dir = self.cache_dir / name
            if model_dir.exists():
                size = sum(
                    f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                )
                model_sizes[name] = size / (1024 ** 3)
                total_size += size

        disk_usage = shutil.disk_usage(self.cache_dir)
        return {
            "used_gb": total_size / (1024 ** 3),
            "total_gb": disk_usage.total / (1024 ** 3),
            "free_gb": disk_usage.free / (1024 ** 3),
            "model_sizes": model_sizes,
        }

    def clean_cache(self):
        """不完全なダウンロードなどのキャッシュをクリーンアップ"""
        if not self.cache_dir.exists():
            return
        for item in self.cache_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name in AVAILABLE_MODELS and not (item / "config.json").exists():
                shutil.rmtree(item)
