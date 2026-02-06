"""Color Chart - 美術理論に基づいた画像分析ツール

ローカルVision LLM（Qwen3-VL）を使用して、画像のカラースキーム、
コンポジション、彩度分布などを専門的に分析します。
"""

import colorsys
import json
import re
import threading
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from model_manager import AVAILABLE_MODELS, ModelManager

# ---------------------------------------------------------------------------
# カラー分析
# ---------------------------------------------------------------------------


class ColorAnalyzer:
    """K-meansクラスタリングによるカラー分析"""

    @staticmethod
    def extract_colors(
        image: Image.Image, n_colors: int = 6
    ) -> tuple[list[tuple[int, int, int]], list[float]]:
        """画像から支配的な色を抽出する

        Returns:
            (colors, percentages): RGBタプルのリストと各色の割合リスト
        """
        img = image.copy().convert("RGB")
        # 計算量を抑えるためリサイズ
        img.thumbnail((200, 200))
        pixels = np.array(img).reshape(-1, 3).astype(np.float64)

        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        labels = kmeans.fit_predict(pixels)

        centers = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(labels, minlength=n_colors)
        percentages = counts / counts.sum()

        # 割合の大きい順にソート
        order = np.argsort(-percentages)
        colors = [tuple(centers[i]) for i in order]
        percentages = [float(percentages[i]) for i in order]

        return colors, percentages

    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
        """RGB→HSV (H: 0-360, S: 0-1, V: 0-1)"""
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        return h * 360, s, v

    @staticmethod
    def classify_scheme(
        colors: list[tuple[int, int, int]],
    ) -> tuple[str, str]:
        """カラースキームを分類する

        Returns:
            (scheme_name, emoji)
        """
        if len(colors) < 2:
            return "Monochromatic", "🎨"

        hues = [ColorAnalyzer.rgb_to_hsv(*c)[0] for c in colors]
        sats = [ColorAnalyzer.rgb_to_hsv(*c)[1] for c in colors]

        # 彩度が低い色はスキームの判定に使わない
        chromatic = [(h, s) for h, s in zip(hues, sats) if s > 0.15]
        if len(chromatic) < 2:
            return "Monochromatic", "🎨"

        chroma_hues = [h for h, _ in chromatic]

        # 色相の最大差分を算出
        def hue_diff(h1: float, h2: float) -> float:
            d = abs(h1 - h2)
            return min(d, 360 - d)

        diffs = []
        for i in range(len(chroma_hues)):
            for j in range(i + 1, len(chroma_hues)):
                diffs.append(hue_diff(chroma_hues[i], chroma_hues[j]))

        max_diff = max(diffs)
        avg_diff = sum(diffs) / len(diffs)

        # 色相のスプレッドが狭い → Analogous
        if max_diff < 60:
            return "Analogous", "🌈"

        # 補色に近い対（150-180度）
        has_complement = any(150 <= d <= 180 for d in diffs)
        # 120度付近の対
        has_triad = any(100 <= d <= 140 for d in diffs)

        if has_complement:
            # 分裂補色: 補色ペア + 中間色
            split = any(30 <= d <= 90 for d in diffs)
            if split:
                return "Split-Complementary", "🔀"
            return "Complementary", "🔴🔵"

        if has_triad:
            return "Triadic", "🔺"

        if max_diff < 90:
            return "Analogous", "🌈"

        return "Complementary", "🔴🔵"

    @staticmethod
    def get_color_temperature(
        colors: list[tuple[int, int, int]], percentages: list[float]
    ) -> tuple[float, float]:
        """暖色/寒色の比率を返す (warm_pct, cool_pct)"""
        warm = 0.0
        cool = 0.0
        for (r, g, b), pct in zip(colors, percentages):
            h, s, _ = ColorAnalyzer.rgb_to_hsv(r, g, b)
            if s < 0.1:
                # 無彩色は中立扱い
                warm += pct * 0.5
                cool += pct * 0.5
            elif h <= 60 or h >= 300:
                # 暖色（赤〜黄、マゼンタ寄り）
                warm += pct
            elif 60 < h < 180:
                # 中間〜寒色へのグラデーション
                ratio = (h - 60) / 120  # 0→1
                warm += pct * (1 - ratio)
                cool += pct * ratio
            else:
                # 180-300: 寒色
                cool += pct

        total = warm + cool
        if total == 0:
            return 50.0, 50.0
        return round(warm / total * 100, 1), round(cool / total * 100, 1)


# ---------------------------------------------------------------------------
# コンポジション分析（Vision LLM）
# ---------------------------------------------------------------------------


COMPOSITION_PROMPT = """\
あなたは美術の専門家です。この画像のコンポジション（構図）を分析してください。
以下のJSON形式で回答してください。JSONのみ出力し、他のテキストは含めないでください。

{
  "focal_points": [
    {"x": 0.0-1.0の相対座標, "y": 0.0-1.0の相対座標, "description": "説明"}
  ],
  "composition_type": "三分割法 / 黄金比 / 中央配置 / 対角線 / S字 / L字 / その他",
  "balance": "対称 / 非対称 / ラジアル",
  "negative_space": "多い / 適度 / 少ない",
  "eye_flow": "視線の流れの説明",
  "strengths": ["強みのリスト"],
  "improvements": ["改善提案のリスト"]
}
"""


class CompositionAnalyzer:
    """Vision LLMを使ったコンポジション分析"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = None

    def load_model(self, model_path: str):
        """Qwen3-VLモデルを読み込む"""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if self.model_name == model_path:
            return  # 既にロード済み

        # 既存モデルのメモリ解放
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            self.model = self.model.to(device)

        self.model_name = model_path

    def analyze(self, image: Image.Image) -> dict:
        """画像のコンポジションを分析する"""
        if self.model is None:
            raise RuntimeError("モデルが読み込まれていません。先にload_model()を呼んでください。")

        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": COMPOSITION_PROMPT},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)

        # 入力部分を除いた出力のみデコード
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> dict:
        """LLMのレスポンスからJSONを抽出してパースする"""
        # JSONブロックを探す
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "focal_points": [],
            "composition_type": "解析不能",
            "balance": "不明",
            "negative_space": "不明",
            "eye_flow": response,
            "strengths": [],
            "improvements": [],
            "raw_response": response,
        }


# ---------------------------------------------------------------------------
# ビジュアライゼーション
# ---------------------------------------------------------------------------


class Visualizer:
    """オーバーレイとビジュアライゼーション生成"""

    @staticmethod
    def draw_rule_of_thirds(image: Image.Image) -> Image.Image:
        """三分割法グリッドを描画"""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        w, h = overlay.size

        line_color = (255, 255, 255, 180)
        line_width = 2

        for i in range(1, 3):
            x = w * i // 3
            draw.line([(x, 0), (x, h)], fill=line_color, width=line_width)
        for i in range(1, 3):
            y = h * i // 3
            draw.line([(0, y), (w, y)], fill=line_color, width=line_width)

        return overlay

    @staticmethod
    def draw_focal_points(
        image: Image.Image,
        focal_points: list[dict],
    ) -> Image.Image:
        """フォーカルポイントを描画"""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        w, h = overlay.size

        for i, fp in enumerate(focal_points):
            x = fp.get("x", 0.5) * w
            y = fp.get("y", 0.5) * h
            radius = min(w, h) * 0.03
            color = (255, 50, 50)

            # 円
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline=color,
                width=3,
            )
            # 十字
            cross = radius * 1.5
            draw.line([(x - cross, y), (x + cross, y)], fill=color, width=2)
            draw.line([(x, y - cross), (x, y + cross)], fill=color, width=2)

            # ラベル
            label = fp.get("description", f"Point {i + 1}")
            try:
                font = ImageFont.truetype("arial.ttf", max(12, min(w, h) // 40))
            except OSError:
                font = ImageFont.load_default()
            draw.text(
                (x + radius + 5, y - radius),
                label,
                fill=color,
                font=font,
            )

        return overlay

    @staticmethod
    def create_saturation_heatmap(image: Image.Image) -> Image.Image:
        """彩度ヒートマップを生成"""
        img_np = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        # ヒートマップ（COLORMAP_JET）
        heatmap = cv2.applyColorMap(saturation, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 元画像とブレンド
        blended = cv2.addWeighted(img_np, 0.4, heatmap_rgb, 0.6, 0)
        return Image.fromarray(blended)

    @staticmethod
    def create_color_swatch(
        colors: list[tuple[int, int, int]],
        percentages: list[float],
    ) -> Image.Image:
        """カラースウォッチ画像を生成"""
        swatch_w, swatch_h = 600, 120
        swatch = Image.new("RGB", (swatch_w, swatch_h), (30, 30, 30))
        draw = ImageDraw.Draw(swatch)

        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except OSError:
            font = ImageFont.load_default()

        x = 0
        for color, pct in zip(colors, percentages):
            w = max(int(swatch_w * pct), 1)
            draw.rectangle([x, 0, x + w, swatch_h - 30], fill=color)

            hex_str = ColorAnalyzer.rgb_to_hex(*color)
            pct_str = f"{pct * 100:.1f}%"
            text_x = x + 4
            text_y = swatch_h - 26
            # テキストの輝度に応じて白か黒
            brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            draw.text((text_x, text_y), f"{hex_str} {pct_str}", fill=text_color, font=font)

            x += w

        return swatch


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

# グローバルインスタンス
model_manager = ModelManager()
composition_analyzer = CompositionAnalyzer()


def get_model_choices() -> list[str]:
    """ダウンロード済みモデルのリストを返す"""
    return [m["name"] for m in model_manager.list_downloaded_models()] or []


def run_analysis(
    image: Optional[Image.Image],
    model_name: str,
    do_color: bool,
    do_composition: bool,
) -> tuple:
    """分析を実行する（Gradioコールバック）

    Returns:
        (color_swatch, color_text, composition_text, overlay_image, composition_data_json)
    """
    if image is None:
        return None, "画像をアップロードしてください。", "", None, "{}"

    color_swatch = None
    color_text = ""
    composition_text = ""
    composition_data = {}
    overlay_image = image.copy()

    # --- カラー分析 ---
    if do_color:
        colors, percentages = ColorAnalyzer.extract_colors(image)
        scheme, emoji = ColorAnalyzer.classify_scheme(colors)
        warm, cool = ColorAnalyzer.get_color_temperature(colors, percentages)
        color_swatch = Visualizer.create_color_swatch(colors, percentages)

        lines = [f"## カラースキーム: {scheme} {emoji}\n"]
        lines.append(f"**色温度**: 暖色 {warm}% / 寒色 {cool}%\n")
        lines.append("### 支配色\n")
        lines.append("| # | HEX | RGB | 割合 |")
        lines.append("|---|-----|-----|------|")
        for i, (c, p) in enumerate(zip(colors, percentages), 1):
            hex_val = ColorAnalyzer.rgb_to_hex(*c)
            lines.append(f"| {i} | {hex_val} | ({c[0]}, {c[1]}, {c[2]}) | {p*100:.1f}% |")
        color_text = "\n".join(lines)

    # --- コンポジション分析 ---
    if do_composition:
        if not model_name:
            composition_text = "モデルが選択されていません。モデル管理タブからダウンロードしてください。"
        else:
            model_path = model_manager.get_model_path(model_name)
            if model_path is None:
                composition_text = f"モデル '{model_name}' が見つかりません。"
            else:
                try:
                    composition_text = "モデルを読み込み中..."
                    composition_analyzer.load_model(model_path)
                    composition_data = composition_analyzer.analyze(image)

                    lines = ["## コンポジション分析\n"]
                    lines.append(f"**構図タイプ**: {composition_data.get('composition_type', '不明')}")
                    lines.append(f"**バランス**: {composition_data.get('balance', '不明')}")
                    lines.append(f"**ネガティブスペース**: {composition_data.get('negative_space', '不明')}")
                    lines.append(f"\n**視線の流れ**: {composition_data.get('eye_flow', '不明')}\n")

                    fps = composition_data.get("focal_points", [])
                    if fps:
                        lines.append("### フォーカルポイント\n")
                        for i, fp in enumerate(fps, 1):
                            desc = fp.get("description", "")
                            x = fp.get("x", 0)
                            y = fp.get("y", 0)
                            lines.append(f"{i}. ({x:.2f}, {y:.2f}) - {desc}")

                    strengths = composition_data.get("strengths", [])
                    if strengths:
                        lines.append("\n### 強み\n")
                        for s in strengths:
                            lines.append(f"- {s}")

                    improvements = composition_data.get("improvements", [])
                    if improvements:
                        lines.append("\n### 改善提案\n")
                        for imp in improvements:
                            lines.append(f"- {imp}")

                    composition_text = "\n".join(lines)
                except Exception as e:
                    composition_text = f"コンポジション分析エラー: {e}"

    return (
        color_swatch,
        color_text,
        composition_text,
        overlay_image,
        json.dumps(composition_data, ensure_ascii=False),
    )


def apply_overlay(
    image: Optional[Image.Image],
    composition_data_json: str,
    show_grid: bool,
    show_focal: bool,
    show_heatmap: bool,
) -> Optional[Image.Image]:
    """オーバーレイを適用する"""
    if image is None:
        return None

    result = image.copy()

    if show_heatmap:
        result = Visualizer.create_saturation_heatmap(result)

    if show_grid:
        result = Visualizer.draw_rule_of_thirds(result)

    if show_focal and composition_data_json:
        try:
            data = json.loads(composition_data_json)
            fps = data.get("focal_points", [])
            if fps:
                result = Visualizer.draw_focal_points(result, fps)
        except (json.JSONDecodeError, TypeError):
            pass

    return result


# --- モデル管理タブのコールバック ---


def refresh_available_models():
    """利用可能モデルテーブルを更新"""
    models = model_manager.get_available_models()
    rows = []
    for m in models:
        status = "✅ ダウンロード済み" if m["downloaded"] else "⬇️ 未ダウンロード"
        rows.append([
            m["name"],
            f"{m['size_gb']} GB",
            f"{m['vram_gb']} GB",
            m["speed"],
            m["accuracy"],
            status,
        ])
    return rows


def refresh_downloaded_models():
    """ダウンロード済みモデルテーブルを更新"""
    models = model_manager.list_downloaded_models()
    if not models:
        return [["（なし）", "", ""]]
    rows = []
    for m in models:
        rows.append([m["name"], f"{m['size_gb']:.2f} GB", m["last_used"]])
    return rows


def get_storage_info() -> str:
    """ストレージ情報のテキストを返す"""
    info = model_manager.get_cache_info()
    lines = [
        f"**使用量**: {info['used_gb']:.2f} GB",
        f"**空き容量**: {info['free_gb']:.1f} GB",
        f"**合計**: {info['total_gb']:.1f} GB",
    ]
    if info["model_sizes"]:
        lines.append("\n**モデル別使用量**:")
        for name, size in info["model_sizes"].items():
            lines.append(f"- {name}: {size:.2f} GB")
    return "\n".join(lines)


def download_model_ui(model_name: str, progress=gr.Progress()):
    """モデルダウンロード（UIコールバック）"""
    if not model_name:
        return "モデル名を選択してください。", refresh_available_models(), refresh_downloaded_models(), get_storage_info(), gr.update(choices=get_model_choices())

    if model_manager.is_downloaded(model_name):
        return f"{model_name} は既にダウンロード済みです。", refresh_available_models(), refresh_downloaded_models(), get_storage_info(), gr.update(choices=get_model_choices())

    progress(0, desc=f"{model_name} をダウンロード中...")
    try:
        model_manager.download_model(model_name)
        progress(1.0, desc="完了")
        return (
            f"✅ {model_name} のダウンロードが完了しました。",
            refresh_available_models(),
            refresh_downloaded_models(),
            get_storage_info(),
            gr.update(choices=get_model_choices()),
        )
    except Exception as e:
        return (
            f"❌ ダウンロードエラー: {e}",
            refresh_available_models(),
            refresh_downloaded_models(),
            get_storage_info(),
            gr.update(choices=get_model_choices()),
        )


def delete_model_ui(model_name: str):
    """モデル削除（UIコールバック）"""
    if not model_name:
        return "モデル名を入力してください。", refresh_available_models(), refresh_downloaded_models(), get_storage_info(), gr.update(choices=get_model_choices())

    if model_manager.delete_model(model_name):
        return (
            f"✅ {model_name} を削除しました。",
            refresh_available_models(),
            refresh_downloaded_models(),
            get_storage_info(),
            gr.update(choices=get_model_choices()),
        )
    return (
        f"❌ {model_name} が見つかりません。",
        refresh_available_models(),
        refresh_downloaded_models(),
        get_storage_info(),
        gr.update(choices=get_model_choices()),
    )


# ---------------------------------------------------------------------------
# アプリケーション構築
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Gradio UIを構築して返す"""

    has_models = len(get_model_choices()) > 0

    initial_tab = "analysis" if has_models else "model_management"

    with gr.Blocks(
        title="Color Chart - 画像分析ツール",
    ) as app:
        gr.Markdown("# 🎨 Color Chart\n美術理論に基づいた画像分析ツール")

        # 分析結果の内部状態
        composition_data_state = gr.State("{}")

        with gr.Tabs(selected=initial_tab) as tabs:
            # ===== 分析タブ =====
            with gr.Tab("🎨 分析", id="analysis"):
                with gr.Row():
                    # --- 左カラム: 入力 ---
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="画像をアップロード",
                            type="pil",
                            height=400,
                        )
                        model_dropdown = gr.Dropdown(
                            label="モデル選択",
                            choices=get_model_choices(),
                            value=get_model_choices()[0] if has_models else None,
                            interactive=True,
                        )
                        with gr.Row():
                            do_color = gr.Checkbox(label="カラー分析", value=True)
                            do_composition = gr.Checkbox(label="コンポジション分析", value=True)
                        analyze_btn = gr.Button("🔍 分析開始", variant="primary", size="lg")

                        gr.Markdown("### オーバーレイ表示")
                        with gr.Row():
                            show_grid = gr.Checkbox(label="三分割法", value=False)
                            show_focal = gr.Checkbox(label="フォーカルポイント", value=False)
                            show_heatmap = gr.Checkbox(label="彩度ヒートマップ", value=False)

                    # --- 右カラム: 結果 ---
                    with gr.Column(scale=1):
                        color_swatch = gr.Image(label="カラースウォッチ", height=140)
                        color_result = gr.Markdown(label="カラー分析結果")
                        composition_result = gr.Markdown(label="コンポジション分析結果")
                        overlay_image = gr.Image(label="オーバーレイ表示", height=400)

                # 分析ボタンのイベント
                analyze_btn.click(
                    fn=run_analysis,
                    inputs=[input_image, model_dropdown, do_color, do_composition],
                    outputs=[
                        color_swatch,
                        color_result,
                        composition_result,
                        overlay_image,
                        composition_data_state,
                    ],
                )

                # オーバーレイのチェックボックス変更時
                overlay_inputs = [
                    input_image,
                    composition_data_state,
                    show_grid,
                    show_focal,
                    show_heatmap,
                ]
                for checkbox in [show_grid, show_focal, show_heatmap]:
                    checkbox.change(
                        fn=apply_overlay,
                        inputs=overlay_inputs,
                        outputs=[overlay_image],
                    )

            # ===== モデル管理タブ =====
            with gr.Tab("🔧 モデル管理", id="model_management"):
                gr.Markdown("## モデル管理")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ダウンロード可能なモデル")
                        available_table = gr.Dataframe(
                            headers=["モデル名", "サイズ", "VRAM", "速度", "精度", "状態"],
                            value=refresh_available_models(),
                            interactive=False,
                        )
                        with gr.Row():
                            download_model_name = gr.Dropdown(
                                label="ダウンロードするモデル",
                                choices=list(AVAILABLE_MODELS.keys()),
                            )
                            download_btn = gr.Button("⬇️ ダウンロード", variant="primary")
                        download_status = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("### ダウンロード済みモデル")
                        downloaded_table = gr.Dataframe(
                            headers=["モデル名", "サイズ", "最終使用"],
                            value=refresh_downloaded_models(),
                            interactive=False,
                        )
                        with gr.Row():
                            delete_model_name = gr.Dropdown(
                                label="削除するモデル",
                                choices=get_model_choices(),
                            )
                            delete_btn = gr.Button("🗑️ 削除", variant="stop")
                        delete_status = gr.Markdown("")

                gr.Markdown("### ストレージ情報")
                storage_info = gr.Markdown(value=get_storage_info())

                # ダウンロードボタン
                download_btn.click(
                    fn=download_model_ui,
                    inputs=[download_model_name],
                    outputs=[
                        download_status,
                        available_table,
                        downloaded_table,
                        storage_info,
                        model_dropdown,
                    ],
                )

                # 削除ボタン
                delete_btn.click(
                    fn=delete_model_ui,
                    inputs=[delete_model_name],
                    outputs=[
                        delete_status,
                        available_table,
                        downloaded_table,
                        storage_info,
                        model_dropdown,
                    ],
                )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), share=False)
