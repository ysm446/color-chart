# Color Chart

美術理論に基づいた画像分析ツール。ローカルVision LLM（Qwen3-VL）を使用して、画像のカラースキーム、コンポジション、彩度分布などを専門的に分析します。

## 機能

### カラー分析
- 支配的な色の抽出（K-means clustering）
- カラースキーム分類
  - Complementary（補色）
  - Split-Complementary（分裂補色）
  - Analogous（類似色）
  - Triadic（三角配色）
  - Monochromatic（単色）
- 色温度分析（暖色vs寒色の比率）
- RGB/HEX値の表示と割合

### コンポジション分析
- フォーカルポイント検出（相対座標）
- ネガティブスペースの評価
- 視線の流れ分析
- 構図タイプの判定（三分割法、黄金比など）
- バランス評価（対称/非対称）
- 強みと改善提案

### ビジュアライゼーション
- 三分割法グリッドオーバーレイ
- フォーカルポイント表示
- 彩度ヒートマップ
- 複合表示モード

### モデル管理
- ダウンロード可能なモデル一覧表示
- モデルのダウンロード進捗表示
- ローカルにダウンロード済みモデルの管理
- モデルの削除機能
- ストレージ使用量の表示

## 技術スタック

- **UI**: Gradio（ドラッグ&ドロップ対応、タブベースUI）
- **Vision LLM**: Qwen3-VL-4B/8B (Transformers経由)
- **画像処理**: Pillow, OpenCV
- **クラスタリング**: scikit-learn (K-means)
- **Python**: 3.10+

## セットアップ

### 必要要件

- Python 3.10以上
- VRAM 8GB以上（4Bモデル）/ 16GB以上（8Bモデル推奨）
- CUDA対応GPU（推奨、CPUでも動作可能だが非常に遅い）
- ストレージ: 20GB以上（モデル保存用）

### インストール手順

1. **リポジトリのクローン**
```bash
git clone <repository-url>
cd color-chart
```

2. **Pythonパッケージのインストール**
```bash
# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 基本パッケージ
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1版
# または CPU版: pip install torch torchvision

# 必要なライブラリ
pip install -r requirements.txt
```

3. **アプリケーションの起動**
```bash
python color_chart.py
```

初回起動時は自動的にモデル管理ページが表示されます。

4. **モデルのダウンロード**
- ブラウザで `http://localhost:7860` にアクセス
- 「🔧 モデル管理」タブを開く
- ダウンロード可能なモデルリストから選択
- 「ダウンロード」ボタンをクリック
- ダウンロード完了後、「🎨 分析」タブで使用可能

## 使い方

### モデル管理
1. **初回セットアップ**: 「🔧 モデル管理」タブでモデルをダウンロード
2. **モデル確認**: ダウンロード済みモデルリストで状態確認
3. **モデル削除**: 不要なモデルを削除してストレージを節約

### 画像分析
1. **画像のアップロード**: ドラッグ&ドロップまたはクリックして画像を選択
2. **モデル選択**: ダウンロード済みモデルから選択
3. **分析項目の選択**: カラー分析、コンポジション分析などをチェック
4. **分析実行**: 「🔍 分析開始」ボタンをクリック
5. **オーバーレイ表示**: 必要に応じてグリッドやフォーカルポイントを表示
6. **結果の確認**: カラースウォッチとコンポジション詳細を確認

## プロジェクト構成
```
color-chart/
├── color_chart.py          # メインアプリケーション
├── model_manager.py        # モデル管理モジュール
├── CLAUDE.md               # このファイル（プロジェクト仕様書）
├── requirements.txt        # Python依存関係
├── .gitignore              # Git除外設定
├── config.json             # アプリケーション設定（自動生成）
└── models/                 # ローカルモデル保存ディレクトリ
    ├── Qwen3-VL-4B-Instruct/
    └── Qwen3-VL-8B-Instruct/
```

## モデル管理システム

### 対応モデル一覧

アプリケーションは以下のモデルに対応しています：

| モデル名 | サイズ | VRAM | 速度 | 精度 | 推奨用途 |
|---------|--------|------|------|------|----------|
| Qwen3-VL-4B-Instruct | ~8GB | 8-10GB | 高速 | 良好 | プロトタイプ/リアルタイム |
| Qwen3-VL-8B-Instruct | ~16GB | 14-18GB | 中速 | 優秀 | 本番使用/詳細分析 |

### モデル管理ページの機能

1. **ダウンロード可能モデル**
   - Hugging Faceで公開されているモデルのリスト
   - モデルサイズ、VRAM要件の表示
   - ダウンロードボタン

2. **ダウンロード済みモデル**
   - ローカルに保存されているモデルのリスト
   - ディスク使用量の表示
   - 最終使用日時
   - 削除ボタン

3. **ダウンロード進捗**
   - リアルタイムの進捗バー
   - ダウンロード速度
   - 推定残り時間

4. **ストレージ情報**
   - 合計使用量
   - 利用可能な空き容量
   - モデルごとの詳細

### モデルの保存場所

デフォルト: `./models/`

カスタマイズする場合は `config.json` を編集:
```json
{
  "model_cache_dir": "./models/",
  "max_cache_size_gb": 100
}
```

または環境変数で設定:
```bash
export COLOR_CHART_MODEL_DIR=/path/to/models
```

## カスタマイズ

### 新しいモデルの追加

`model_manager.py`の`AVAILABLE_MODELS`に追加:
```python
AVAILABLE_MODELS = {
    "Qwen3-VL-4B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "size_gb": 8,
        "vram_gb": 10,
        "description": "軽量・高速モデル"
    },
    "Qwen3-VL-8B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "size_gb": 16,
        "vram_gb": 16,
        "description": "高精度モデル（推奨）"
    },
    # カスタムモデルを追加
    "Custom-Model": {
        "repo_id": "username/custom-model",
        "size_gb": 12,
        "vram_gb": 14,
        "description": "カスタムモデル"
    }
}
```

### モデルダウンロードのカスタマイズ
```python
from model_manager import ModelManager

manager = ModelManager(cache_dir="./my_models")

# モデルダウンロード（プログラム的に）
manager.download_model(
    "Qwen/Qwen3-VL-8B-Instruct",
    progress_callback=lambda p: print(f"Progress: {p}%")
)

# ダウンロード済みモデルの確認
models = manager.list_downloaded_models()
for model in models:
    print(f"{model['name']}: {model['size_gb']:.2f} GB")
```

### 量子化による軽量化

モデルダウンロード時に量子化オプションを指定可能（将来実装予定）:
```python
# 4-bit量子化でダウンロード（VRAM使用量を50%削減）
manager.download_model(
    "Qwen/Qwen3-VL-8B-Instruct",
    quantization="4bit"
)
```

## トラブルシューティング

### モデルダウンロードが失敗する

1. **ネットワーク接続を確認**
```bash
ping huggingface.co
```

2. **Hugging Face トークンの設定**（プライベートモデルの場合）
```bash
huggingface-cli login
```

または環境変数:
```bash
export HF_TOKEN=your_token_here
```

3. **再試行**: モデル管理ページの「再ダウンロード」ボタン

### ディスク容量不足

1. **不要なモデルを削除**: モデル管理ページから削除
2. **キャッシュをクリーンアップ**:
```bash
python -c "from model_manager import ModelManager; ModelManager().clean_cache()"
```

### モデルが読み込めない

1. **整合性チェック**:
```python
from model_manager import ModelManager
manager = ModelManager()
manager.verify_model("Qwen3-VL-8B-Instruct")
```

2. **モデルの再ダウンロード**: 破損している場合は削除して再ダウンロード

### PyTorchのインストールエラー
```bash
# CUDA版（GPU使用）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU版（GPUなし）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDAが認識されない
```python
import torch
print(torch.cuda.is_available())  # Trueになるか確認
print(torch.cuda.get_device_name(0))  # GPU名を確認
```

### メモリ不足エラー（CUDA Out of Memory）
1. **4Bモデルに切り替え**
2. **量子化を使用**（将来実装）
3. **バッチサイズを削減**

### Transformersのバージョンエラー
```bash
pip install transformers>=4.37.0 --upgrade
```

## パフォーマンス最適化

### GPU使用時（推奨設定）
```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
```

### メモリ効率重視
```python
from transformers import BitsAndBytesConfig

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto"
)
```

## 開発ロードマップ

### Phase 1: コア機能（v1.0）
- [x] カラー分析機能
- [x] コンポジション分析機能
- [x] オーバーレイ表示
- [x] モデル管理システム

### Phase 2: 拡張機能（v1.1）
- [ ] バッチ処理機能（複数画像の一括分析）
- [ ] 分析結果のエクスポート（JSON/CSV/PDF）
- [ ] カスタムカラーパレット生成
- [ ] 画像比較モード（2枚の画像を並べて分析）

### Phase 3: データ管理（v1.2）
- [ ] 分析履歴の保存・検索機能（SQLite）
- [ ] タグ機能（Clip Managerスタイル）
- [ ] 分析結果のフィルタリング

### Phase 4: 統合機能（v2.0）
- [ ] RESTful API対応
- [ ] Stable Diffusionプロンプト生成統合
- [ ] Houdini連携（HDAとしてのエクスポート）
- [ ] CLI版の実装

### Phase 5: デプロイメント（v2.1）
- [ ] Docker対応
- [ ] Web API版
- [ ] モバイルアプリ対応

## API使用例

### モデル管理API
```python
from model_manager import ModelManager

# モデルマネージャー初期化
manager = ModelManager(cache_dir="./models")

# 利用可能なモデルを取得
available = manager.get_available_models()
for model in available:
    print(f"{model['name']}: {model['size_gb']} GB")

# モデルダウンロード
def on_progress(percent, speed_mbps, eta_seconds):
    print(f"進捗: {percent}% | 速度: {speed_mbps:.1f} MB/s | 残り: {eta_seconds}秒")

manager.download_model(
    "Qwen/Qwen3-VL-8B-Instruct",
    progress_callback=on_progress
)

# ダウンロード済みモデルを取得
downloaded = manager.list_downloaded_models()
for model in downloaded:
    print(f"{model['name']}: {model['size_gb']:.2f} GB ({model['last_used']})")

# モデル削除
manager.delete_model("Qwen/Qwen3-VL-4B-Instruct")

# キャッシュ情報
cache_info = manager.get_cache_info()
print(f"使用量: {cache_info['used_gb']:.2f} GB / {cache_info['total_gb']:.2f} GB")
```

### カラー分析API
```python
from color_chart import ColorChartAnalyzer
from PIL import Image

# 初期化（ダウンロード済みモデルを指定）
analyzer = ColorChartAnalyzer(model_path="./models/Qwen3-VL-8B-Instruct")

# 画像読み込み
image = Image.open("artwork.jpg")

# カラー分析
colors, percentages = analyzer.extract_colors(image)
scheme, emoji = analyzer.classify_scheme(colors)
warm_pct, cool_pct = analyzer.get_color_temperature(colors, percentages)

print(f"カラースキーム: {scheme} {emoji}")
print(f"色温度: 暖色{warm_pct:.0f}% / 寒色{cool_pct:.0f}%")

# コンポジション分析
composition_data = analyzer.analyze_composition("artwork.jpg")
print(f"フォーカルポイント: {composition_data['focal_point']}")
```

## 環境変数
```bash
# モデル保存ディレクトリ
export COLOR_CHART_MODEL_DIR=/path/to/models

# Hugging Face トークン（プライベートモデル用）
export HF_TOKEN=your_token_here

# Hugging Face キャッシュディレクトリ
export HF_HOME=/path/to/cache

# ログレベル
export TRANSFORMERS_VERBOSITY=error  # warning, info, debug

# 最大キャッシュサイズ（GB）
export COLOR_CHART_MAX_CACHE_GB=100
```

## システム要件

### 最小要件
- CPU: 4コア以上
- RAM: 16GB以上
- GPU: NVIDIA GPU (CUDA対応) VRAM 8GB以上
- ストレージ: 30GB以上（モデル保存用）

### 推奨要件
- CPU: 8コア以上
- RAM: 32GB以上
- GPU: NVIDIA RTX 3090/4090, A100, VRAM 16GB以上
- ストレージ: SSD 50GB以上

## ライセンス

MIT License

## 作成者

Ken - Technical Artist specializing in Houdini and procedural workflows

## 参考資料

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen2-VL)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Gradio Documentation](https://www.gradio.app/docs)
- Color Theory: Josef Albers "Interaction of Color"

## 謝辞

このプロジェクトは以下のオープンソースプロジェクトを使用しています:
- Qwen3-VL by Alibaba Cloud
- Transformers by Hugging Face
- Gradio by Hugging Face
- PyTorch by Meta AI