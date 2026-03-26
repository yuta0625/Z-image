# Z-Image

Hugging Face の拡散モデル（Tongyi-MAI/Z-Image）を使って、テキストプロンプトからアニメ画像を生成するPythonツール。

## 全体アーキテクチャ

```mermaid
graph TD
    A[ユーザー] -->|テキストプロンプト| B[prompts/prompt.txt]
    C[configs/z_image.yaml] -->|設定読み込み| D[main]
    B -->|プロンプト読み込み| D

    D --> E{パイプライン初期化}
    E -->|prefer_specialized_pipeline: true| F[ZImagePipeline]
    F -->|失敗| G[DiffusionPipeline\ntrust_remote_code=True]
    E -->|フォールバック| G

    F --> H[画像生成]
    G --> H

    H -->|seed / steps / guidance| I[生成画像]
    I -->|PNG保存| J[outputs/sample.png]
```

## 推論フロー

```mermaid
sequenceDiagram
    participant User
    participant main as main()
    participant cfg as load_config()
    participant prompt as load_prompt()
    participant pipe as load_pipeline()
    participant gen as generate_image()
    participant save as save_image()

    User->>main: 実行
    main->>cfg: configs/z_image.yaml 読み込み
    cfg-->>main: config dict
    main->>prompt: prompts/prompt.txt 読み込み
    prompt-->>main: prompt str
    main->>pipe: パイプライン初期化 (model_id, dtype, device)
    pipe-->>main: pipeline object
    main->>gen: 画像生成 (pipe, cfg, prompt)
    gen-->>main: PIL.Image
    main->>save: outputs/sample.png に保存
    save-->>User: 完了
```

## 設定パラメータ

```mermaid
graph LR
    subgraph configs/z_image.yaml
        M[model_id\nTongyi-MAI/Z-Image]
        R[resolution\n1024 x 1024]
        S[seed: 42]
        N[num_inference_steps: 32]
        G[guidance_scale: 4.0]
        D[dtype: bfloat16]
        DV[device: cuda]
    end
```

## ディレクトリ構成

```
z-image/
├── src/
│   └── inference.py       # 推論メインスクリプト
├── configs/
│   └── z_image.yaml       # モデル・生成パラメータ設定
├── prompts/
│   └── prompt.txt         # 入力テキストプロンプト
├── outputs/               # 生成画像の出力先
└── pyproject.toml         # プロジェクト依存関係
```

## セットアップ

```bash
pip install -e .
```

## 実行

```bash
python src/inference.py
```

生成画像は `outputs/sample.png` に保存されます。

## 主要パラメータ一覧

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `model_id` | `Tongyi-MAI/Z-Image` | HuggingFace モデル ID |
| `height` / `width` | `1024` | 出力解像度 (px) |
| `num_inference_steps` | `32` | ノイズ除去ステップ数 (推奨: 28〜50) |
| `guidance_scale` | `4.0` | プロンプト忠実度 (推奨: 3.0〜5.0) |
| `seed` | `42` | 再現性のための乱数シード |
| `device` | `cuda` | 推論デバイス |
| `dtype` | `bfloat16` | 計算精度 (fp16 / bf16 / fp32) |

## 依存ライブラリ

- [diffusers](https://github.com/huggingface/diffusers)
- [transformers](https://github.com/huggingface/transformers)
- [accelerate](https://github.com/huggingface/accelerate)
- [huggingface-hub](https://github.com/huggingface/huggingface_hub)
- [Pillow](https://python-pillow.org/)
- PyTorch (CUDA 対応推奨)
