# pai-final-so101

SO-101 アームによる言語指示つき pick-and-place の最終課題用リポジトリ。
シーンと環境、VLM プランナ（`pai_final.planner`）、実行スクリプト `run.py` を含む。

- シーン: 赤・緑・青の 4 cm キューブ 3 個、色つきの置き場 3 個（黄・シアン・黒）、俯瞰カメラ `top` と正面カメラ `front`
- 環境: `pai_final.env.PickPlaceSceneEnv`（`SO101GymEnv` 継承）
- 実行: `pai_final.skill.pick_and_place(env, cube, place_xy)`
- 座標変換: `pai_final.projection`（VLM の正規化 `[y, x]`（0〜1000）と、高さ z の水平面上の世界座標 xy を相互変換）

ロボットモデルと把持スキルは上流パッケージ `lerobot-env-so101`（`lerobot_env_so101>=0.4.0`）に依存する。シーン XML は実行時に一時ディレクトリへ展開し、上流のロボット XML とメッシュをシンボリックリンクして読み込む。

## セットアップ

```bash
uv venv
uv pip install -e ".[test]"
```

## 実行

```bash
export GOOGLE_API_KEY=...   # 値はリポジトリに置かない
.venv/bin/python run.py --instruction "put the red cube on the cyan square" --out out/red_to_cyan.mp4
.venv/bin/python run.py --replay out/red_to_cyan.json --out out/replayed.mp4   # キー・SDK 不要
```

流れ: 腕を横へ退避 → 俯瞰画像を VLM（`gemini-robotics-er-2-preview`）に渡し pick / place の 2 点を取得 → 高さが既知の水平面（pick はキューブ上面、place は置き場の表面）との交点で世界座標へ変換 → ホームへ復帰 → `pick_and_place`。`--out` と同名の `.json` に指示文・VLM の生応答・2 点・結果を保存し、`--replay` はその指示文と生応答だけで同じ実行をやり直す。

## 検証

```bash
.venv/bin/python scripts/check_scene.py   # 2 組を同一エピソードで実行し out/ にフレームを書き出す
.venv/bin/python -m pytest -q
```

## 注意

垂直把持が成立するのは x 0.12〜0.30 m、y ±0.12 m の範囲。
