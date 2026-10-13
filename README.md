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
.venv/bin/python run.py --replay records/red_to_cyan.json --out out/replayed.mp4   # キー・SDK 不要
```

流れ: 腕を横へ退避 → 俯瞰画像を VLM（`gemini-robotics-er-2-preview`）に渡し pick / place の 2 点を取得 → 高さが既知の水平面（pick はキューブ上面、place は置き場の表面）との交点で世界座標へ変換 → ホームへ復帰 → `pick_and_place`。`--out` と同名の `.json` に指示文・VLM の生応答・2 点・結果を保存し、`--replay` はその指示文と生応答だけで同じ実行をやり直す。

`records/` に実 API の応答 4 本（`red_to_cyan` `blue_to_yellow` `grass_to_banana` `stop_to_darkest`）を同梱している。

複数の記録を指示文つきの 1 本に連結する（各本の前に指示カード、下に指示文と速度表記の帯。CJK フォントが要る。既定候補が無ければ `--font` で指定）:

```bash
.venv/bin/python scripts/make_video.py out/a.json out/b.json --out out/submission.mp4 [--font <path>]
```

## 学習ポリシーによる把持（手先カメラ）

「VLM の点 → 上空へ移動 → 学習ポリシーで把持と持ち上げ → 置き場へ運搬」を 1 本で実行する経路。ポリシーの入力は手先カメラ画像（96×96）と関節角 6 個のみで、把持から設置まで `env.cube_pos` を読まない（成功判定だけが読む）。

```bash
uv pip install -e ".[learned]"                                                # torch
.venv/bin/python scripts/collect_grasp_demos.py --episodes 300 --seed 0       # 約 2 分（out/learned/demos.npz）
.venv/bin/python scripts/train_grasp_bc.py --epochs 30                        # 約 100 秒（MPS）、out/learned/policy.pt
.venv/bin/python scripts/eval_grasp_policy.py --episodes 50 --seed 10000      # 約 15 秒、成功数/試行数
.venv/bin/python run.py --replay records/red_to_cyan.json --policy out/learned/policy.pt --out out/learned_red_to_cyan.mp4
```

学習済みの重みを `checkpoints/grasp_policy.pt` に同梱している。収集と学習を省いて再生だけするなら次の 1 行（API キー不要）。

```bash
.venv/bin/python run.py --replay records/red_to_cyan.json --policy checkpoints/grasp_policy.pt --out out/learned_red_to_cyan.mp4
```

`--policy` なしの実行は従来のスクリプト把持のまま。`train_grasp_bc.py --no-image` は関節角だけの比較用ポリシーを学習する。

## 検証

```bash
.venv/bin/python scripts/check_scene.py   # 2 組を同一エピソードで実行し out/ にフレームを書き出す
.venv/bin/python -m pytest -q
```

## 注意

垂直把持が成立するのは x 0.12〜0.30 m、y ±0.12 m の範囲。
