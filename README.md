# pai-final-so101

SO-101 アームによる言語指示つき pick-and-place の最終課題用リポジトリ。
現時点では、VLM を使わずに世界座標を直接渡して pick-and-place が通るシーンと環境までを含む。

- シーン: 赤・緑・青の 4 cm キューブ 3 個、色つきの置き場 3 個（黄・シアン・黒）、俯瞰カメラ `top` と正面カメラ `front`
- 環境: `pai_final.env.PickPlaceSceneEnv`（`SO101GymEnv` 継承）
- 実行: `pai_final.skill.pick_and_place(env, cube, place_xy)`
- 座標変換: `pai_final.projection`（VLM の正規化 `[y, x]`（0〜1000）と、高さ z の水平面上の世界座標 xy を相互変換）

ロボットモデルと把持スキルは上流パッケージ `lerobot-env-so101`（`lerobot_env_so101>=0.4.0`）に依存する。シーン XML は実行時に一時ディレクトリへ展開し、上流のロボット XML とメッシュをシンボリックリンクして読み込む。

## セットアップ

```bash
uv venv
uv pip install -e <lerobot_env_so101 のソースディレクトリ>   # 0.4.0 が PyPI に出るまで
uv pip install --no-deps -e .
uv pip install pillow "imageio[ffmpeg]" pytest
```

## 検証

```bash
.venv/bin/python scripts/check_scene.py   # 2 組を同一エピソードで実行し out/ にフレームを書き出す
.venv/bin/python -m pytest -q
```

## 注意

垂直把持が成立するのは x 0.12〜0.30 m、y ±0.12 m の範囲。
