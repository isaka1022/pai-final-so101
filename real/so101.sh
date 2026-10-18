#!/bin/bash
# Real SO-101 with a wrist camera: record teleoperated demos, train ACT, run the policy.
# Usage: real/so101.sh <ports|cameras|hand-probe|calibrate-follower|calibrate-leader|teleop|record|train|rollout>
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=.venv-real/bin
# lerobot launches the rerun viewer by name
export PATH="${PWD}/${BIN}:${PATH}"
DATASET_ID=local/so101_wrist_pick
DATASET_ROOT=out/real/dataset
POLICY_DIR=out/real/act
STEP="${1:-}"

case "${STEP}" in
  ports) exec "${BIN}/lerobot-find-port" ;;
  cameras) exec "${BIN}/lerobot-find-cameras" opencv --output-dir out/real/camera_check ;;
  hand-probe) shift; exec "${BIN}/python" real/hand_probe.py "$@" ;;
esac

if [ ! -f real/config.env ]; then
  echo "real/config.env is missing: copy real/config.env.example and fill in the ports and camera index" >&2
  exit 1
fi
. real/config.env

ROBOT=(--robot.type=so101_follower "--robot.port=${FOLLOWER_PORT}" --robot.id=follower)
TELEOP=(--teleop.type=so101_leader "--teleop.port=${LEADER_PORT}" --teleop.id=leader)
CAMERAS="--robot.cameras={wrist: {type: opencv, index_or_path: ${WRIST_CAMERA}, width: 640, height: 480, fps: 30}}"

case "${STEP}" in
  calibrate-follower) exec "${BIN}/lerobot-calibrate" "${ROBOT[@]}" ;;
  calibrate-leader) exec "${BIN}/lerobot-calibrate" "${TELEOP[@]}" ;;
  teleop) exec "${BIN}/lerobot-teleoperate" "${ROBOT[@]}" "${CAMERAS}" "${TELEOP[@]}" "--display_data=${DISPLAY_DATA}" ;;
  record)
    RESUME=false
    if [ -d "${DATASET_ROOT}/data" ]; then
      RESUME=true
    elif [ -d "${DATASET_ROOT}" ]; then
      # a run that failed before the first episode leaves a stub that blocks a fresh start
      mv "${DATASET_ROOT}" "${DATASET_ROOT}.empty.$(date +%s)"
    fi
    exec "${BIN}/lerobot-record" "${ROBOT[@]}" "${CAMERAS}" "${TELEOP[@]}" \
      "--dataset.repo_id=${DATASET_ID}" "--dataset.root=${DATASET_ROOT}" --dataset.push_to_hub=false \
      "--dataset.single_task=${TASK}" "--dataset.num_episodes=${EPISODES}" \
      "--dataset.episode_time_s=${EPISODE_SECONDS}" "--dataset.reset_time_s=${RESET_SECONDS}" \
      "--resume=${RESUME}" "--display_data=${DISPLAY_DATA}"
    ;;
  train)
    exec "${BIN}/lerobot-train" "--dataset.repo_id=${DATASET_ID}" "--dataset.root=${DATASET_ROOT}" \
      --policy.type=act --policy.device=mps --policy.push_to_hub=false \
      "--output_dir=${POLICY_DIR}" --job_name=act_so101_wrist_pick \
      "--steps=${TRAIN_STEPS}" --batch_size=8 --save_freq=5000 --wandb.enable=false
    ;;
  rollout)
    exec "${BIN}/lerobot-rollout" --strategy.type=base "--policy.path=${POLICY_DIR}/checkpoints/last/pretrained_model" \
      "${ROBOT[@]}" "${CAMERAS}" "--task=${TASK}" "--duration=${ROLLOUT_SECONDS}"
    ;;
  *)
    sed -n 2,3p "$0" >&2
    exit 1
    ;;
esac
