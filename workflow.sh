#!/bin/bash

# Configuration: Set to true to run the corresponding step, or false to skip.
# These can be overridden by environment variables, e.g., RUN_STEP1=false ./workflow.sh
RUN_STEP1=${RUN_STEP1:-false}
RUN_STEP2=${RUN_STEP2:-false}
RUN_STEP3=${RUN_STEP3:-true}
RUN_STEP4=${RUN_STEP4:-false}

# 1. Download all PH sim datasets
if [ "$RUN_STEP1" = true ]; then
    echo "Running Step 1: Download all PH sim datasets..."
    python robomimic/robomimic/scripts/download_datasets.py --tasks sim --dataset_types ph --hdf5_types raw --download_dir ./datasets
else
    echo "Skipping Step 1: Download all PH sim datasets."
fi

# 2. Extract low-dim and image observations from all PH sim datasets
if [ "$RUN_STEP2" = true ]; then
    echo "Running Step 2: Extract low-dim and image observations from all PH sim datasets..."
    BASE_DATASET_DIR="./datasets"
    echo "Using base dataset directory: $BASE_DATASET_DIR"
    BASE_SCRIPT_DIR="./robomimic/robomimic/scripts"
    echo "Using base script directory: $BASE_SCRIPT_DIR"

    # lift - ph
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/lift/ph/demo_v15.hdf5 \
    --output_name low_dim_v15.hdf5
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/lift/ph/demo_v15.hdf5 \
    --output_name image_v15.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # can - ph
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/can/ph/demo_v15.hdf5 \
    --output_name low_dim_v15.hdf5
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/can/ph/demo_v15.hdf5 \
    --output_name image_v15.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # square - ph
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/square/ph/demo_v15.hdf5 \
    --output_name low_dim_v15.hdf5
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/square/ph/demo_v15.hdf5 \
    --output_name image_v15.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # transport - ph
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/transport/ph/demo_v15.hdf5 \
    --output_name low_dim_v15.hdf5
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/transport/ph/demo_v15.hdf5 \
    --output_name image_v15.hdf5 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand --camera_height 84 --camera_width 84

    # tool hang - ph
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/tool_hang/ph/demo_v15.hdf5 \
    --output_name low_dim_v15.hdf5
    python $BASE_SCRIPT_DIR/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/tool_hang/ph/demo_v15.hdf5 \
    --output_name image_v15.hdf5 --camera_names sideview robot0_eye_in_hand --camera_height 240 --camera_width 240
else
    echo "Skipping Step 2: Extract low-dim and image observations from all PH sim datasets."
fi

# choose a task
TASK="can"

# 3. Convert lift task to VLA and back
if [ "$RUN_STEP3" = true ]; then
    echo "Running Step 3: Convert $TASK task to VLA and back..."
    # Ensure robodm is in PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$(pwd)/robodm
    
    python robomimic_to_vla_compressed.py \
        --dataset datasets/$TASK/ph/image_v15.hdf5 \
        --output_dir datasets/$TASK/ph/image_v15_vla
        
    python vla_to_robomimic.py \
        --vla_dir datasets/$TASK/ph/image_v15_vla \
        --output_path datasets/$TASK/ph/image_v15_reconstructed.hdf5
else
    echo "Skipping Step 3: Convert $TASK task to VLA and back."
fi

# 4. Train BC model on all PH sim datasets
if [ "$RUN_STEP4" = true ]; then
    echo "Running Step 4: Train BC model on all PH sim datasets..."
    python robomimic/robomimic/scripts/train.py --config bc.json --dataset datasets/$TASK/ph/image_v15_reconstructed.hdf5 --debug
else
    echo "Skipping Step 4: Train BC model on $TASK task."
fi


# important to benchmark
# python robomimic/robomimic/scripts/run_trained_agent.py     --agent test_output/test_lift_simple/20260107220351/models/model_epoch_500.pth     --n_rollouts 50     --seed 0