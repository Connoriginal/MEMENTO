#! /bin/bash


export PYTHONPATH="$PYTHONPATH:/HabitatLLM"
CUDA_DEVICES=0 ####### SHOULD CHANGE

now=$(date +%m-%d_%H-%M)

model_name=gpt-4o ####### SHOULD CHANGE
job=top_5_wo_gold ####### SHOULD CHANGE [ex. top_5, top_5_wo_gold, ...]
job_name=${model_name}_${job} 

#### Stage 1 ####

echo "Running single utilization stage with ${job_name}"
data_type_name=single_utilization_stage ###### SHOULD CHANGE

run_name=${model_name}_${data_type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name v1_experiment/single_utilization_stage.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \