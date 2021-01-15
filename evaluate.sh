# Copyright 2019-2020 Nvidia Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# PATH_TO_EVAL_DATA=/cluster/scratch/aunagar/graphqsat/data/graphcoloring/flat50_115/
# PATH_TO_EVAL_DATA=/cluster/scratch/aunagar/graphqsat/data/generated/sat-218-50-mc218-restrict2-it1-donefull-deterpi-bs64-core10-lowerlr-tf4-e1-exp5000-checkpointedsolver/test/
# PATH_TO_EVAL_DATA=/cluster/scratch/aunagar/graphqsat/data/generated/sat-430-100-mc430-restrict2-it1-donefull-deterpi-bs64-core10-tf4-e1-exp5000/
PATH_TO_EVAL_DATA=/cluster/scratch/aunagar/graphqsat/data/generated/sat-218-50-mc218-restrict2-it1-donefull-deterpi-bs64-core10-tf4-e1-exp5000_v3/test/
# PATH_TO_EVAL_DATA=/cluster/home/aunagar/dl-project-2020/dataset/generated/gen430-100from218-50/
# PATH_TO_EVAL_DATA=/cluster/home/aunagar/dl-project-2020/dataset/generated/sat-218-50-mc218-restrict2-it1-donefull-deterpi-bs64-core10-tf4-e1-exp5000
# PATH_TO_EVAL_DATA=/cluster/scratch/aunagar/graphqsat/data/unifrandom3sat/uf100_430/test/
# PATH_TO_EVAL_DATA=/cluster/scratch/aunagar/graphqsat/data/unifrandom3sat/uf125_538/test/
# PATH_TO_EVAL_DATA=/cluster/scratch/aunagar/graphqsat/data/unifrandom3sat/uf50_218/test/
MODEL_DIR=/cluster/scratch/aunagar/dlproject/trained_models/solver/cuda/
CHECKPOINT=model_31000
# MODEL_DIR=/cluster/scratch/aunagar/graphqsat/runs//Jan13_GenSATdata_checkpointed/
# CHECKPOINT=model_27000
# MODEL_DIR=/cluster/scratch/aunagar/graphqsat/runs//50_218_mixeddata_checkpointed/
# CHECKPOINT=model_28000

python3 main.py \
  --evaluate \
  --logdir ./log \
  --env-name sat-v0 \
  --core-steps -1 \
  --eps-final 0.0 \
  --eval-time-limit 100000000000000 \
  --no_restarts \
  --test_time_max_decisions_allowed 500 \
  --eval-problems-paths $PATH_TO_EVAL_DATA \
  --model-dir $MODEL_DIR \
  --model-checkpoint $CHECKPOINT.chkp
