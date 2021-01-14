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
PATH_TO_TEST_DATA=/cluster/scratch/aunagar/graphqsat/data/unifrandom3sat/uuf50_218/test/
MODEL_DIR=/cluster/scratch/aunagar/dlproject/trained_models/solver/cuda/
CHECKPOINT=model_31000
LOGDIR=./log
python3 main.py \
  --evaluate \
  --logdir $LOGDIR \
  --env-name sat-v0 \
  --core-steps -1 \
  --eps-final 0.0 \
  --eval-time-limit 100000000000000 \
  --no_restarts \
  --test_time_max_decisions_allowed 500 \
  --eval-problems-paths $PATH_TO_TEST_DATA \
  --model-dir $MODEL_DIR \
  --model-checkpoint $CHECKPOINT.chkp
