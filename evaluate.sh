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

########### choose your test data ############
##### random
PATH_TO_TEST_DATA=../../dataset/random/sat50_218/test/
# PATH_TO_TEST_DATA=../../dataset/random/unsat50_218/test/
# PATH_TO_TEST_DATA=../../dataset/random/sat100_430/test/
# PATH_TO_TEST_DATA=../../dataset/random/unsat100_430/test/

#### generated
# PATH_TO_TEST_DATA=../../dataset/generated/gensat/50_218/test/
# PATH_TO_TEST_DATA=../../dataset/generated/gensat/100_430/test

##### graph coloring
# PATH_TO_TEST_DATA=../../dataset/graphcoloring/flat50_115/
# PATH_TO_TEST_DATA=../../dataset/graphcoloring/flat100_239/
# PATH_TO_TEST_DATA=../../dataset/graphcoloring/flat150_360/
# PATH_TO_TEST_DATA=../../dataset/graphcoloring/flat200_479/

########## choose your model ############
# MODEL_DIR=../../trained_models/solver/gqsat-original/
# CHECKPOINT=model_31000
MODEL_DIR=../../trained_models/solver/gqsat-star/
CHECKPOINT=model_28000

########## choose dir for tensorboard logs #####
LOGDIR=./runs/test/

python add_metadata.py --eval-problems-paths=$PATH_TO_TEST_DATA
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
