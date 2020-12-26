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

import numpy as np
import torch

import yaml

from gqsat.utils import build_argparser
from dqn2 import DQN

def train(args):
    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )

    if args.status_dict_path:
        # training mode, resuming from the status dict

        # load the train status dict
        with open(args.status_dict_path, "r") as f:
            train_status = yaml.load(f, Loader=yaml.Loader)
        # swap the args
        args = train_status["args"]

        dqn = DQN(args, train_status)
    else:
        dqn = DQN(args)

    # train
    dqn.train(args)



if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )

    # training
    train(args)