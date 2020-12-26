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

import os
from collections import deque
import pickle
import copy
import yaml

from gqsat.utils import build_argparser, evaluate, make_env
from gqsat.models import EncoderCoreDecoder, SatModel
from gqsat.agents import GraphAgent
from gqsat.learners import GraphLearner
from gqsat.buffer import ReplayGraphBuffer

from tensorboardX import SummaryWriter


def save_training_state(
    model,
    learner,
    episodes_done,
    transitions_seen,
    best_eval_so_far,
    args,
    in_eval_mode=False,
):
    # save the model
    model_path = os.path.join(args.logdir, f"model_{learner.step_ctr}.chkp")
    torch.save(model.state_dict(), model_path)

    # save the experience replay
    buffer_path = os.path.join(args.logdir, "buffer.pkl")

    with open(buffer_path, "wb") as f:
        pickle.dump(learner.buffer, f)

    # save important parameters
    train_status = {
        "step_ctr": learner.step_ctr,
        "latest_model_name": model_path,
        "buffer_path": buffer_path,
        "args": args,
        "episodes_done": episodes_done,
        "logdir": args.logdir,
        "transitions_seen": transitions_seen,
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "optimizer_class": type(learner.optimizer),
        "best_eval_so_far": best_eval_so_far,
        "scheduler_class": type(learner.lr_scheduler),
        "scheduler_state_dict": learner.lr_scheduler.state_dict(),
        "in_eval_mode": in_eval_mode,
    }
    status_path = os.path.join(args.logdir, "status.yaml")

    with open(status_path, "w") as f:
        yaml.dump(train_status, f, default_flow_style=False)

    return status_path


def get_annealed_eps(n_trans, args):
    if n_trans < args.init_exploration_steps:
        return args.eps_init
    if n_trans > args.eps_decay_steps:
        return args.eps_final
    else:
        assert n_trans - args.init_exploration_steps >= 0
        return (args.eps_init - args.eps_final) * (
            1 - (n_trans - args.init_exploration_steps) / args.eps_decay_steps
        ) + args.eps_final


def arg2activation(activ_str):
    if activ_str == "relu":
        return torch.nn.ReLU
    elif activ_str == "tanh":
        return torch.nn.Tanh
    elif activ_str == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise ValueError("Unknown activation function")

class DQN(object):
    """
    DQN object for setting up env, agent, learner
    Training happens in train() function
    """
    def __init__(self, args, train_status = None):
        self.writer = SummaryWriter()
        
        if train_status is not None:
            self._init_from_status(args, train_status)
        else:
            self._init_from_scratch(args)
        print(args.__str__())

    def _init_from_status(self, args, train_status):
        
        self.eval_resume_signal = train_status["in_eval_mode"]
        # load the model
        net = SatModel.load_from_yaml(os.path.join(args.logdir, "model.yaml")).to(
            args.device
        )
        net.load_state_dict(torch.load(train_status["latest_model_name"]))

        target_net = SatModel.load_from_yaml(
            os.path.join(args.logdir, "model.yaml")
        ).to(args.device)
        target_net.load_state_dict(net.state_dict())

        # load the buffer
        with open(train_status["buffer_path"], "rb") as f:
            self.buffer = pickle.load(f)
        self.learner = GraphLearner(net, target_net, self.buffer, args)
        learner.step_ctr = train_status["step_ctr"]

        learner.optimizer = train_status["optimizer_class"](
            net.parameters(), lr=args.lr
        )
        learner.optimizer.load_state_dict(train_status["optimizer_state_dict"])
        learner.lr_scheduler = train_status["scheduler_class"](
            learner.optimizer, args.lr_scheduler_frequency, args.lr_scheduler_gamma
        )
        learner.lr_scheduler.load_state_dict(train_status["scheduler_state_dict"])

        # load misc training status params
        self.n_trans = train_status["transitions_seen"]
        self.ep = train_status["episodes_done"]

        self.env = make_env(args.train_problems_paths, args, test_mode=False)

        self.agent = GraphAgent(net, args)

        self.best_eval_so_far = train_status["best_eval_so_far"]

    def _init_from_scratch(self, args):
        # training mode, learning from scratch or continuing learning from some previously trained model
        args.logdir = self.writer.logdir

        model_save_path = os.path.join(args.logdir, "model.yaml")
        self.best_eval_so_far = (
            {args.eval_problems_paths: -1}
            if not args.eval_separately_on_each
            else {k: -1 for k in args.eval_problems_paths.split(":")}
        )

        self.env = make_env(args.train_problems_paths, args, test_mode=False)
        if args.model_dir is not None:
            # load an existing model and continue training
            net = SatModel.load_from_yaml(
                os.path.join(args.model_dir, "model.yaml")
            ).to(args.device)
            net.load_state_dict(
                torch.load(os.path.join(args.model_dir, args.model_checkpoint))
            )
        else:
            # learning from scratch
            net = EncoderCoreDecoder(
                (self.env.vertex_in_size, self.env.edge_in_size, self.env.global_in_size),
                core_out_dims=(
                    args.core_v_out_size,
                    args.core_e_out_size,
                    args.core_e_out_size,
                ),
                out_dims=(2, None, None),
                core_steps=args.core_steps,
                dec_out_dims=(
                    args.decoder_v_out_size,
                    args.decoder_e_out_size,
                    args.decoder_e_out_size,
                ),
                encoder_out_dims=(
                    args.encoder_v_out_size,
                    args.encoder_e_out_size,
                    args.encoder_e_out_size,
                ),
                save_name=model_save_path,
                e2v_agg=args.e2v_aggregator,
                n_hidden=args.n_hidden,
                hidden_size=args.hidden_size,
                activation=arg2activation(args.activation),
                independent_block_layers=args.independent_block_layers,
            ).to(args.device)
        print(str(net))
        target_net = copy.deepcopy(net)

        self.buffer = ReplayGraphBuffer(args, args.buffer_size)
        self.agent = GraphAgent(net, args)

        self.n_trans = 0
        self.ep = 0
        self.learner = GraphLearner(net, target_net, self.buffer, args)
        self.eval_resume_signal = False

    def train(self, args):
        """
        training happens here.
        args: arguments in training
        """
        while self.learner.step_ctr < args.batch_updates:

            ret = 0
            obs = self.env.reset(args.train_time_max_decisions_allowed)
            done = self.env.isSolved

            if args.history_len > 1:
                raise NotImplementedError(
                    "History len greater than one is not implemented for graph nets."
                )
            hist_buffer = deque(maxlen=args.history_len)
            for _ in range(args.history_len):
                hist_buffer.append(obs)
            ep_step = 0

            save_flag = False

            while not done:
                annealed_eps = get_annealed_eps(self.n_trans, args)
                action = self.agent.act(hist_buffer, eps=annealed_eps)
                next_obs, r, done, _ = self.env.step(action)
                self.buffer.add_transition(obs, action, r, done)
                obs = next_obs

                hist_buffer.append(obs)
                ret += r

                if (not self.n_trans % args.step_freq) and (
                    self.buffer.ctr > max(args.init_exploration_steps, args.bsize + 1)
                    or self.buffer.full
                ):
                    step_info = self.learner.step()
                    if annealed_eps is not None:
                        step_info["annealed_eps"] = annealed_eps

                    # we increment the step_ctr in the learner.step(), that's why we need to do -1 in tensorboarding
                    # we do not need to do -1 in checking for frequency since 0 has already passed

                    if not self.learner.step_ctr % args.save_freq:
                        # save the exact model you evaluated and make another save after the episode ends
                        # to have proper transitions in the replay buffer to pickle
                        status_path = save_training_state(
                            self.agent.net, #TODO : It was only net (but this should also be correct)
                            self.learner,
                            self.ep - 1,
                            self.n_trans,
                            self.best_eval_so_far,
                            args,
                            in_eval_mode=self.eval_resume_signal,
                        )
                        save_flag = True
                    if (
                        args.env_name == "sat-v0" and not self.learner.step_ctr % args.eval_freq
                    ) or self.eval_resume_signal:
                        scores, _, self.eval_resume_signal = evaluate(
                            self.agent, args, include_train_set=False
                        )

                        for sc_key, sc_val in scores.items():
                            # list can be empty if we hit the time limit for eval
                            if len(sc_val) > 0:
                                res_vals = [el for el in sc_val.values()]
                                median_score = np.nanmedian(res_vals)
                                if (
                                    self.best_eval_so_far[sc_key] < median_score
                                    or self.best_eval_so_far[sc_key] == -1
                                ):
                                    self.best_eval_so_far[sc_key] = median_score
                                self.writer.add_scalar(
                                    f"data/median relative score: {sc_key}",
                                    np.nanmedian(res_vals),
                                    self.learner.step_ctr - 1,
                                )
                                self.writer.add_scalar(
                                    f"data/mean relative score: {sc_key}",
                                    np.nanmean(res_vals),
                                    self.learner.step_ctr - 1,
                                )
                                self.writer.add_scalar(
                                    f"data/max relative score: {sc_key}",
                                    np.nanmax(res_vals),
                                    self.learner.step_ctr - 1,
                                )
                        for k, v in self.best_eval_so_far.items():
                            self.writer.add_scalar(k, v, self.learner.step_ctr - 1)

                    for k, v in step_info.items():
                        self.writer.add_scalar(k, v, self.learner.step_ctr - 1)

                    self.writer.add_scalar("data/num_episodes", self.ep, self.learner.step_ctr - 1)

                self.n_trans += 1
                ep_step += 1

            self.writer.add_scalar("data/ep_return", ret, self.learner.step_ctr - 1)
            self.writer.add_scalar("data/ep_steps", self.env.step_ctr, self.learner.step_ctr - 1)
            self.writer.add_scalar("data/ep_last_reward", r, self.learner.step_ctr - 1)
            print(f"Episode {self.ep + 1}: Return {ret}.")
            self.ep += 1

            if save_flag:
                status_path = save_training_state(
                    self.agent.net, #TODO: Is agent net the same as net?
                    self.learner,
                    self.ep - 1,
                    self.n_trans,
                    self.best_eval_so_far,
                    args,
                    in_eval_mode=self.eval_resume_signal,
                )
                save_flag = False