import gym
import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl
from tensorboardX import SummaryWriter

from NormalizedActions import NormalizedActions
from Metrics import RunningMean
from Networks import SoftQNetwork, PolicyNetwork
from Hyperparameters import *
import utils


class SAC(object):
    def __init__(self, env_name, data_save_dir=None):
        """Initializes: 
            environment
            networks
            directories
            optimizers
            variables


        Args:
            env_name (str): Which environment?
            data_save_dir (None, str, optional): Where to save summary and checkpoints?
        """
        assert isinstance(env_name, str)
        assert isinstance(data_save_dir, str) or data_save_dir is None

        self.iteration = 0

        self.env = NormalizedActions(gym.make(env_name))
        self.eval_env = NormalizedActions(gym.make(env_name))

        self.a_dim = self.eval_env.action_space.shape[0]
        self.o_dim = self.eval_env.observation_space.shape[0]

        self.data_save_dir = data_save_dir

        if not os.path.exists(self.data_save_dir):
            os.mkdir(self.data_save_dir)
            os.mkdir(os.path.join(self.data_save_dir, "saves"))

        self.q0_net = SoftQNetwork(
            self.o_dim, self.a_dim, HIDDEN_SIZE).to(DEVICE, non_blocking=True)
        self.q1_net = SoftQNetwork(
            self.o_dim, self.a_dim, HIDDEN_SIZE).to(DEVICE, non_blocking=True)
        self.q0_target_net = SoftQNetwork(
            self.o_dim, self.a_dim, HIDDEN_SIZE).to(DEVICE, non_blocking=True)
        self.q0_target_net = utils.hard_copy(self.q0_target_net, self.q0_net)
        self.q1_target_net = SoftQNetwork(
            self.o_dim, self.a_dim, HIDDEN_SIZE).to(DEVICE, non_blocking=True)
        self.q1_target_net = utils.hard_copy(self.q1_target_net, self.q1_net)

        self.pi_net = PolicyNetwork(self.o_dim, self.a_dim, HIDDEN_SIZE, LOGSTD_MIN, LOGSTD_MAX)
        self.pi_net = self.pi_net.to(DEVICE, non_blocking=True)

        self.replay_buffer = REPLAY_BUFFER(REPLAY_SIZE)

        self.q0_optim = Q_OPTIM(self.q0_net.parameters(), lr=Q_LEARNING_RATE)
        self.q1_optim = Q_OPTIM(self.q1_net.parameters(), lr=Q_LEARNING_RATE)
        self.pi_optim = POLICY_OPTIM(self.pi_net.parameters(), lr=POLICY_LEARNING_RATE)

        self.log_alpha = torch.zeros(1, device=DEVICE, requires_grad=True)
        self.alpha_optim = ALPHA_OPTIM([self.log_alpha], lr=ALPHA_LEARNING_RATE)
        self.entropy_target = - \
            torch.prod(torch.Tensor(self.eval_env.action_space.shape).to(
                DEVICE, non_blocking=True)).item()

    def get_checkpoint_file(self, model_name):
        """Returns the checkpoint path for saving a model given the file name

        Args:
            model_name (str): file name

        Returns:
            str: Path
        """
        assert isinstance(model_name, str)
        return os.path.join(
            f"{self.data_save_dir}/saves/iter_{self.iteration}",
            model_name
        )

    def save_models(self):
        """Saves a checkpoint of all the models
        """
        if not os.path.exists(f"{self.data_save_dir}/saves/iter_{self.iteration}"):
            os.mkdir(f"{self.data_save_dir}/saves/iter_{self.iteration}")
        self.q0_net.save(
            self.get_checkpoint_file("Q0_net_state_dict")
        )
        self.q1_net.save(
            self.get_checkpoint_file("Q1_net_state_dict")
        )
        self.q0_target_net.save(
            self.get_checkpoint_file("Q0_target_net_state_dict")
        )
        self.q1_target_net.save(
            self.get_checkpoint_file("Q1_target_net_state_dict")
        )
        self.pi_net.save(
            self.get_checkpoint_file("Policy_net_state_dict")
        )
        self.replay_buffer.save(
            self.get_checkpoint_file("Replay_Buffer_data")
        )
        torch.save(
            self.q0_optim.state_dict(),
            self.get_checkpoint_file("Q0_optimizer_state_dict")
        )
        torch.save(
            self.q1_optim.state_dict(),
            self.get_checkpoint_file("Q1_optimizer_state_dict")
        )
        torch.save(
            self.pi_optim.state_dict(),
            self.get_checkpoint_file("Policy_optimizer_state_dict")
        )
        torch.save(
            self.log_alpha,
            self.get_checkpoint_file("Entropy_Coefficient")
        )

    def load_models(self):
        """Loads all the models from the latest checkpoint
        """
        load_path, self.iteration = self.get_latest_checkpoint(
            return_iteration=True
        )
        self.q0_net.load(os.path.join(
            load_path, "Q0_net_state_dict"
        ))
        self.q1_net.load(os.path.join(
            load_path, "Q1_net_state_dict"
        ))
        self.q0_target_net.load(os.path.join(
            load_path, "Q0_target_net_state_dict"
        ))
        self.q1_target_net.load(os.path.join(
            load_path, "Q1_target_net_state_dict"
        ))
        self.pi_net.load(os.path.join(
            load_path, "Policy_net_state_dict"
        ))
        self.q0_optim.load_state_dict(torch.load(os.path.join(
            load_path, "Q0_optimizer_state_dict"
        )))
        self.q1_optim.load_state_dict(torch.load(os.path.join(
            load_path, "Q1_optimizer_state_dict"
        )))
        self.pi_optim.load_state_dict(torch.load(os.path.join(
            load_path, "Policy_optimizer_state_dict"
        )))
        self.log_alpha = torch.load(os.path.join(
            load_path,
            "Entropy_Coefficient"
        ))
        self.replay_buffer.load(os.path.join(
            load_path,
            "Replay_Buffer_data"
        ))

    def train(self, resume_training=False):
        """Does the full training procedure

        Args:
            resume_training (bool, optional): Whether to load from the latest checkpoint and continue training or not.
        """
        assert isinstance(resume_training, bool)

        self.writer = SummaryWriter(
            os.path.join(self.data_save_dir, "summary")
        )

        resume_training = resume_training is True and self.data_save_dir is not None

        if resume_training:
            self.load_models()

        print(f"Starting training from iteration {self.iteration} for a total of {NUM_ITERATIONS} iterations")

        cumulative_reward = 0
        average_return = RunningMean()
        start_time = time.time()
        start_iteration = self.iteration
        while self.iteration < NUM_ITERATIONS:
            s = self.env.reset()
            d = False
            while not d:
                if len(self.replay_buffer) >= INITIAL_REPLAY_SIZE:
                    a = self.pi_net.forward_action(s)
                else:
                    a = self.env.action_space.sample()
                ns, r, d, info = self.env.step(a)
                self.replay_buffer.add(s, a, r, ns, d)
                s = ns

                cumulative_reward += r
                self.iteration += 1

                update_cond = \
                    len(self.replay_buffer) >= BATCH_SIZE and \
                    self.iteration % GRADIENT_STEPS == 0
                if update_cond:
                    self.exploration_flag = 1
                    self._update()

                if self.iteration % SAVE_FREQ == 0:
                    print("Saving checkpoint...")
                    self.save_models()
                    print("Resuming training...")

                if self.iteration % EVAL_FREQ == 0:
                    mean_cumulative_reward, time_eval = self.test()
                    self.writer.add_scalar(
                        "metrics/eval_rewards",
                        mean_cumulative_reward,
                        self.iteration
                    )
                    print(
                        f"Iter {self.iteration} - Evaluation Reward {mean_cumulative_reward:.4f}"
                    )
                    start_time += time_eval

            average_return.add(cumulative_reward)

            self.writer.add_scalar(
                "metrics/cumulative_reward",
                cumulative_reward,
                self.iteration
            )

            self.writer.add_scalar(
                "metrics/average_epoch_return",
                average_return.result(),
                self.iteration
            )
            epoch_time = time.time() - start_time

            self.writer.add_scalar(
                "metrics/seconds_per_epoch",
                epoch_time,
                self.iteration
            )
            epoch_steps = self.iteration - start_iteration

            self.writer.add_scalar(
                "metrics/Steps",
                epoch_steps,
                self.iteration
            )
            print(
                f"Iter {self.iteration} - Reward {cumulative_reward:.4f} - Return {average_return.result():.4f}"
            )
            cumulative_reward = 0
            start_time = time.time()
            start_iteration = self.iteration

    def get_latest_checkpoint(self, return_iteration=False):
        """Returns the latest checkpoint path

        Args:
            return_iteration (bool, optional): Also pass information about iteration of latest checkpoint?

        Returns:
            (str, tuple): (checkpoint path, (checkpoint path, iteration))
        """
        assert isinstance(return_iteration, bool)

        entries = os.listdir(f"{self.data_save_dir}/saves/")
        iters = np.empty((len(entries),), dtype=np.int32)
        for i, entry in enumerate(entries):
            iters[i] = entry.split("_")[1]
        model_path = f"{self.data_save_dir}/saves/iter_{iters.max()}"
        if return_iteration:
            return model_path, iters.max()
        return model_path

    def test(self, render=False, use_internal_policy=True, num_games=None):
        """Play some number of environments for testing purposes

        Args:
            render (bool, optional): Render the gym environment?
            use_internal_policy (bool, optional): If yes, just take current policy, if no, load latest policy checkpoint.
            num_games (None, int, optional): How many games to play?

        Returns:
            tuple: (mean cumulative reward, the time everything took)
        """
        assert isinstance(render, bool)
        assert isinstance(use_internal_policy, bool)
        assert isinstance(num_games, int) or num_games is None

        start_time = time.time()
        if use_internal_policy:
            PI_NET = self.pi_net
        else:
            PI_NET = PolicyNetwork(self.o_dim, self.a_dim,
                                   HIDDEN_SIZE, LOGSTD_MIN, LOGSTD_MAX)
            PI_NET = PI_NET.to(DEVICE, non_blocking=True)
            PI_NET.load(os.path.join(
                self.get_latest_checkpoint(), "Policy_net_state_dict"))
        cumulative_rewards = np.empty(
            (NUM_EVAL_GAMES if num_games is None else num_games, ))
        for i in range(NUM_EVAL_GAMES if num_games is None else num_games):
            s = self.eval_env.reset()
            cumulative_reward = 0
            while True:
                if render is True:
                    self.eval_env.render()
                a = PI_NET.forward_action(s, deterministic=True)
                ns, r, d, info = self.eval_env.step(a)
                cumulative_reward += r

                s = ns

                if d:
                    # self.eval_env.close()
                    break
            cumulative_rewards[i] = cumulative_reward
        self.eval_env.close()
        return cumulative_rewards.mean(), time.time() - start_time

    def _update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).to(DEVICE, non_blocking=True)
        action = torch.FloatTensor(action).to(DEVICE, non_blocking=True)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE, non_blocking=True)
        next_state = torch.FloatTensor(next_state).to(DEVICE, non_blocking=True)
        done = torch.FloatTensor(np.float16(done)).unsqueeze(1).to(DEVICE, non_blocking=True)

        new_action, logprob, _, mean, log_stddev = self.pi_net.evaluate(state)

        alpha_loss = ((-self.log_alpha) * (logprob + self.entropy_target).detach()).sum()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        alpha = self.log_alpha.exp()

        q_value = torch.min(
            self.q0_net(state, new_action),
            self.q1_net(state, new_action)
        )
        pi_loss = (alpha * logprob - q_value).sum()

        pred_q_value_0 = self.q0_net(state, action)
        pred_q_value_1 = self.q1_net(state, action)

        # with torch.no_grad():
        next_action, next_logprob, _, _, _ = self.pi_net.evaluate(next_state)
        next_q_value = torch.min(
            self.q0_target_net(next_state, next_action),
            self.q1_target_net(next_state, next_action)
        )
        next_soft_q_value = next_q_value - alpha * next_logprob
        q_value_target = reward + GAMMA * (1. - done) * next_soft_q_value

        # q0_loss = 0.5 * self.q_loss(
        #    pred_q_value_0, q_value_target
        # )
        q0_loss = 0.5 * (pred_q_value_0 - q_value_target.detach()).pow(2)
        q0_loss = q0_loss.sum()
        # q1_loss = 0.5 * self.q_loss(
        #    pred_q_value_1, q_value_target
        # )
        q1_loss = 0.5 * (pred_q_value_1 - q_value_target.detach()).pow(2)
        q1_loss = q1_loss.sum()

        self.q0_optim.zero_grad()
        q0_loss.backward()
        self.q0_optim.step()

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        self.q0_target_net = utils.soft_copy(self.q0_target_net, self.q0_net, TAU)
        self.q1_target_net = utils.soft_copy(self.q1_target_net, self.q1_net, TAU)

        if self.iteration % SUMMARY_FREQ == 0:

            _, _, _, new_mean, new_log_stddev = self.pi_net.evaluate(next_state)

            old_pi = Normal(mean, log_stddev.exp())
            new_pi = Normal(new_mean, new_log_stddev.exp())

            self.writer.add_scalar(
                "metrics/KL", kl.kl_divergence(old_pi, new_pi).sum(), self.iteration
            )

            self.writer.add_scalar(
                "metrics/alpha", self.log_alpha.exp(), self.iteration
            )
            self.writer.add_scalar(
                "metrics/stddev", log_stddev.mean().exp(), self.iteration
            )

            self.writer.add_scalar(
                "losses/q_loss", (q0_loss + q1_loss) / 2.0, self.iteration
            )
            self.writer.add_scalar(
                "losses/pi_loss", pi_loss, self.iteration
            )
            self.writer.add_scalar(
                "losses/alpha_loss", alpha_loss, self.iteration
            )
