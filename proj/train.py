# Based on file in ppo.py
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ai_economist import foundation
from proj.config import env_config
from proj.model import ActorCritic
from tutorials.utils import plotting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############## Hyperparameters ##############
solved_reward = 10_000
log_interval = 10
max_episodes = 50000
graphical_episode_log_frequency = 1
n_latent_var = 256
update_timestep = 2000
lr = 0.001
betas = (0.9, 0.999)
gamma = 0.99
K_epochs = 4
eps_clip = 0.2
random_seed = None


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)  # rewards = [discounted_reward, *rewards]

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def sample_random_action(agent, mask):  # For planner
    """Sample random UNMASKED action(s) for agent."""
    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
        # print(len(split_masks))
        # print(split_masks[0] / split_masks[0].sum())
        return [np.random.choice(np.arange(len(m_)), p=m_ / m_.sum()) for m_ in split_masks]

    # Return a single action
    else:
        return np.random.choice(np.arange(agent.action_spaces), p=mask / mask.sum())


def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""

    actions = {
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions


def main():
    log_dir = Path(__file__).parent / f"exp{int(time.time())}"
    log_dir.mkdir()
    env = foundation.make_env_instance(**env_config)
    state = env.reset()

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = [Memory() for _ in range(env.n_agents)]

    action_dim = state['0']["action_mask"].size  # todo mask tells which action cannot be taken
    state_dim = state['0']["flat"].size

    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, device)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        if i_episode % graphical_episode_log_frequency == 0:
            obs = env.reset(force_dense_logging=True)
        else:
            obs = env.reset()

        for t in range(env.episode_length):
            time_step += 1
            # Running policy_old:

            actions = sample_random_actions(env, obs)

            for agent_id in range(env.n_agents):
                agent_id_str = str(agent_id)
                memory_agent = memory[agent_id]
                action_mask = torch.tensor(state[agent_id_str]["action_mask"], device=device)
                agent_state = state[agent_id_str]["flat"]
                action = ppo.policy_old.act(agent_state, memory_agent, action_mask)
                actions[agent_id_str] = action

            state, reward, done, info = env.step(actions)
            # Saving reward and is_terminals:
            for agent_id in range(env.n_agents):
                agent_id_str = str(agent_id)
                memory_agent = memory[agent_id]
                agent_reward = -reward[agent_id_str] # TBH NOT SURE
                memory_agent.rewards.append(agent_reward)
                memory_agent.is_terminals.append(done)

                # update if its time
                if time_step % update_timestep == 0:
                    ppo.update(memory_agent)
                    memory_agent.clear_memory()
                    time_step = 0
                running_reward += agent_reward
            if done['__all__']:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), log_dir / 'ckpt-final.pth')
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), log_dir / f'ckpt-{i_episode}.pth')

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = float((running_reward / log_interval))

            print(f'Episode {i_episode} \t Avg length: {avg_length} \t Avg reward: {running_reward}')
            running_reward = 0
            avg_length = 0

        if i_episode % graphical_episode_log_frequency == 0:
            dense_log = env.previous_episode_dense_log
            (fig0, fig1, fig2), incomes, endows, c_trades, all_builds = plotting.breakdown(dense_log)
            print(f"{incomes=}, {endows=}, {c_trades=}, {all_builds=}")
            # fig0.savefig(log_dir / f"fig0-{i_episode}.png", dpi=fig0.dpi)
            fig1.savefig(log_dir / f"fig1-{i_episode:04d}.png", dpi=fig1.dpi)
            fig2.savefig(log_dir / f"fig2-{i_episode:04d}.png", dpi=fig2.dpi)
            plt.close(fig0)
            plt.close(fig1)
            plt.close(fig2)
            with open(log_dir / f'logs-{i_episode:04d}.pickle', 'wb') as handle:
                pickle.dump(dense_log, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
