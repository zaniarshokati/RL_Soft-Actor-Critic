"""
train.py

Training script for the Soft Actor-Critic (SAC) agent on the LunarLanderContinuous-v2 environment.

This script:
- Initializes the environment and agent.
- Uses a prioritized replay buffer.
- Implements improvements including layer normalization, gradient clipping,
  learning rate scheduling, TensorBoard logging, and checkpointing.
- Trains the agent and saves the final model.
"""

import gym
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sac_model import SACAgent, PrioritizedReplayBuffer
import os

# Hyperparameters
ENV_NAME = "LunarLanderContinuous-v2"
HIDDEN_SIZE = 256
REPLAY_BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005         # Soft update factor
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
INIT_ALPHA = 0.2

MAX_EPISODES = 1000
MAX_STEPS = 1000    # Max steps per episode
START_STEPS = 10000 # Use random actions until this many steps are collected
UPDATE_AFTER = 1000 # Begin network updates after these many steps
UPDATE_EVERY = 50   # Perform updates every n steps
GRAD_CLIP = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """Main training loop for the SAC agent."""
    env = gym.make(ENV_NAME)
    state, _ = env.reset(seed=0)
    env.action_space.seed(0)
    env.observation_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize the prioritized replay buffer
    replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE, alpha=0.6)

    # Initialize SAC agent
    agent = SACAgent(state_dim, action_dim, HIDDEN_SIZE, init_alpha=INIT_ALPHA)
    # Set up optimizers
    agent.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=LR_ACTOR)
    agent.q1_optimizer = optim.Adam(agent.q1.parameters(), lr=LR_CRITIC)
    agent.q2_optimizer = optim.Adam(agent.q2.parameters(), lr=LR_CRITIC)
    agent.alpha_optimizer = optim.Adam([agent.log_alpha], lr=LR_ACTOR)

    # Set up learning rate schedulers (decay every 100 episodes)
    policy_scheduler = optim.lr_scheduler.StepLR(agent.policy_optimizer, step_size=100, gamma=0.95)
    q1_scheduler = optim.lr_scheduler.StepLR(agent.q1_optimizer, step_size=100, gamma=0.95)
    q2_scheduler = optim.lr_scheduler.StepLR(agent.q2_optimizer, step_size=100, gamma=0.95)
    alpha_scheduler = optim.lr_scheduler.StepLR(agent.alpha_optimizer, step_size=100, gamma=0.95)

    writer = SummaryWriter("runs/sac_training")
    os.makedirs("checkpoints", exist_ok=True)

    total_steps = 0
    beta_start = 0.4
    beta_frames = MAX_EPISODES * MAX_STEPS
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS):
            if total_steps < START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= UPDATE_AFTER and total_steps % UPDATE_EVERY == 0:
                # Linearly increase beta from beta_start to 1 over beta_frames
                beta = min(1.0, beta_start + total_steps * (1.0 - beta_start) / beta_frames)
                for _ in range(UPDATE_EVERY):
                    (state_batch, action_batch, reward_batch, next_state_batch,
                     done_batch, weights, indices) = replay_buffer.sample(BATCH_SIZE, beta=beta)
                    with torch.no_grad():
                        next_action, next_log_prob, _ = agent.policy.sample(next_state_batch)
                        q1_next = agent.q1_target(next_state_batch, next_action)
                        q2_next = agent.q2_target(next_state_batch, next_action)
                        min_q_next = torch.min(q1_next, q2_next) - agent.alpha * next_log_prob
                        q_target = reward_batch + (1 - done_batch) * GAMMA * min_q_next

                    q1_current = agent.q1(state_batch, action_batch)
                    q2_current = agent.q2(state_batch, action_batch)
                    # Compute TD errors for updating priorities
                    td_error1 = (q1_current - q_target).abs().detach().cpu().numpy().flatten() + 1e-6
                    td_error2 = (q2_current - q_target).abs().detach().cpu().numpy().flatten() + 1e-6
                    td_error = (td_error1 + td_error2) / 2.0
                    replay_buffer.update_priorities(indices, td_error)

                    q1_loss = (weights * F.mse_loss(q1_current, q_target, reduction='none')).mean()
                    q2_loss = (weights * F.mse_loss(q2_current, q_target, reduction='none')).mean()

                    agent.q1_optimizer.zero_grad()
                    q1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.q1.parameters(), GRAD_CLIP)
                    agent.q1_optimizer.step()

                    agent.q2_optimizer.zero_grad()
                    q2_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.q2.parameters(), GRAD_CLIP)
                    agent.q2_optimizer.step()

                    new_action, log_prob, _ = agent.policy.sample(state_batch)
                    q1_new = agent.q1(state_batch, new_action)
                    q2_new = agent.q2(state_batch, new_action)
                    q_new = torch.min(q1_new, q2_new)
                    policy_loss = (agent.alpha * log_prob - q_new).mean()

                    agent.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), GRAD_CLIP)
                    agent.policy_optimizer.step()

                    alpha_loss = -(agent.log_alpha * (log_prob + agent.target_entropy).detach()).mean()
                    agent.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    torch.nn.utils.clip_grad_norm_([agent.log_alpha], GRAD_CLIP)
                    agent.alpha_optimizer.step()

                    # Soft-update target networks
                    for target_param, param in zip(agent.q1_target.parameters(), agent.q1.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
                    for target_param, param in zip(agent.q2_target.parameters(), agent.q2.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

                # Log update losses (logged once per update block)
                writer.add_scalar("Loss/q1", q1_loss.item(), total_steps)
                writer.add_scalar("Loss/q2", q2_loss.item(), total_steps)
                writer.add_scalar("Loss/policy", policy_loss.item(), total_steps)
                writer.add_scalar("Loss/alpha", alpha_loss.item(), total_steps)

            if done:
                break
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Total Steps: {total_steps}")

        # Step learning rate schedulers if updates have begun
        if total_steps >= UPDATE_AFTER:
            policy_scheduler.step()
            q1_scheduler.step()
            q2_scheduler.step()
            alpha_scheduler.step()

        # Checkpointing: Save model every 50 episodes
        if episode % 50 == 0:
            checkpoint_path = f"checkpoints/sac_agent_episode_{episode}.pth"
            torch.save(agent, checkpoint_path)
            print(f"Checkpoint saved at episode {episode} to {checkpoint_path}")

    env.close()
    torch.save(agent, "sac_agent.pth")
    print("Training complete. Final agent saved as sac_agent.pth.")
    writer.close()

if __name__ == "__main__":
    main()
