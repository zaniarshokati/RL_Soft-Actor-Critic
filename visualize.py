"""
visualize.py

Visualization script for the trained Soft Actor-Critic (SAC) agent on LunarLanderContinuous-v2.

This script:
- Loads the trained agent from a saved checkpoint.
- Uses Gym’s RecordVideo wrapper to record evaluation episodes.
- Saves the recorded videos in the specified folder.
"""

import gym
import torch
from gym.wrappers import RecordVideo
import sac_model
import __main__
# Register classes in __main__ for pickle compatibility
__main__.GaussianPolicy = sac_model.GaussianPolicy
__main__.QNetwork = sac_model.QNetwork
__main__.SACAgent = sac_model.SACAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "LunarLanderContinuous-v2"

def visualize_policy_video(agent, num_episodes=3, video_folder="videos"):
    """
    Record and save evaluation episodes as videos.

    Args:
        agent (SACAgent): The trained SAC agent.
        num_episodes (int): Number of episodes to record.
        video_folder (str): Folder where the videos will be saved.
    """
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode: True)
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        print(f"Recorded Evaluation Episode {episode+1}: Reward {episode_reward:.2f}")
    env.close()
    print(f"Videos saved to folder '{video_folder}'.")

def main():
    """
    Load the trained SAC agent and record evaluation videos.
    """
    agent = torch.load("sac_agent.pth", map_location=device)
    agent.policy.to(device)
    visualize_policy_video(agent, num_episodes=3, video_folder="videos")

if __name__ == "__main__":
    main()
