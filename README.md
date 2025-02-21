# Soft Actor-Critic (SAC) for LunarLanderContinuous-v2

This repository provides a comprehensive and professional implementation of the Soft Actor-Critic (SAC) algorithm for the [LunarLanderContinuous-v2](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) environment from OpenAI Gym. The project includes several advanced improvements:
 
- **Prioritized Experience Replay:** Samples more informative transitions based on TD error.
- **Layer Normalization:** Stabilizes training in both actor and critic networks.
- **Gradient Clipping:** Prevents exploding gradients during training.
- **Learning Rate Scheduling:** Decays learning rates dynamically using PyTorch's schedulers.
- **TensorBoard Logging & Checkpointing:** Monitors training progress and saves model checkpoints.
- **Video Visualization:** Uses Gym’s RecordVideo wrapper to record and save evaluation videos.

## Repository Structure

- `sac_model.py`: Contains all the core model components including the replay buffer (both prioritized and uniform versions), the actor (GaussianPolicy), the critic (QNetwork), and the SACAgent class.
- `train.py`: Implements the training loop, sets up the environment and agent, and saves checkpoints along with TensorBoard logs.
- `visualize.py`: Loads the saved agent and records evaluation episodes as videos.
- `README.md`: This file.
  
## Requirements

- Python 3.6+
- [Gym](https://github.com/openai/gym) (with Box2D support: `gym[box2d]`)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard) (for logging)
- (Optional) [TensorFlow](https://www.tensorflow.org/) if required by TensorBoard

You can install the required packages using:

```bash
pip install gym[box2d] numpy torch torchvision tensorboard
