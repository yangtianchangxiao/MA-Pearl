# Robot Catch Environment

This repository contains two implementations of a Soft Actor-Critic (SAC) agent for training a robotic arm to catch balls:

1. OpenRL Implementation: A simple and efficient implementation using the OpenRL framework
2. Optimized SAC Implementation: A highly optimized custom implementation with various performance enhancements

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/robot_catch_env.git
cd robot_catch_env
```

2. Install dependencies for OpenRL version:
```bash
pip install -r requirements_openrl.txt
```

Or for the optimized version:
```bash
pip install -r requirements_optimized.txt
```

## Usage

### OpenRL Version

1. Train the agent:
```bash
python robot_catch/train_openrl.py --config robot_catch/config.yaml --version openrl
```

2. The trained model will be saved in `models/openrl/` directory.

### Optimized SAC Version

1. Train the agent:
```bash
python robot_catch/train_optimized.py --config robot_catch/config.yaml --version optimized
```

2. The trained model will be saved in `models/optimized/` directory.

## Configuration

Both implementations use the same configuration file (`config.yaml`) with different sections:

- `openrl`: Configuration for OpenRL version
- `optimized`: Configuration for optimized SAC version

Key hyperparameters:
- `total_time_steps`: Total number of training steps
- `learning_rate`: Learning rate for all networks
- `batch_size`: Batch size for training
- `gamma`: Discount factor
- `tau`: Target network update rate
- `buffer_size`: Size of replay buffer
- `num_envs`: Number of parallel environments
- `device`: Training device (cuda/cpu)

## Features

### OpenRL Version
- Simple and clean implementation
- Easy to modify and extend
- Built-in optimizations
- Excellent documentation and community support

### Optimized SAC Version
- Mixed precision training
- Prioritized experience replay
- Parallel environment sampling
- Advanced network architectures
- Memory-efficient implementations
- Gradient clipping
- Layer normalization

## Project Structure

```
robot_catch_env/
├── robot_catch/
│   ├── sac_optimized/
│   │   ├── buffer.py      # Optimized replay buffer
│   │   ├── networks.py    # Network architectures
│   │   └── sac.py         # SAC implementation
│   ├── train_openrl.py    # OpenRL training script
│   ├── train_optimized.py # Optimized SAC training script
│   └── config.yaml        # Configuration file
├── requirements_openrl.txt
└── requirements_optimized.txt
```

## Monitoring Training

Both implementations support:
- WandB logging
- TensorBoard visualization
- Console progress updates
- Model checkpointing

To view training progress:
1. WandB: Visit your WandB project page
2. TensorBoard: Run `tensorboard --logdir logs/`

## Performance Comparison

You can compare the performance of both implementations using the following metrics:
- Training speed (steps/second)
- Sample efficiency
- Final performance
- Memory usage
- GPU utilization

## Contributing

Feel free to submit issues and enhancement requests!
