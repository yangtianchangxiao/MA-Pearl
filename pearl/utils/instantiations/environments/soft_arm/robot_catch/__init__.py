from gymnasium.envs.registration import register

register(
    id="RobotArmEnv-v0",
    entry_point="robot_catch.env:RobotArmEnv",
)

register(
    id="RobotCatchEnv-v0",
    entry_point="robot_catch.env:RobotCatchEnv",
    max_episode_steps=1000,
)
