from gymnasium.envs.registration import register

register(
    id="custom_gyms/JugglingWorld-v0",
    entry_point="",
    max_episode_steps=300,
)
