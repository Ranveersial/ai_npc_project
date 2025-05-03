import gymnasium as gym
from gymnasium import spaces
import numpy as np

class JanitorTrustEnv(gym.Env):
    def __init__(self):
        super(JanitorTrustEnv, self).__init__()

        # Action space: 0 = Answer A, 1 = Answer B, 2 = Answer C
        self.action_space = spaces.Discrete(3)

        # Observation: [confidence, memory_flag]
        # confidence ∈ [0.0, 1.0] (float), memory_flag ∈ [0, 1] (int)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.correct_answer = None
        self.state = None
        self.last_wrong_answer = False
        self.done = False

    def reset(self, seed=None, options=None):
        # Random confidence (player's certainty level)
        confidence = np.round(np.random.uniform(0.0, 1.0), 2)

        # Memory: Did player lie last round? 1 = Yes, 0 = No
        memory_flag = 1.0 if self.last_wrong_answer else 0.0

        self.state = np.array([confidence, memory_flag], dtype=np.float32)

        # Randomly assign correct answer A/B/C
        self.correct_answer = np.random.randint(0, 3)

        self.done = False
        return self.state, {}

    def step(self, action):
        confidence = self.state[0]
        memory_flag = self.state[1]
        reward = 0

        # Define player "knows" if confidence > 0.6
        player_knows = confidence > 0.6

        # Calculate reward
        if action == self.correct_answer:
            reward = 5 if player_knows else 3  # Reward higher if confident and correct
            self.last_wrong_answer = False
        else:
            reward = -5 if player_knows else -3  # Heavier penalty if confidently wrong
            self.last_wrong_answer = True

        self.done = True
        return self.state, reward, self.done, False, {}

    def render(self):
        conf = round(float(self.state[0]), 2)
        memory = "Yes" if self.state[1] == 1.0 else "No"
        print(f"Player confidence: {conf} | Lied in last round? {memory}")

    def close(self):
        pass
