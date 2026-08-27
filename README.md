
## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the SARSA control algorithm using the Gymnasium FrozenLake-v1 environment and learn an optimal or near-optimal action-value function. The agent must learn to select appropriate actions to reach the goal while avoiding the hole states.

## Software Requirements

Python 3.x
Gymnasium
NumPy
Matplotlib
Jupyter Notebook / Google Colab


## Environment Description

FrozenLake-v1 is a discrete reinforcement-learning environment consisting of a 4×4 grid. The agent starts from the initial state and must reach the goal while avoiding holes. The environment has 16 states and 4 possible actions: Left, Down, Right, and Up. A reward of 1 is obtained for reaching the goal, while other transitions normally provide zero reward.

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm

1. Create the custom FrozenLake-v1 environment with start state 5 and goal state 10.
2. Initialize the Q-table with zeros and set α, γ, ε, ε_min, and ε_decay.
3. Reset the environment and obtain the starting state S.
4. Select action A using the epsilon-greedy policy.
5. Execute A and observe reward R and next state S'.
6. Select the next action A' using the current epsilon-greedy policy.
7. Update Q(S,A) using the SARSA update rule.
8. Set S = S' and A = A', and repeat until the episode ends.
9. Decrease epsilon using ε = max(ε_min, ε × ε_decay).
10. Repeat training for all episodes and obtain the final Q-table, policy, and rewards.


## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

# Custom 4x4 map:
# S = Start, G = Goal, H = Hole, F = Frozen surface
# Start state = 5, Goal state = 10

custom_map = [
    "FFFF",
    "FSHF",
    "FHGF",
    "FFFF"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=False
)

n_states = env.observation_space.n
n_actions = env.action_space.n

START_STATE = 5
GOAL_STATE = 10

print("Number of states:", n_states)
print("Number of actions:", n_actions)
print("Starting state:", START_STATE)
print("Goal state:", GOAL_STATE)

assert START_STATE != 0, "Start state must not be 0"
assert GOAL_STATE != n_states - 1, "Goal state must not be the last state"

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))

# Store reward obtained in each episode
episode_rewards = []

# Store epsilon value for each episode
epsilon_history = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """Select an action using an epsilon-greedy policy."""

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    max_actions = np.flatnonzero(
        Q[state] == np.max(Q[state])
    )

    # Randomly choose among equally good actions in case of a tie
    return np.random.choice(max_actions)

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

for episode in range(num_episodes):

    # Reset environment and force the required starting state.
    # FrozenLake reset starts from the S tile in custom_map, which is state 5.
    state, info = env.reset()
    assert state == START_STATE

    # Record the epsilon value used in this episode
    epsilon_history.append(epsilon)

    # Select initial action using the current epsilon
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Select next action using the current epsilon
        if terminated or truncated:
            next_action = None
        else:
            next_action = epsilon_greedy_action(next_state, epsilon)

        # SARSA target
        if terminated or truncated:
            td_target = reward
        else:
            td_target = reward + gamma * Q[next_state, next_action]

        # SARSA update
        Q[state, action] += alpha * (
            td_target - Q[state, action]
        )

        total_reward += reward

        if terminated or truncated:
            break

        # Move to next state-action pair
        state = next_state
        action = next_action

    episode_rewards.append(total_reward)

    # Variable epsilon: gradually reduce exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training completed.")
print("Initial epsilon:", epsilon_history[0])
print("Final epsilon:", epsilon)

# -------------------------------------------------
# Derive State-Value Function and Learned Policy
# -------------------------------------------------

# V(s) = max_a Q(s,a)
state_values = np.max(Q, axis=1)

# Learned policy: best action for every state
learned_policy = np.argmax(Q, axis=1)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

```
---

## Output

Final Q-table:
<img width="568" height="345" alt="image" src="https://github.com/user-attachments/assets/eb9d84cf-f0ea-4546-b5b3-477d71866b64" />



Estimated State-Value Function:
<img width="535" height="116" alt="image" src="https://github.com/user-attachments/assets/ea8b89d8-7271-41bc-b819-329850f1d9f8" />




Learned Policy:

<img width="391" height="115" alt="image" src="https://github.com/user-attachments/assets/d36c3c3b-95de-40ad-a346-4ace10d3ad90" />


Average reward over last 1000 episodes: 

<img width="557" height="52" alt="image" src="https://github.com/user-attachments/assets/b0804eb4-ccae-455a-93ee-4ead5c587f83" />


<img width="937" height="528" alt="image" src="https://github.com/user-attachments/assets/2a34f31b-39c7-4c42-bccd-8b241e588666" />


## Result
```text
The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

```

---

## Inference
```text

The experiment demonstrates that SARSA can learn an effective policy through an epsilon-greedy exploration strategy. During training, the agent initially explores different actions and gradually exploits actions with higher Q-values as epsilon decreases.

```
---

