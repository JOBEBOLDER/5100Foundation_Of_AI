import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from vis_gym import *
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from typing import Tuple


BOLD = '\033[1m'  # ANSI escape sequence for bold text
RESET = '\033[0m' # ANSI escape sequence to reset text formatting

train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	'''
    Compute a unique compact integer ID representing the given observation.

    Encoding scheme:
      - Observation fields:
          * player_health: integer in {0, 1, 2}
          * window: a 3×3 grid of cells, indexed by (dx, dy) with dx, dy ∈ {-1, 0, 1}
          * guard_in_cell: optional identifier of a guard in the player’s cell (e.g. 'G1', 'G2', ...)

      - Each cell contributes a single digit (0–8) to a base-9 number:
          * If the cell is out of bounds → code = 8
          * Otherwise:
                tile_type = 
                    0 → empty
                    1 → trap
                    2 → heal
                    3 → goal
                has_guard = 1 if one or more guards present, else 0
                cell_value = has_guard * 4 + tile_type  # ranges from 0 to 7

        The 9 cell_values (row-major order: top-left → bottom-right) form a 9-digit base-9 integer `window_hash`.

      - The final state_id packs:
            * window_hash  → fine-grained local state
            * guard_index  → identity of guard in player’s cell (0 if none, 1–4 otherwise)
            * player_health → coarse health component

        Specifically:
            WINDOW_SPACE = 9 ** 9
            GUARD_SPACE  = WINDOW_SPACE       # for guard_index (0–4)
            HEALTH_SPACE = GUARD_SPACE * 5    # for health (0–2)

            state_id = (player_health * HEALTH_SPACE) 
                     + (guard_index * GUARD_SPACE) 
                     + window_hash

    Returns:
        int: A unique, compact integer ID suitable for tabular RL (e.g. as a Q-table key).
    '''
	health = int(obs.get('player_health', 0))
	window = obs.get('window', {})

	# Build cell values in a stable order: dx -1..1 (rows), dy -1..1 (cols)
	cell_values = []
	for dx in [-1, 0, 1]:
		for dy in [-1, 0, 1]:
			cell = window.get((dx, dy))
			if cell is None or not cell.get('in_bounds', False):
				cell_values.append(8)
				continue

			# Determine tile type
			if cell.get('is_trap'):
				tile_type = 1
			elif cell.get('is_heal'):
				tile_type = 2
			elif cell.get('is_goal'):
				tile_type = 3
			else:
				tile_type = 0

			has_guard = 1 if cell.get('guards') else 0
			cell_value = has_guard * 4 + tile_type
			cell_values.append(cell_value)

	# Pack into base-9 integer
	window_hash = 0
	base = 1
	for v in cell_values:
		window_hash += v * base
		base *= 9

	# Include guard identity when player is in the center cell.
	# guard_in_cell is a convenience field set by the environment (e.g. 'G1' or None).
	guard_in_cell = obs.get('guard_in_cell')
	if guard_in_cell:
		# map 'G1' -> 1, 'G2' -> 2, etc.
		try:
			guard_index = int(str(guard_in_cell)[-1])
		except Exception:
			guard_index = 0
	else:
		guard_index = 0

	# window_hash uses 9^9 space; reserve an extra multiplier for guard identity (0..4)
	WINDOW_SPACE = 9 ** 9
	GUARD_SPACE = WINDOW_SPACE  # one slot per guard id
	HEALTH_SPACE = GUARD_SPACE * 5  # 5 possible guard_id values (0 = none, 1-4 = guards)

	state_id = int(health) * HEALTH_SPACE + int(guard_index) * GUARD_SPACE + window_hash
	return state_id


def decode_state_id(state_id: int) -> Tuple[int, int, int]:
	"""
	Decode `hash(obs)` output into (player_health, guard_index, center_tile_type).
	- guard_index: 0 means no guard in player's cell, 1..4 correspond to G1..G4.
	- center_tile_type: 0 empty, 1 trap, 2 heal, 3 goal.
	"""
	WINDOW_SPACE = 9 ** 9
	GUARD_SPACE = WINDOW_SPACE
	HEALTH_SPACE = GUARD_SPACE * 5

	health = int(state_id // HEALTH_SPACE)  # 0..2
	tmp = int(state_id - health * HEALTH_SPACE)
	guard_index = int(tmp // GUARD_SPACE)  # 0..4
	window_hash = int(tmp % GUARD_SPACE)

	# Center digit is the 5th digit in the window hash (0-indexed): index 4 -> (dx,dy)=(0,0)
	v_center = int((window_hash // (9 ** 4)) % 9)  # 0..8
	center_tile_type = int(v_center % 4)  # has_guard*4 + tile_type -> tile_type is mod 4
	return health, guard_index, center_tile_type

'''
Complete the function below to do the following:

		1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
			 configuration and taking actions until a terminal state is reached.
		2. Maintain and update Q-values for each state-action pair encountered by the agent in a dictionary (Q-table).
		3. Use epsilon-greedy action selection when choosing actions (explore vs exploit).
		4. Update Q-values using the standard Q-learning update rule.

Important notes about the current environment and state representation

		- The environment is partially observable: observations returned by env.get_observation() include a centered 3x3
			"window" around the player plus the player's health. Each observation is a dict with these relevant keys:
					- 'player_position': (x, y)
					- 'player_health': integer (0=Critical, 1=Injured, 2=Full)
					- 'window': a dict keyed by (dx,dy) offsets in {-1,0,1} x {-1,0,1}. Each entry contains:
								{ 'guards': list or None, 'is_trap': bool, 'is_heal': bool, 'is_goal': bool, 'in_bounds': bool }
					- 'at_trap', 'at_heal', 'at_goal', and 'guard_in_cell' are convenience fields for the center cell.

		- To make a compact and consistent state hash for tabular Q-learning, encode the 3x3 window plus player health into a single integer.
			use the provided hash(obs) function above. Note that the player position is not included in the hash, as it is not needed for local decision-making.

		- Your Q-table should be a dict mapping state_id -> np.array of length env.action_space.n. Initialize arrays to zeros
			when you first encounter a state.

		- The actions available in this environment now include movement, combat, healing and waiting. The action indices are:
					0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: FIGHT, 5: HIDE, 6: HEAL, 7: WAIT

		- Remember to call obs, reward, done, info = env.reset() at the start of each episode.

		- Use a learning-rate schedule per (s,a) pair, i.e. eta = 1/(1 + N(s,a)) where N(s,a) is the
			number of updates applied to that pair so far.

Finally, return the dictionary containing the Q-values (called Q_table).

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}  # state_id -> np.array of Q-values, length env.action_space.n
    N_table = {}  # state_id -> np.array of counts for each action

    # 训练过程统计：学习曲线 & 需要的评估前置信息
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(num_episodes)):
        obs, reward, done, info = env.reset()
        total_reward = 0
        steps_in_episode = 0

        while not done:
            state = hash(obs)

            # 如果从未见过该 state，则初始化
            if state not in Q_table:
                Q_table[state] = np.zeros(env.action_space.n, dtype=float)
                N_table[state] = np.zeros(env.action_space.n, dtype=int)

            # ε-greedy 选动作
            if np.random.rand() < epsilon:
                # 探索：随机动作（作业要求不能硬编码合法动作，这里直接 sample）
                action = env.action_space.sample()
            else:
                # 利用：选择当前 Q 值最大的动作
                action = int(np.argmax(Q_table[state]))

            # 与环境交互
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            steps_in_episode += 1

            next_state = hash(next_obs)

            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(env.action_space.n, dtype=float)
                N_table[next_state] = np.zeros(env.action_space.n, dtype=int)

            # 计算该 (s,a) 对应的学习率 eta = 1 / (1 + N(s,a))
            N_sa = N_table[state][action]
            eta = 1.0 / (1.0 + N_sa)

            # TD 目标：r + gamma * max_a' Q(s', a')
            max_next_Q = np.max(Q_table[next_state])
            td_target = reward + gamma * max_next_Q
            td_error = td_target - Q_table[state][action]

            # Q-learning 更新
            Q_table[state][action] += eta * td_error

            # 更新计数
            N_table[state][action] += 1

            # 准备下一步
            obs = next_obs

        # 一集结束，衰减 epsilon
        epsilon *= decay_rate

        episode_rewards.append(total_reward)
        episode_lengths.append(steps_in_episode)

    # ====== 导出训练曲线 & 5x8 加权 Q 表（供 results.pdf 引用）======
    with open('training_rewards.pickle', 'wb') as f:
        pickle.dump(episode_rewards, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        plt.figure(figsize=(8, 5))
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Q-learning Learning Curve')
        plt.tight_layout()
        plt.savefig('training_rewards.png', dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to save training_rewards.png: {e}")

    # 5x8 表：第 1 行 Heal cell，其余 4 行分别是守卫 G1..G4
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE', 'HEAL', 'WAIT']
    q_5x8 = np.zeros((5, env.action_space.n), dtype=float)
    w_5x8 = np.zeros(5, dtype=float)

    for s_id, q_vals in Q_table.items():
        # 每次进入 state 并执行一个动作 -> 视为一次 encounter
        weight = float(np.sum(N_table[s_id]))
        _, guard_index, center_tile_type = decode_state_id(s_id)

        if guard_index == 0 and center_tile_type == 2:
            row = 0
        elif 1 <= guard_index <= 4:
            row = guard_index
        else:
            continue

        q_5x8[row] += weight * q_vals
        w_5x8[row] += weight

    for row in range(5):
        if w_5x8[row] > 0:
            q_5x8[row] /= w_5x8[row]

    try:
        with open('qtable_5x8.csv', 'w', encoding='utf-8') as f:
            f.write('State,' + ','.join(action_names) + '\n')
            state_rows = ['HEAL', 'G1', 'G2', 'G3', 'G4']
            for i, row_name in enumerate(state_rows):
                vals = [f"{v:.6f}" for v in q_5x8[i].tolist()]
                f.write(row_name + ',' + ','.join(vals) + '\n')
    except Exception as e:
        print(f"[WARN] Failed to save qtable_5x8.csv: {e}")

    return Q_table
'''
Specify number of episodes and decay rate for training and evaluation.
'''

num_episodes = 1000
decay_rate = 0.99

'''
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
'''

if train_flag:
	Q_table = Q_learning(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

	# Save the Q-table dict to a file
	with open('Q_table.pickle', 'wb') as handle:
		pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Evaluation mode: play episodes using the saved Q-table. Useful for debugging/visualization.
Based on autograder logic used to execute actions using uploaded Q-tables.
'''

def softmax(x, temp=1.0):
	e_x = np.exp((x - np.max(x)) / temp)
	return e_x / e_x.sum(axis=0)

if not train_flag:
	
	rewards = []
	episode_lengths = []
	unseen_states = set()
	unseen_actions = 0
	total_actions = 0
	GUI_EVAL_EPISODES = 5
	GUI_DELAY = 0.01

	filename = 'Q_table.pickle'
	if gui_flag:
		print(f"\n{BOLD}Currently loading Q-table from {filename}{RESET}.")
	else:
		input(f"\n{BOLD}Currently loading Q-table from "+filename+f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
	with open(filename, 'rb') as handle:
		Q_table = pickle.load(handle)

	eval_episodes = GUI_EVAL_EPISODES if gui_flag else 10000
	for episode in tqdm(range(eval_episodes)):
		obs, reward, done, info = env.reset()
		total_reward = 0
		steps_in_episode = 0
		
		while not done:
			state = hash(obs)
			if state not in Q_table:
				unseen_states.add(state)
				unseen_actions += 1
				action = env.action_space.sample()
			else:
				# Greedy action for known states during evaluation/GUI.
				# This avoids crashes on unseen states and follows trained policy directly.
				try:
					action = int(np.argmax(Q_table[state]))
				except Exception:
					# Fallback safeguard if the entry is malformed.
					action = env.action_space.sample()
			
			obs, reward, done, info = env.step(action)
			
			total_reward += reward
			steps_in_episode += 1
			total_actions += 1
			if gui_flag:
				refresh(obs, reward, done, info, delay=GUI_DELAY)  # Update the game screen [GUI only]

		#print("Total reward:", total_reward)
		rewards.append(total_reward)
		episode_lengths.append(steps_in_episode)

	avg_reward = sum(rewards) / len(rewards)
	avg_episode_length = sum(episode_lengths) / len(episode_lengths)
	unique_states_in_q = len(Q_table)
	unique_unseen_states = len(unseen_states)
	percent_actions_from_unseen = (100.0 * unseen_actions / total_actions) if total_actions > 0 else 0.0

	print("\n=== Evaluation Summary ===")
	print(f"avg_episode_length: {avg_episode_length}")
	print(f"avg_reward: {avg_reward}")
	print(f"unique_states_in_Q_table: {unique_states_in_q}")
	print(f"unique_unseen_states_during_eval: {unique_unseen_states}")
	print(f"percent_actions_from_unseen_states: {percent_actions_from_unseen:.2f}%")

	# 保存 metrics，方便你直接粘到 results.pdf 里
	with open('evaluation_metrics.txt', 'w', encoding='utf-8') as f:
		f.write(f"decay_rate: {decay_rate}\n")
		f.write(f"num_episodes: {num_episodes}\n")
		f.write(f"avg_episode_length: {avg_episode_length}\n")
		f.write(f"avg_reward: {avg_reward}\n")
		f.write(f"unique_states_in_Q_table: {unique_states_in_q}\n")
		f.write(f"unique_unseen_states_during_eval: {unique_unseen_states}\n")
		f.write(f"percent_actions_from_unseen_states: {percent_actions_from_unseen:.2f}%\n")
