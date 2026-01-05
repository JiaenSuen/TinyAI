import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper

class MiniGridCNNEnv:
    def __init__(self, env_name="MiniGrid-DoorKey-8x8-v0", test_N=100):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env = ImgObsWrapper(self.env)

        self.env_name = env_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self._test_seeds = list(range(test_N))
        self._test_seed_idx = 0

        # For reward shaping and event detection
        self.prev_dist = None
        self.goal_pos = None
        self.key_pos = None
        self.door_pos = None
        self.picked_key = False  # Flag to give pickup bonus only once per episode
        self.opened_door = False  # Flag for open bonus only once

        self.repeat_turn_penalty = -0.02  
        self.last_action = None
        self.repeat_turn_count = 0


    def reset(self, mode="train"):
        obs, info = self.env.reset(
            seed=None if mode == "train"
            else self._test_seeds[self._test_seed_idx]
        )
        if mode == "test":
            self._test_seed_idx = (self._test_seed_idx + 1) % len(self._test_seeds)
        
        # Reset shaping states and flags
        self.prev_dist = None
        self.picked_key = False
        self.opened_door = False
        self.goal_pos, self.key_pos, self.door_pos = self._find_positions()
        
        # Initialize prev_dist to current subgoal
        self.prev_dist = self._get_current_dist()
        
        self.repeat_turn_count = 0
        self.last_action = None
            

        return obs, info

    def _find_positions(self):
        unwrapped = self.env.unwrapped
        goal = None
        key = None
        door = None
        for i in range(unwrapped.grid.width):
            for j in range(unwrapped.grid.height):
                cell = unwrapped.grid.get(i, j)
                if cell is not None:
                    if cell.type == 'goal':
                        goal = (i, j)
                    elif cell.type == 'key':
                        key = (i, j)
                    elif cell.type == 'door':
                        door = (i, j)
        return goal, key, door

    def _manhattan_dist(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return 0
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_current_subgoal(self):
        unwrapped = self.env.unwrapped
        if self.key_pos is not None:  # DoorKey-like env with key
            if unwrapped.carrying is None or unwrapped.carrying.type != 'key':
                return self.key_pos
            else:
                if self.door_pos is not None:
                    door_cell = unwrapped.grid.get(*self.door_pos)
                    if door_cell and not door_cell.is_open:
                        return self.door_pos
        else:  # MultiRoom-like env, no key
            if self.door_pos is not None:
                door_cell = unwrapped.grid.get(*self.door_pos)
                if door_cell and not door_cell.is_open:
                    return self.door_pos
        return self.goal_pos  # Default to goal if no key/door or door already open
    
    
    def _get_current_dist(self):
        agent_pos = self.env.unwrapped.agent_pos
        subgoal = self._get_current_subgoal()
        return self._manhattan_dist(agent_pos, subgoal)

    def step(self, action):
        unwrapped = self.env.unwrapped
        
        # Record previous states
        prev_carrying = unwrapped.carrying
        prev_front = unwrapped.grid.get(*unwrapped.front_pos)
        prev_door_open = prev_front.is_open if prev_front and prev_front.type == 'door' else False
        
        obs, _, terminated, truncated, info = self.env.step(action)
        
        reward = -0.002 

        prev_pos = tuple(unwrapped.agent_pos)
        curr_pos = tuple(unwrapped.agent_pos)
        moved = (curr_pos != prev_pos)  


        #  Collision penalty 
        front = unwrapped.grid.get(*unwrapped.front_pos)
        if action == 2 and front is not None and not front.can_overlap():
            reward -= 0.05

        #  Pickup key bonus (only once per episode) 
        if unwrapped.carrying and unwrapped.carrying.type == 'key' and (prev_carrying is None or prev_carrying.type != 'key'):
            if not self.picked_key:
                reward += 0.5
                self.picked_key = True

        #  Drop penalty (if dropping key) 
        if action == 4 and prev_carrying and prev_carrying.type == 'key':
            reward -= 0.1  # Discourage dropping key

        #  Open door bonus (only once per episode) 
        curr_front = unwrapped.grid.get(*unwrapped.front_pos)
        if curr_front and curr_front.type == 'door' and curr_front.is_open and not prev_door_open:
            if not self.opened_door:
                reward += 0.5
                self.opened_door = True

        #  Goal reward 
        if terminated:
            reward += 1.0

        #  Subgoal-based distance shaping 
        curr_dist = self._get_current_dist()
        if self.prev_dist is not None:
            shaping = 1 * (self.prev_dist - curr_dist)
            reward += shaping
        self.prev_dist = curr_dist

        # Reset if episode ends
        if terminated or truncated:
            self.prev_dist = None
            self.picked_key = False
            self.opened_door = False

        #  Implement reversal penalty
        if not moved and action in [0, 1]:    # left or right
            if self.last_action in [0, 1]:  self.repeat_turn_count += 1
            else:                           self.repeat_turn_count = 1
        else:self.repeat_turn_count = 0

        if self.repeat_turn_count >= 4:
            reward += self.repeat_turn_penalty    

        self.last_action = action

        return obs, reward, terminated, truncated, info

    def get_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        unwrapped = self.env.unwrapped
        mask[0] = 1  # left
        mask[1] = 1  # right
        front = unwrapped.grid.get(*unwrapped.front_pos)
        if front is None or front.can_overlap():
            mask[2] = 1  # forward
        if front and front.type == "key" and not unwrapped.carrying:
            mask[3] = 1  # pickup
        if unwrapped.carrying:
            mask[4] = 1  # drop
        if front and front.type == "door" and not front.is_open:
            # Allow toggle if door is unlocked OR agent has key (for locked doors)
            if not front.is_locked or (unwrapped.carrying and unwrapped.carrying.type == "key"):
                mask[5] = 1
        mask[6] = 0  # done disabled
        return mask

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @classmethod
    def reset_test_index(cls):
        cls._test_seed_idx = 0