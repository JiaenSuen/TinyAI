import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import imageio
import os

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
 

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render  
        # init display
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        else:
            self.display = pygame.Surface((self.w, self.h))  # 不顯示

        self.clock = pygame.time.Clock()
        self.reset()

        # For Recording
        self.recording = False
        self.frames = []
        self.video_path = None



    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(
            (self.w//BLOCK_SIZE//2)*BLOCK_SIZE,
            (self.h//BLOCK_SIZE//2)*BLOCK_SIZE
        )
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1

        old_head = self.head
        old_distance = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        reward = -0.01
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = -10
            game_over = True
            return reward, game_over, self.score

        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

            if new_distance < old_distance:
                reward += 0.1
            else:
                reward -= 0.1

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

   
 

    def _update_ui(self):
        self.display.fill(BLACK)


        for i, pt in enumerate(self.snake):
            if i == 0:
                color = RED  # Snake Head
            else:
                color = BLUE1  # Snake Body
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

 
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        if self.render:
 
            text = font.render("Score: " + str(self.score), True, WHITE)
            self.display.blit(text, [0, 0])
            pygame.display.flip()

        # filming
        if self.recording:
            frame = pygame.surfarray.array3d(self.display)
            frame = np.transpose(frame, (1, 0, 2))
            self.frames.append(frame)


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)



    

    def start_recording(self, filename="output.mp4"):
        self.recording = True
        self.frames = []
        
        os.makedirs("videos", exist_ok=True)
        self.video_path = os.path.join("videos", filename)
    

    def stop_recording(self):
        if not self.recording or len(self.frames) == 0:
            return
        
        print("Saving video...")
        imageio.mimsave(self.video_path, self.frames, fps=40)
        print(f"Video saved to {self.video_path}")
        
        self.recording = False
        self.frames = []

    