import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI , Direction , Point 
from model import Linear_QNet , QTrainer
from helper import plot
import sys  
import pygame
from collections import deque
from datetime import datetime



MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001



class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = Linear_QNet(11, 256, 3).to(self.device)
        self.target_model = Linear_QNet(11, 256, 3).to(self.device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(
            model=self.model,
            target_model=self.target_model,
            lr=LR,
            gamma=self.gamma
        )

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

 

    def get_action(self, state, train=True):
        if train:
            self.epsilon = max(0.01, self.epsilon * 0.995)  # Faster decay; adjust if needed
        else:
            self.epsilon = 0.0

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.model(state_tensor)
                move = torch.argmax(q_values, dim=1).item()

        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
         
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory , BATCH_SIZE )
        else:
            mini_sample = self.memory

        states ,actions , rewards ,next_states , dones = zip(*mini_sample)
        self.trainer.train_step(states ,actions , rewards ,next_states , dones )

        #for state,action ,reward,next_state,done in mini_sample:
        #    self.trainer.train_step(state,action ,reward,next_state,done)
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action ,reward,next_state,done)




def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()  
    game = SnakeGameAI(render=False)

    while True:
        if agent.n_games % 20 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        
        best_mean_score = 0
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            print(f"Game {agent.n_games}  Score {score}  Record {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            if score > record:
                record = score
                agent.model.save(file_name="best.pt")  

            if mean_score > best_mean_score:
                best_mean_score = mean_score
                agent.model.save()
            
            plot(plot_scores, plot_mean_scores)


def eval ():   
    
    agent = Agent()
    game = SnakeGameAI()

    import time
    time.sleep(3)
    while True:
        # get old state
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old,train=False)

        reward , done , score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(
            state=state_old,
            action=final_move,
            reward = reward,
            next_state=state_new,
            done=done
        )

        agent.remember(state_old , final_move , reward , state_new , done)
        if done :
            print(f"Game {agent.n_games}  Score {score}  ")
            exit()




def test(model_path="model/best.pt"):
    agent = Agent()  

  
    state_dict = torch.load(model_path, map_location=agent.device, weights_only=True)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    
    game = SnakeGameAI()  
    now = datetime.now()

    year = now.year;month = now.month; day = now.day; hour = now.hour;  minute = now.minute
    game.start_recording(f"test_run_{year}_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}.mp4")
    
    max_iterations = 1000  
    iterations = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("User closed the window.")
                game.stop_recording()
                pygame.quit()
                import sys
                sys.exit()  
        
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, train=False)
        reward, done, score = game.play_step(final_move)
        
        iterations += 1
        if done or iterations > max_iterations:
            print(f"Final Score: {score}")
            game.stop_recording()
            break
    
    pygame.quit()
    

if __name__ == '__main__':
    option = input("Train (1) / Test(2) : ")
    if option == "1" :
        train()
    else:
        test("model/best.pt")