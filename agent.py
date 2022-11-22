import torch
import random
import numpy as np
from collections import deque
from snake import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEM = 100_000
BATCH_SIZE = 10_000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEM)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        
        head = game.snake[0]
        dir = game.direction

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = dir == Direction.LEFT
        dir_r = dir == Direction.RIGHT
        dir_u = dir == Direction.UP
        dir_d = dir == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            dir_u,
            dir_r,
            dir_d,
            dir_l,

            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]

        return np.array(state, dtype=np.uint8)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_mem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_mem(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):

        self.epsilon = 100 - self.number_of_games

        action = [0, 0, 0]

        if(random.randint(0, 250) < self.epsilon):
            move = random.randint(0,2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        
        action[move] = 1
        
        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    
    while True:

        state = agent.get_state(game)
        action = agent.get_action(state)

        reward, game_over, score = game.play_frame(action, agent.number_of_games)
        next_state = agent.get_state(game)

        agent.train_short_mem(state, action, reward, next_state, game_over)
        agent.remember(state, action, reward, next_state, game_over)

        if game_over:
            
            game.reset()
            agent.number_of_games += 1
            agent.train_long_mem()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.number_of_games, 'Score', score, 'Record', record)
            plot_scores.append(score)
            total_score += score
            mean_score_last_10 = np.sum(plot_scores[-10:]) / 10
            plot_mean_scores.append(mean_score_last_10)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()