import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pt'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

 
 

class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = model.linear1.weight.device  # 自動取得 model 的 device

    def train_step(self, state, action, reward, next_state, done):
        # 轉成 tensor，並指定 device
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        action = torch.tensor(np.array(action), dtype=torch.long, device=self.device)  # 調整為 long，因為 argmax 需要
        reward = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device)
        done = torch.tensor(np.array(done), dtype=torch.bool, device=self.device)

        # 如果是單筆資料，添加批次維度
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # 轉 action index
        action_index = torch.argmax(action, dim=1)

        # Current Q
        q_values = self.model(state)
        q_selected = q_values.gather(1, action_index.unsqueeze(1)).squeeze(1)

        # Double DQN Target
        with torch.no_grad():
            # online network 決定 action
            next_q_online = self.model(next_state)
            next_actions = torch.argmax(next_q_online, dim=1)

            # target network 給 value
            next_q_target = self.target_model(next_state)
            next_q_selected = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target = reward + self.gamma * next_q_selected * (~done)

        # Loss
        loss = F.smooth_l1_loss(q_selected, target)  # 比 MSE 穩定

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()