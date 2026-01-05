import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from TestFunction import test_cnn_model, select_action
from ENV import MiniGridCNNEnv

 
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Set matmul precision for better performance (fix warning)
torch.set_float32_matmul_precision('high')

# Enable CUDNN benchmark for better GPU utilization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

 
def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        init.orthogonal_(layer.weight, gain=gain)
        init.constant_(layer.bias, 0)

# ActorCriticPolicy 
class ActorCriticCNNPolicy(nn.Module):
    def __init__(self, action_dim=7):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=0),
            nn.LayerNorm([16, 6, 6]),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.LayerNorm([32, 5, 5]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.LayerNorm([64, 4, 4]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

        orthogonal_init(self.actor, gain=0.01)
        orthogonal_init(self.critic)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

# PPO class 
class PPO:
    def __init__(self,
                 env_class,
                 policy_class=ActorCriticCNNPolicy,
                 env_name="MiniGrid-DoorKey-8x8-v0",
                 num_envs=64,  
                 device=None,
                 learning_rate=2.5e-4,
                 n_steps=512,  
                 batch_size=128,  
                 n_epochs=10,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 ent_coef=0.02,   
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 load_model=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_class = env_class
        self.policy_class = policy_class
        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = [env_class(env_name=env_name) for _ in range(num_envs)]
        self.action_dim = self.envs[0].action_space.n

        self.policy = policy_class(action_dim=self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.total_timesteps = 0
        self.losses = []


        env_split = self.env_name.split('-')
        env_task = env_split[1] + env_split[2] if len(env_split) > 2 else self.env_name

        if load_model:
            model_path = f"models/PPO_{self.policy_class.__name__}_{env_task}.pth"
            if os.path.exists(model_path):
                self.policy.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=True))
                print(f"Loaded model from {model_path} for continued training.")
            else:
                print(f"No model found at {model_path}, starting training from scratch.")


    def collect_rollouts(self):
        obs = [env.reset(mode="train")[0] for env in self.envs]
        dones = [False] * self.num_envs
        rollout = {
            'obs': [], 'actions': [], 'log_probs': [], 'rewards': [],
            'dones': [], 'values': []
        }

        for _ in range(self.n_steps):
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device) / 255.0
            with torch.no_grad():
                logits, values = self.policy(obs_tensor)

                mask = torch.tensor(
                    np.stack([env.get_action_mask() for env in self.envs]),
                    device=self.device
                )

                logits = logits + torch.log(mask + 1e-10)
                dists = torch.distributions.Categorical(logits=logits)

                actions = dists.sample()
                log_probs = dists.log_prob(actions)

            actions_cpu = actions.cpu().numpy()
            log_probs_cpu = log_probs.cpu().numpy()
            values_cpu = values.cpu().numpy()

            next_obs = []
            rewards = []
            new_dones = []
            for i in range(self.num_envs):
                if dones[i]:
                    obs[i], _ = self.envs[i].reset(mode="train")
                n_obs, reward, terminated, truncated, _ = self.envs[i].step(actions_cpu[i])
                done = terminated or truncated
                next_obs.append(n_obs)
                rewards.append(reward)
                new_dones.append(done)

            rollout['obs'].extend(obs)
            rollout['actions'].extend(actions_cpu)
            rollout['log_probs'].extend(log_probs_cpu)
            rollout['rewards'].extend(rewards)
            rollout['dones'].extend(new_dones)
            rollout['values'].extend(values_cpu)

            obs = next_obs
            dones = new_dones

        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device) / 255.0
        with torch.no_grad():
            _, next_values = self.policy(obs_tensor)
        next_values = next_values.cpu().numpy()
        next_dones = np.array(dones)

        advantages = np.zeros(len(rollout['rewards']))
        returns = np.zeros(len(rollout['rewards']))
        gae = np.zeros(self.num_envs)
        for t in reversed(range(self.n_steps)):
            for env_i in range(self.num_envs):
                idx = t * self.num_envs + env_i
                if t == self.n_steps - 1:
                    next_value = next_values[env_i] * (1 - next_dones[env_i])
                else:
                    next_idx = (t + 1) * self.num_envs + env_i
                    next_value = rollout['values'][next_idx] * (1 - rollout['dones'][next_idx])
                delta = rollout['rewards'][idx] + self.gamma * next_value - rollout['values'][idx]
                gae[env_i] = delta + self.gamma * self.gae_lambda * (1 - rollout['dones'][idx]) * gae[env_i]
                advantages[idx] = gae[env_i]
                returns[idx] = advantages[idx] + rollout['values'][idx]

        rollout['advantages'] = advantages.tolist()
        rollout['returns'] = returns.tolist()

        self.total_timesteps += self.n_steps * self.num_envs
        return rollout

    def update(self, rollout):
        obs = torch.tensor(np.array(rollout['obs']), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device) / 255.0
        actions = torch.tensor(rollout['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(rollout['log_probs'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(rollout['returns'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(rollout['advantages'], dtype=torch.float32).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        num_batches = 0

        indices = np.arange(len(obs))
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                logits, values = self.policy(obs[batch_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns[batch_idx])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                total_loss += loss.item()
                num_batches += 1

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def learn(self, total_timesteps=10_000_000, test_N=100, log_dir=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_task = self.env_name.split('-')[1] + self.env_name.split('-')[2]
        log_dir = f"output/TrainPPO_{self.policy_class.__name__}_{env_task}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training_log.txt")

        with open(log_path, "w") as f:
            f.write(f"=== {env_task} PPO Training Log ===\n")
            f.write(f"Start time: {datetime.now()}\n")
            f.write(f"Device: {self.device}, Num Envs: {self.num_envs}\n")
            f.write(f"Total Timesteps: {total_timesteps}\n")
            f.write(f"Hyperparams: LR={self.optimizer.param_groups[0]['lr']}, Clip={self.clip_range}, Ent={self.ent_coef}\n")
            f.write("\nReward Design: Subgoal shaping, bonuses once/episode, drop penalty\n")
            f.write("Timesteps | Avg Loss\n")

        print(f"Training started on {self.device} with {self.num_envs} envs")

        n_updates = 0
        initial_lr = self.optimizer.param_groups[0]['lr']
        total_updates = total_timesteps // (self.n_steps * self.num_envs)
        with tqdm(total=total_updates, desc="Training") as pbar:
            while self.total_timesteps < total_timesteps:
                rollout = self.collect_rollouts()
                avg_loss = self.update(rollout)
                n_updates += 1

                progress = self.total_timesteps / total_timesteps
                self.optimizer.param_groups[0]['lr'] = initial_lr * (1 - progress)

                log_line = f"{self.total_timesteps:8d} | {avg_loss:.4f}"
                with open(log_path, "a") as f:
                    f.write(log_line + "\n")

                pbar.update(1)

        # Save model
        env_task = self.env_name.split('-')[1] + self.env_name.split('-')[2]
        model_path = f"models/PPO_{self.policy_class.__name__}_{env_task}.pth"
        torch.save(self.policy.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        print("\nFinal evaluation on 100 test maps...")
        MiniGridCNNEnv.reset_test_index()
        final_result = test_cnn_model(
            model=self.policy,
            env_class=MiniGridCNNEnv,
            env_name=self.env_name,  # Pass env_name for correct testing
            action_type="discrete",
            test_N=test_N,
            save=True,
            modelname=self.policy_class.__name__
        )

        with open(log_path, "a") as f:
            f.write("\n=== FINAL TEST RESULT ===\n")
            f.write(f"Success Rate: {final_result['success_rate']:.2%}\n")
            f.write(f"Average Steps: {final_result['avg_steps']:.1f}\n")
            f.write(f"Average Reward: {final_result['avg_reward']:.4f}\n")
            f.write(f"Total Cost (Steps): {final_result['total_cost']}\n")
            f.write(f"Report & Videos: {final_result['report_path']}\n")

        print(f"\nTest complete!")
        print(f"Report storage : {final_result['report_path']}")
        print(f"Videos storage : {final_result['video_dir']}")
        print(f"Success Rate: {final_result['success_rate']:.2%}")
        print(f"Avg Steps: {final_result['avg_steps']:.1f}")
        print(f"Total Cost: {final_result['total_cost']}")
        for env in self.envs:
            env.close()

# modified_select_action 
def modified_select_action(model, obs, env, action_type="discrete", deterministic=True):
    device = next(model.parameters(), torch.tensor(0.)).device
    obs_tensor = torch.from_numpy(obs).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
    model.eval()
    with torch.no_grad():
        output, _ = model(obs_tensor)  # Unpack tuple (logits, value)
        mask = torch.from_numpy(env.get_action_mask()).to(device)
        logits = output + torch.log(mask + 1e-10)
        if action_type == "discrete":
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                m = torch.distributions.Categorical(logits=logits)
                action = m.sample().item()
            return action

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    import TestFunction
    TestFunction.select_action = modified_select_action

    np.random.seed(0)
    torch.manual_seed(0)

 
    
    env_name = "MiniGrid-MultiRoom-N4-S5-v0"  

    ppo = PPO(env_class=MiniGridCNNEnv, policy_class=ActorCriticCNNPolicy, env_name=env_name, device=device, num_envs=128 ,load_model=False)
    ppo.learn(total_timesteps=1_000_000, test_N=100)