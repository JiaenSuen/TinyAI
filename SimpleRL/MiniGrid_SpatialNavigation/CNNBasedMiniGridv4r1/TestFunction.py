import torch
from torch.distributions import Categorical, Normal
import os
from datetime import datetime
import imageio  
import numpy as np

def select_action(model, obs, env, action_type="discrete", deterministic=True):
    device = next(model.parameters(), torch.tensor(0.)).device
    obs_tensor = torch.from_numpy(obs).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
    model.eval()
    with torch.no_grad():
        output = model(obs_tensor)

        if action_type == "discrete":
            # Apply action mask like in training
            with torch.no_grad():
                logits, value = model(obs_tensor)

                if action_type == "discrete":
                    mask = torch.from_numpy(env.get_action_mask()).to(device)
                    logits = logits + torch.log(mask + 1e-10)

                    if deterministic:
                        action = logits.argmax(dim=-1).item()
                    else:
                        m = Categorical(logits=logits)
                        action = m.sample().item()

                    return action


        elif action_type == "continuous":
            mean = output[0]
            if mean.shape[-1] % 2 == 0:
                dim = mean.shape[-1] // 2
                mean, log_std = mean[:dim], mean[dim:]
                std = log_std.exp()
            else:
                std = torch.ones_like(mean) * 0.1
            if deterministic:
                return mean.cpu().numpy()
            else:  # train PPO/A2C
                m = Normal(mean, std)
                return m.sample().cpu().numpy()
        else:
            raise ValueError("action_type must be 'discrete' or 'continuous'")

def test_cnn_model(model, env_class, env_name="MiniGrid-DoorKey-16x16-v0", action_type="discrete", test_N=100, save=True,modelname=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"output/Test_{modelname}_{env_name}_{timestamp}"
    video_dir = os.path.join(base_dir, "recordVideo")
    report_path = None   
    video_paths = []

    if save:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        report_path = os.path.join(base_dir, "test_report.txt")

    env_class.reset_test_index()
    
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
        print("Warning: Model has no parameters, using CPU for testing.")
    model.to(device)
    model.eval()   
    
    success_count = 0
    total_steps = 0
    total_reward = 0.0
    
    env = env_class(env_name=env_name, test_N=test_N)  # Pass env_name for 16x16
    
    for i in range(test_N):
        obs, info = env.reset(mode="test")
        done = False
        step_count = 0
        episode_reward = 0.0
        frames = []   
        
        while not done and step_count < 1000:  # Increased max steps for 16x16
            action = select_action(model, obs, env, action_type=action_type)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            frame = env.render()
            frames.append(frame)
            
            episode_reward += reward
            step_count += 1
        
        total_steps += step_count
        total_reward += episode_reward
        
        if terminated:  
            success_count += 1
        
        if save and i < 10:
            video_path = os.path.join(video_dir, f"test_episode_{i+1}.mp4")
            imageio.mimsave(video_path, frames, fps=8)
            video_paths.append(video_path)
            print(f"Video storage : {video_path}")

    env.close()
    
    success_rate = success_count / test_N
    avg_steps = total_steps / test_N
    avg_reward = total_reward / test_N
    
    if save:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"MiniGrid CNN Model Test Report\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Environment: {env.env_name if hasattr(env, 'env_name') else 'Unknown'}\n")
            f.write(f"Test Maps: {test_N}\n\n")
            f.write(f"Success Rate: {success_rate:.4f} ({success_count}/{test_N})\n")
            f.write(f"Average Steps per Episode: {avg_steps:.2f}\n")
            f.write(f"Average Reward: {avg_reward:.4f}\n")
            f.write(f"Total Cost (Steps): {total_steps}\n")
            f.write(f"\nVideo Records (first 5 episodes):\n")
            for vp in video_paths:
                f.write(f"  {os.path.basename(vp)}\n")
        
        print(f"\nTest complete!")
        print(f"Report storage : {report_path}")
        print(f"Videos storage : {video_dir}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Avg Steps: {avg_steps:.1f}")
        print(f"Total Cost: {total_steps}")
    
    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "total_cost": total_steps,
        "report_path": report_path if save else None,
        "video_dir": video_dir if save else None
    }