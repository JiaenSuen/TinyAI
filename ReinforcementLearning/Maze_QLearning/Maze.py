import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


class MazeEnv:
    actions = ['up', 'down', 'left', 'right']

  

    def __init__(self, map_file=None):
        """
        map_file: txt file path, each line is a row, e.g.
        0=road, 1=wall, 2=end
        """
        if map_file:
            self.maze = self._load_maze_from_txt(map_file)
        else:
            # default maze
            self.maze = [
                [0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 0, 2],
            ]
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.n_rows = len(self.maze)
        self.n_cols = len(self.maze[0])
        self.actions = ['up', 'down', 'left', 'right']

        self._fig = None
        self._ax  = None


    def _load_maze_from_txt(self, map_file):
        maze = []
        with open(map_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = [int(ch) for ch in line]
                    maze.append(row)
        return maze



    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos
    

    def step(self, action:str):
        r, c = self.agent_pos
        if   action == 'up'   : r -= 1
        elif action == 'down' : r += 1
        elif action == 'left' : c -= 1
        elif action == 'right': c += 1

        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols or self.maze[r][c] == 1:
            reward = -1
            next_state = self.agent_pos 
            done = False
        else:
            next_state = (r, c)
            self.agent_pos = next_state 
            if self.maze[r][c] == 2:
                reward = 10
                done = True
            else:
                reward = -0.1
                done = False

        return next_state, reward, done
    
    def _render_window(self):
        if self._fig is None:
            plt.ion() 
            self._fig, self._ax = plt.subplots()
            self._ax.set_xticks([])
            self._ax.set_yticks([])

        img = np.copy(self.maze)
        r, c = self.agent_pos
        img[r, c] = 3  # agent

        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'gold', 'pink'])
        self._ax.imshow(img, cmap=cmap)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.3)  
        self._ax.cla()  




    # Demo Functions
    def _render_frame(self):
        img = np.copy(self.maze)
        r, c = self.agent_pos
        img[r, c] = 3
        return img

    def play(self, actions, delay=0.3, save_path=None, show=True):
        self.reset()
        frames = []

        plt.ioff()
        fig, ax = plt.subplots()
        ax.set_xticks([]); ax.set_yticks([])
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'gold', 'pink'])


        frames.append([ax.imshow(self._render_frame(), cmap=cmap, animated=True)])
        time.sleep(delay)
        for act in actions:
            next_state, reward, done = self.step(act)
            frames.append([ax.imshow(self._render_frame(), cmap=cmap, animated=True)])
            time.sleep(delay)
            if done:
                break 
        for _ in range(5):
            frames.append([ax.imshow(self._render_frame(), cmap=cmap, animated=True)])
            time.sleep(delay)


        ani = animation.ArtistAnimation(fig, frames, interval=int(delay*1000), blit=True)
        if save_path:
            if save_path.endswith(".gif"):
                ani.save(save_path, writer='pillow', fps=int(1/delay))
            elif save_path.endswith(".mp4"):
                ani.save(save_path, writer='ffmpeg', fps=int(1/delay))
            print(f"-- Animation Saved : {save_path}")
        if show:
            plt.show(block=False)
            plt.pause(1)
        plt.close(fig)
