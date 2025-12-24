# 產生一個複雜且可通行的 20x20 01 地圖，右下角為終點 2，並存成 txt 檔案供下載。
import random
from pathlib import Path

WIDTH = 20
HEIGHT = 20
p_wall = 0.45  # 初始牆的機率

def generate_map(width, height, p_wall):
 
    grid = [[ '1' if random.random() < p_wall else '0' for _ in range(width)] for _ in range(height)]
 
    grid[0][0] = '0'
    grid[height-1][width-1] = '0'
    # 再用隨機路徑保證從 (0,0) 到 (w-1,h-1) 有一條可通行的路
    x, y = 0, 0
    path = [(x,y)]
    while (x, y) != (width-1, height-1):
 
        dx = 1 if random.random() < 0.6 and x < width-1 else 0
        dy = 1 if dx == 0 and y < height-1 and random.random() < 0.7 else 0
 
        if dx == 0 and dy == 0:
            choices = []
            if x < width-1: choices.append((x+1, y))
            if y < height-1: choices.append((x, y+1))
            if x > 0: choices.append((x-1, y))
            if y > 0: choices.append((x, y-1))
            nx, ny = random.choice(choices)
        else:
            nx, ny = x+dx, y+dy
        x, y = nx, ny
        grid[y][x] = '0'  
        path.append((x,y))
    return grid, path

grid, path = generate_map(WIDTH, HEIGHT, p_wall)

grid[HEIGHT-1][WIDTH-1] = '2'


out_path = Path('map20x20.txt')
with out_path.open('w', encoding='utf-8') as f:
    for row in grid:
        f.write(''.join(row) + '\n')


map_str_preview = '\n'.join(''.join(row) for row in grid)
print("Generated map20x20.txt\n")
print(map_str_preview)

