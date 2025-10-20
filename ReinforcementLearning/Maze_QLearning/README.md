# Simple Maze Reinforcement Learning

![Q-learning Maze](./Record/QL_Demo.gif)

### MazeEnv
Maze Enviroment design included 3 essential part and make it easy to use for RL.   
1. Action Input Function
2. Hidden Enviroment Running
3. Display The State Map

This design makes the code very clean and only requires us to consider how to design the model or Q table.  
  
So when design Enviorment , I just need to think the logic and rule about game. Take a input, and code run what happen inside the enviorment. After complete enviorment, can just use a simple interface and easy to connect with human user game or RL agent.

### Q Learning : Tabular
#### Parameters
* learning rate = 0.1   
* discount factor = 0.95  
* epsilon-greedy  = 0.3  
* iteration times = 5000 
* max step for each eppch = 400

#### Q-table
Q Table is a Dictionary Struct which record every actions' Action Value for states of whole maze map.
Calculate them with transfer function formula.
