# Implement Deep Reinforcement Learning Play Greedy Snake Game with DQN & MLP
**After played 200 Games for Training, it could reach 50 score.**
### Agent 

- Model
- Game

Training State :

    - State :  get_state(game)

    - Action  :  get_move(state) :

    model.predict()     ->  Return Action

    - Reward , Game Over  , Score

    - New State  :  get_state(game)

    - Remember

    - Model.train



### Game PyGame

play_step (action)  

    -> Reward , Game Over ,  Score


### Model 

Linear Deep Q Net (DQN)

Model.predict ()


#### **Action**

( 0 , 0 , 1 )  ->  straight

( 0 , 1 , 0 )  ->  right turn

( 1 , 0 , 0 )  -> left turn



#### State : (11 values)

danger straight , danger right , danger left

direct left , direct right , direct up , direct down

food left , food right , food up , food down
