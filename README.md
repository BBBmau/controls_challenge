# From mau

This repo is where I attempt the controls challenge by using Reinforced Learning for teaching a controller to use the best steering_action in regards to the state of the controller [`vEgo`, `aEgo`, `rollAccel`, `current_lataccel`].

The idea I have is to have the controller (in RL terms the `agent`) be rewarded on every epsiode where it successfully uses the appropriate action that leads to a difference between `current_lataccel` and `target_lataccel` be close to 0 while also maintaining the smallest jerk _cost as possible.

For the agent this will take some generations to produce since I'm sure initially it will need to understand the actions as well as the patterns from agent states to be able to produce great results.

As this is my first time ever looking into RL, I'll also include Resources that I've used to help me in this challenge.

As of now I have full understanding of the problem and what the RL agent will need in order to start learning. The next step is to actually train the model which will be a first for me.

## Training a dumby RL model

When first looking to training an agent for this problem, the goto seemed to be tensorflow due to the amount of documentation that existed for it, however I soon realized how difficult to get something up for something that wasn't entirely massive (by this I mean not a lot of inputs and outputs for the neural network.)

From this I then looked into tinygrad as an alternative, I had this in mind at first but felt that it would be too complicated to follow due to docs but the selling point was how tinygrad is meant for training models very quickly without the need of large code blocks such as what's found when using tensorflow/pytorch.

I was able to use [examples/beautiful_cartpole.py](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_cartpole.py) as reference when working on this and will also be using [examples/rl/lightupbutton.py](https://github.com/tinygrad/tinygrad/blob/master/examples/rl/lightupbutton.py) to improve how rewards are handled.

As of `5/13/24` I have a model that can control the steering, not the best controller but it is a start to how RL can be used to tackle the controls_challenge.

## Resource Links

* [For understanding RL concepts](https://www.youtube.com/watch?v=TCCjZe0y4Qc&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&index=2)
* [For understanding RL Implementation in Python](https://www.youtube.com/watch?v=Mut_u40Sqz4&t=2853s)

# Comma Controls Challenge!
![Car](./imgs/car.jpg)

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.


## Geting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual routes with actual car and road states.

```
# download necessary dataset (~0.6G)
bash ./download_dataset.sh

# install required packages
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller simple
```

There are some other scripts to help you get aggregate metrics: 
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller simple

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller simple --baseline_controller open

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. It's inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`) and a steer input (`steer_action`) and predicts the resultant lateral acceleration fo the car.


## Controllers
Your controller should implement an [update function](https://github.com/commaai/controls_challenge/blob/1a25ee200f5466cb7dc1ab0bf6b7d0c67a2481db/controllers.py#L2) that returns the `steer_action`. This controller is then run in-loop, in the simulator to autoregressively predict the car's response.


## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(actual\\_lat\\_accel - target\\_lat\\_accel)^2}{steps} * 100$

- `jerk_cost`: $\dfrac{\Sigma((actual\\_lat\\_accel\_t - actual\\_lat\\_accel\_{t-1}) / \Delta t)^2}{steps - 1} * 100$

It is important to minimize both costs. `total_cost`: $(lataccel\\_cost *5) + jerk\\_cost$

## Submission
Run the following command, and send us a link to your fork of this repo, and the `report.html` this script generates.
```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller simple
```

## Work at comma
Like this sort of stuff? You might want to work at comma!
https://www.comma.ai/jobs
