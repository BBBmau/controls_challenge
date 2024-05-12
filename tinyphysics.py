import argparse
import numpy as np
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal
from typing import Tuple
from tinygrad import Tensor, TinyJit, nn
import time
from collections import namedtuple
from hashlib import md5
from pathlib import Path
from typing import List, Union, Tuple
from tqdm import tqdm

from controllers import BaseController, CONTROLLERS

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

ACC_G = 9.81
CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 5.0

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])

# hyperparameters that are taken from `beautiful_cartpole.py`
# https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_cartpole.py
BATCH_SIZE = 256
ENTROPY_SCALE = 0.0005
REPLAY_BUFFER_SIZE = 2000
PPO_EPSILON = 0.2
HIDDEN_UNITS = 32
LEARNING_RATE = 1e-2
TRAIN_STEPS = 5
EPISODES = 40
DISCOUNT_FACTOR = 0.99
LEARNING = False
st, steps = time.perf_counter(), 0
Xn, An, Rn = [], [], []
reward = []

class ActorCritic:
  def __init__(self, in_features=State.count + 2, out_features=np.arange(-2,2,0.01), hidden_state=HIDDEN_UNITS) -> None:
    # our in_features is our observation space which is our State[v_ego, a_ego, roll_lataccel] + [target_lataccel, current_lataccel]
    # out features is the steering_action (float32) with a step of 0.01 we get 400 different steering actions
    self.l1 = nn.Linear(in_features, hidden_state)
    self.l2 = nn.Linear(hidden_state, out_features)

    self.c1 = nn.Linear(in_features, hidden_state)
    self.c2 = nn.Linear(hidden_state, 1)
    self.opt = nn.optim.Adam(nn.state.get_parameters(self), lr=LEARNING_RATE)

  def __call__(self, obs:Tensor) -> Tuple[Tensor, Tensor]:
    x = self.l1(obs).tanh()
    act = self.l2(x).log_softmax()
    x = self.c1(obs).relu()
    return act, self.c2(x)

class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    return self.bins[token]

  def clip(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
  def __init__(self, model_path: str, debug: bool) -> None:
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if 'CUDAExecutionProvider' in ort.get_available_providers():
      if debug:
        print("ONNX Runtime is using GPU")
      provider = ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'})
    else:
      if debug:
        print("ONNX Runtime is using CPU")
      provider = 'CPUExecutionProvider'

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, temperature=1.) -> int:
    res = self.ort_session.run(None, input_data)[0]
    probs = self.softmax(res / temperature, axis=-1)
    # we only care about the last timestep (batch size is just 1)
    assert probs.shape[0] == 1
    assert probs.shape[2] == VOCAB_SIZE
    sample = np.random.choice(probs.shape[2], p=probs[0, -1])
    return sample

  def get_current_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
    tokenized_actions = self.tokenizer.encode(past_preds)
    raw_states = [list(x) for x in sim_states]
    states = np.column_stack([actions, raw_states])
    input_data = {
      'states': np.expand_dims(states, axis=0).astype(np.float32),
      'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    return self.tokenizer.decode(self.predict(input_data, temperature=1.))


class TinyPhysicsSimulator:
  def __init__(self, model: TinyPhysicsModel, data_path: str, controller: BaseController, debug: bool = False) -> None:
    self.data_path = data_path
    self.sim_model = model
    self.data = self.get_data(data_path)
    self.controller = controller
    self.debug = debug
    self.reset()

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    self.state_history = [self.get_state_target(i)[0] for i in range(self.step_idx)]
    self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
    self.current_lataccel_history = [self.get_state_target(i)[1] for i in range(self.step_idx)]
    self.target_lataccel_history = [self.get_state_target(i)[1] for i in range(self.step_idx)]
    self.current_lataccel = self.current_lataccel_history[-1]
    seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)

  def get_data(self, data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': df['steerCommand'].values
    })
    return processed_df

  def sim_step(self, step_idx: int) -> None:
    pred = self.sim_model.get_current_lataccel(
      sim_states=self.state_history[-CONTEXT_LENGTH:],
      actions=self.action_history[-CONTEXT_LENGTH:],
      past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
    )
    pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = pred
    else:
      self.current_lataccel = self.get_state_target(step_idx)[1]
             #(difference in predicted lataccel, difference in before/after lataccel [jerk])
             # the lower the reward the better
    reward.append(abs(self.current_lataccel - self.get_state_target(step_idx)[1]) + abs(self.current_lataccel - self.current_lataccel_history[-1]))
    self.current_lataccel_history.append(self.current_lataccel)

  def control_step(self, step_idx: int) -> None:
    if step_idx >= CONTROL_START_IDX:
      if LEARNING:
        obs = [self.target_lataccel_history[step_idx], self.current_lataccel] + self.state_history[step_idx]
        action = get_action(Tensor(obs).item())
        Xn.append(np.copy(obs))
        An.append(action)
      else: 
        action = self.controller.update(self.target_lataccel_history[step_idx], self.current_lataccel, self.state_history[step_idx])
    else:
      action = self.data['steer_command'].values[step_idx]
    action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
    self.action_history.append(action)

  def get_state_target(self, step_idx: int) -> Tuple[State, float]:
    state = self.data.iloc[step_idx]
    return State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']), state['target_lataccel']

  def step(self) -> None:
    state, target = self.get_state_target(self.step_idx)
    self.state_history.append(state)
    self.target_lataccel_history.append(target)
    self.control_step(self.step_idx) # function where the steering_action is computed based on state
    self.sim_step(self.step_idx) # applies the steering_action which results in going into a new state (step) with new lataccel
    self.step_idx += 1

  def plot_data(self, ax, lines, axis_labels, title) -> None:
    ax.clear()
    for line, label in lines:
      ax.plot(line, label=label)
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
    ax.set_title(f"{title} | Step: {self.step_idx}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

  def compute_cost(self) -> dict:
    target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:]
    pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:]
    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

  def rollout(self) -> float:
    if self.debug:
      plt.ion()
      fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)
    get_action.reset()
    for _ in range(CONTEXT_LENGTH, len(self.data)):
      self.step()
      if self.debug and self.step_idx % 10 == 0:
        print(f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}, Target lataccel: {self.target_lataccel_history[-1]:>6.2f}")
        self.plot_data(ax[0], [(self.target_lataccel_history, 'Target lataccel'), (self.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
        self.plot_data(ax[1], [(self.action_history, 'Action')], ['Step', 'Action'], 'Action')
        self.plot_data(ax[2], [(np.array(self.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
        self.plot_data(ax[3], [(np.array(self.state_history)[:, 1], 'v_ego')], ['Step', 'v_ego'], 'v_ego')
        plt.pause(0.01)
    if LEARNING:
      discounts = np.power(DISCOUNT_FACTOR, np.arange(len(reward)))
      Rn += [np.sum(reward[i:] * discounts[:len(reward)-i]) for i in range(len(reward))]
      Xn, An, Rn = Xn[-REPLAY_BUFFER_SIZE:], An[-REPLAY_BUFFER_SIZE:], Rn[-REPLAY_BUFFER_SIZE:]
      X, A, R = Tensor(Xn), Tensor(An), Tensor(Rn)
      old_log_dist = model(X)[0].detach()   # TODO: could save these instead of recomputing
      for i in range(TRAIN_STEPS):
        samples = Tensor.randint(BATCH_SIZE, high=X.shape[0]).realize()  # TODO: remove the need for this
        # TODO: is this recompiling based on the shape?
        action_loss, entropy_loss, critic_loss = train_step(X[samples], A[samples], R[samples], old_log_dist[samples])
      t.set_description(f"sz: {len(Xn):5d} steps/s: {steps/(time.perf_counter()-st):7.2f} action_loss: {action_loss.item():7.3f} entropy_loss: {entropy_loss.item():7.3f} critic_loss: {critic_loss.item():8.3f} reward: {sum(reward):6.2f}")

    if self.debug:
      plt.ioff()
      plt.show()
    return self.compute_cost()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--learn", type=bool, default=False)
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--debug", action='store_true')
  parser.add_argument("--controller", default='simple', choices=CONTROLLERS.keys())
  args = parser.parse_args()

  tinyphysicsmodel = TinyPhysicsModel(args.model_path, debug=args.debug)
  acmodel = ActorCritic() # default values are set for this environment
  opt = nn.optim.Adam(nn.state.get_parameters(acmodel), lr=LEARNING_RATE)

  @TinyJit
  def train_step(x:Tensor, selected_action:Tensor, reward:Tensor, old_log_dist:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    with Tensor.train():
      log_dist, value = acmodel(x)
      action_mask = (selected_action.reshape(-1, 1) == Tensor.arange(log_dist.shape[1]).reshape(1, -1).expand(selected_action.shape[0], -1)).float()
          # get real advantage using the value function
      advantage = reward.reshape(-1, 1) - value
      masked_advantage = action_mask * advantage.detach()
          # PPO
      ratios = (log_dist - old_log_dist).exp()
      unclipped_ratio = masked_advantage * ratios
      clipped_ratio = masked_advantage * ratios.clip(1-PPO_EPSILON, 1+PPO_EPSILON)
      action_loss = -unclipped_ratio.minimum(clipped_ratio).sum(-1).mean()
      entropy_loss = (log_dist.exp() * log_dist).sum(-1).mean()   # this encourages diversity
      critic_loss = advantage.square().mean()
      opt.zero_grad()
      (action_loss + entropy_loss*ENTROPY_SCALE + critic_loss).backward()
      opt.step()
      return action_loss.realize(), entropy_loss.realize(), critic_loss.realize()
        
  @TinyJit
  def get_action(obs:Tensor) -> Tensor:
    # TODO: with no_grad
    Tensor.no_grad = True
    ret = acmodel(obs)[0].exp().multinomial().realize()
    Tensor.no_grad = False
    return ret
  LEARNING = args.learning
  data_path = Path(args.data_path)
  if LEARNING:
      files = sorted(data_path.iterdir())[:args.num_segs]
      for data_file in tqdm(files, total=len(files)): # every file is an episode
        controller = CONTROLLERS["learning_agent"]()
        sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_file), controller=controller, debug=args.debug)
        _ = sim.rollout()
  else:
    if data_path.is_file():
      controller = CONTROLLERS[args.controller]()
      sim = TinyPhysicsSimulator(tinyphysicsmodel, args.data_path, controller=controller, debug=args.debug)
      costs = sim.rollout()
      print(f"\nAverage lataccel_cost: {costs['lataccel_cost']:>6.4}, average jerk_cost: {costs['jerk_cost']:>6.4}, average total_cost: {costs['total_cost']:>6.4}")
    elif data_path.is_dir():
      costs = []
      files = sorted(data_path.iterdir())[:args.num_segs]
      for data_file in tqdm(files, total=len(files)):
        controller = CONTROLLERS[args.controller]()
        sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_file), controller=controller, debug=args.debug)
        cost = sim.rollout()
        costs.append(cost)
      costs_df = pd.DataFrame(costs)
      print(f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, average total_cost: {np.mean(costs_df['total_cost']):>6.4}")
      for cost in costs_df.columns:
        plt.hist(costs_df[cost], bins=np.arange(0, 1000, 10), label=cost, alpha=0.5)
      plt.xlabel('costs')
      plt.ylabel('Frequency')
      plt.title('costs Distribution')
      plt.legend()
      plt.show()
