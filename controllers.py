import numpy as np
from tinygrad import Tensor, TinyJit, nn
from typing import Tuple
import tinyphysics

class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError

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
steerVal = np.arange(-2,2,0.01)

# class ActorCriticController:
#   def __init__(self, in_features=tinyphysics.State.count + 2, out_features=np.arange(-2,2,0.01), hidden_state=HIDDEN_UNITS) -> None:
#     # our in_features is our observation space which is our State[v_ego, a_ego, roll_lataccel] + [target_lataccel, current_lataccel]
#     # out features is the steering_action (float32) with a step of 0.01 we get 400 different steering actions
#     self.l1 = nn.Linear(in_features, hidden_state)
#     self.l2 = nn.Linear(hidden_state, out_features)

#     self.c1 = nn.Linear(in_features, hidden_state)
#     self.c2 = nn.Linear(hidden_state, 1)
#     self.opt = nn.optim.Adam(nn.state.get_parameters(self), lr=LEARNING_RATE)

#   def __call__(self, obs:Tensor) -> Tuple[Tensor, Tensor]:
#     x = self.l1(obs).tanh()
#     act = self.l2(x).log_softmax()
#     x = self.c1(obs).relu()
#     return act, self.c2(x)

#   def update(self, target_lataccel, current_lataccel, state):
      
#       @TinyJit
#       def train_step(x:Tensor, selected_action:Tensor, reward:Tensor, old_log_dist:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
#         with Tensor.train():
#           log_dist, value = self(x)
#           action_mask = (selected_action.reshape(-1, 1) == Tensor.arange(log_dist.shape[1]).reshape(1, -1).expand(selected_action.shape[0], -1)).float()

#               # get real advantage using the value function
#           advantage = reward.reshape(-1, 1) - value
#           masked_advantage = action_mask * advantage.detach()

#               # PPO
#           ratios = (log_dist - old_log_dist).exp()
#           unclipped_ratio = masked_advantage * ratios
#           clipped_ratio = masked_advantage * ratios.clip(1-PPO_EPSILON, 1+PPO_EPSILON)
#           action_loss = -unclipped_ratio.minimum(clipped_ratio).sum(-1).mean()

#           entropy_loss = (log_dist.exp() * log_dist).sum(-1).mean()   # this encourages diversity
#           critic_loss = advantage.square().mean()
#           self.opt.zero_grad()
#           (action_loss + entropy_loss*ENTROPY_SCALE + critic_loss).backward()
#           self.opt.step()
#           return action_loss.realize(), entropy_loss.realize(), critic_loss.realize()
        
#       @TinyJit
#       def get_action(obs:Tensor) -> Tensor:
#         # TODO: with no_grad
#         Tensor.no_grad = True
#         ret = self(obs)[0].exp().multinomial().realize()
#         Tensor.no_grad = False
#         return ret

class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel

class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
  
# State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
class RLController(BaseController):
    def __init__(self):
      # and load it back in
      state_dict = tinyphysics.safe_load("controllermodel.safetensors")
      tinyphysics.load_state_dict(tinyphysics.acmodel, state_dict)
    def update(self, target_lataccel, current_lataccel, state):
      stateD = state._asdict()
      obs = [target_lataccel, current_lataccel] + [stateD['v_ego']] + [stateD['roll_lataccel']] + [stateD['a_ego']]
      return steerVal[tinyphysics.acmodel(Tensor(obs))[0].argmax().item()]

CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  "learning_agent": RLController,
}
