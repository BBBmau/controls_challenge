import numpy as np

class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel

class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
  
class Agent(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    # Create a default RandomState
    rng = np.random.default_rng()

    # Generate a random float32 between -2 and 2
    random_steering_action = rng.uniform(-2, 2, dtype=np.float32)   
    return random_steering_action
  
# State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
class MauController(BaseController):
    def update(self, target_lataccel, current_lataccel, state):
      print("[UPDATE] Current lataccel: ",current_lataccel, ", Target lataccel: ", target_lataccel)
      if target_lataccel - current_lataccel > 0 :
        return -0.1
      if target_lataccel - current_lataccel < 0 :
        return 0.1
      return 0

CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'mau' : MauController,
  'learning_agent' : Agent,
}
