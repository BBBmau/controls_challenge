# import tinyphysics
import numpy as np
import tinyphysics
from collections import namedtuple
from tf_agents.environments import py_environment
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

# Create a range of values from -1 to 1 with a step size of 0.01
action_range = np.arange(-1, 1.01, 0.01)
State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
ACC_G = 9.81
CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 5.0

class TinySimEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        self._stepIndx = 0
        self._state = tinyphysics.State
        # sim setup
        self.model = tinyphysics.TinyPhysicsModel(tinyphysicsmodel = tinyphysics.TinyPhysicsModel("./models/tinyphysics.onnx"))
        self.sim = tinyphysics.TinyPhysicsSimulator(self.model, "./data", controller=tinyphysics.CONTROLLERS["learning_agent"])

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), 
            dtype=np.float32, 
            minimum=STEER_RANGE[0], 
            maximum=STEER_RANGE[1], 
            name='steer_action')
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), 
            dtype=np.float32, 
            minimum=[LATACCEL_RANGE[0], 0, -1, LATACCEL_RANGE[0]], 
            maximum=[LATACCEL_RANGE[1], 30, 1, LATACCEL_RANGE[1]]) # our state array + current_lataccel
        
        def _reset(self):
            self._step = 0
            self._episode_ended = False

        def _step(self, action):
            if self._episode_ended:
                return self._reset()
            self.sim.step()
            reward = 0
            lataccel_diff = abs(self.sim.current_lataccel_history[self._stepIndx] - self.sim.target_lataccel_history[self._stepIndx])
            if lataccel_diff <= 0.1:
                reward = -0.1
            else:
                reward = 2
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward)




if __name__ == "__main__":
    tinysim_env = TinySimEnvironment()