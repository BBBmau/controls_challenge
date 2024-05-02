class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
  
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
}
