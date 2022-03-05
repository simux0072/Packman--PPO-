import Packman

class EnvManager():
    def __init__(self, device):
        self.device = device
        self.done = False
        self.player = Packman.Player()
        self.ghost = Packman.ghost(self.player)
        self.env = Packman.env(self.player, self.ghost, self.device)
        self.get_env()

    def get_env(self):
        self.player.get_env(self.env)
        self.ghost.get_env(self.env)

    def reset(self):
        self.player._reset()
        self.ghost._reset()
        self.env.reset()
        self.done = False
    
    def take_action_player(self, action):
        reward, points, self.done = self.player.update(action)
        self.env.draw_state()
        return reward, points, self.done

    def take_action_ghost(self, action):
        reward, self.done = self.ghost.update(action)
        return reward, self.done

    def get_state(self):
        return self.env.get_state()