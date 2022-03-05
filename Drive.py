import torch
from os.path import exists

class model_drive():
    def __init__(self, path, model_name):
        self.path = path
        self.model_name = model_name

    def upload(self, player, ghost, optimizer_player, optimizer_ghost, Main_episode, High_score):
        torch.save({
            'player_state_dict': player.state_dict(),
            'ghost_state_dict': ghost.state_dict(),
            'player_optimizer_state_dict': optimizer_player.state_dict(),
            'ghost_optimizer_state_dict': optimizer_ghost.state_dict(),
            'Main_episode': Main_episode,
            'High_score': High_score
            }, self.path + self.model_name)

    def does_exist(self):
        return exists(self.path + self.model_name)

    def download(self):
        return torch.load(self.path + self.model_name)