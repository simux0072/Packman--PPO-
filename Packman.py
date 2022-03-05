import torch
import cv2
import pygame
from pygame import draw, display, font

pygame.init()

SIZE = 30
DIMENSIONS = [28, 31]

SCREEN = display.set_mode([DIMENSIONS[0] * SIZE, DIMENSIONS[1] * SIZE])
display.set_caption('PPO Packman')
FONT = font.SysFont('aria', 30)

torch.set_printoptions(threshold=10_000)

Direction = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0)
}

class Player():
    def __init__(self):
        self.coordinates = [13, 23] 
        self.direction = 3
        self.points = 0

        self.hunter = False
        self.hunter_timer = 100
        self.env = None

    def _reset(self):
        self.coordinates = [13, 23] 
        self.direction = 3
        self.points = 0

        self.hunter = False
        self.hunter_timer = 100


    def get_env(self, env):
        self.env = env

    def update_direction(self, move):
        self.direction = move
    
    def check_forawrd(self):
        coords = [self.coordinates[1] + Direction[self.direction][1], self.coordinates[0] + Direction[self.direction][0]]
        if coords[0] < 0 or coords[0] >= self.env.wall_map.size(1):
            return True
        if coords[1] < 0 or coords[1] >= self.env.wall_map.size(2):
            return True
        if self.env.wall_map[0][self.coordinates[1] + Direction[self.direction][1]][self.coordinates[0] + Direction[self.direction][0]] == 1:
            return False
        return True

    def update_coordinates(self):
        self.env.character_map[0][self.coordinates[1]][self.coordinates[0]] = 0
        if self.coordinates[1] + Direction[self.direction][1] == 14 and self.coordinates[0] + Direction[self.direction][0] == -1:
            self.coordinates[0] = 27
        elif self.coordinates[1] + Direction[self.direction][1] == 14 and self.coordinates[0] + Direction[self.direction][0] == 28:
            self.coordinates[0] = 0
        else:
            self.coordinates[1] += Direction[self.direction][1]
            self.coordinates[0] += Direction[self.direction][0]
        self.env.character_map[0][self.coordinates[1]][self.coordinates[0]] = 1

    def check_food(self):
        if self.env.point_map[0][self.coordinates[1]][self.coordinates[0]] == 1:
            self.env.point_map[0][self.coordinates[1]][self.coordinates[0]] = 0
            self.points += 10
            return 10
        elif self.env.point_map[0][self.coordinates[1]][self.coordinates[0]] == 2:
            self.hunter = True
            self.hunter_timer = 30
            self.env.point_map[0][self.coordinates[1]][self.coordinates[0]] = 0
            self.points += 50
            return 50
        return 0

    def check_enemy(self):
        if self.env.character_map[0][self.coordinates[1]][self.coordinates[0]] == -1:
            if not self.hunter:
                self.points += -100
                return True, -10
            elif self.hunter:
                self.points += 100
                return False, 100
        return False, 0        

    def update(self, move):
        end = False

        end, reward = self.check_enemy()

        if end:
            return reward, self.points, end

        self.update_direction(move)
        can_move = self.check_forawrd()

        if can_move:
            self.update_coordinates()

            reward = self.check_food()

        end, reward = self.check_enemy()

        if self.hunter:
            if self.hunter_timer == 0:
                self.hunter = False
                self.hunter_timer = 100
            else:
                self.hunter_timer -= 1

        return reward, self.points, end

class ghost():
    def __init__(self, player):
        self.coordinates = [13, 11]
        self.direction = 3
        self.player = player

    def _reset(self):
        self.coordinates = [13, 11] 
        self.direction = 3

    def get_env(self, env):
        self.env = env

    def update_direction(self, move):
        self.direction = move
    
    def check_forawrd(self):
        coords = [self.coordinates[1] + Direction[self.direction][1], self.coordinates[0] + Direction[self.direction][0]]
        if coords[0] < 0 or coords[0] >= self.env.wall_map.size(1):
            return True
        if coords[1] < 0 or coords[1] >= self.env.wall_map.size(2):
            return True
        if self.env.wall_map[0][self.coordinates[1] + Direction[self.direction][1]][self.coordinates[0] + Direction[self.direction][0]] == 1:
            return False
        return True

    def update_coordinates(self):
        self.env.character_map[0][self.coordinates[1]][self.coordinates[0]] = 0
        if self.coordinates[1] + Direction[self.direction][1] == 14 and self.coordinates[0] + Direction[self.direction][0] == -1:
            self.coordinates[0] = 27
        elif self.coordinates[1] + Direction[self.direction][1] == 14 and self.coordinates[0] + Direction[self.direction][0] == 28:
            self.coordinates[0] = 0
        else:
            self.coordinates[1] += Direction[self.direction][1]
            self.coordinates[0] += Direction[self.direction][0]
        self.env.character_map[0][self.coordinates[1]][self.coordinates[0]] = -1

    def check_enemy(self):
        if self.env.character_map[0][self.coordinates[1]][self.coordinates[0]] == 1:
            if self.player.hunter:
                self._reset()
                self.player.points += 100
                return False, -10
            else:
                self.player.points += -100
                return True, 100
        return False, 0
    def update(self, move):
        end = False
        end, reward = self.check_enemy()

        if end:
            return 100, True

        self.update_direction(move)
        can_move = self.check_forawrd()

        if can_move:
            self.update_coordinates()

        end, reward = self.check_enemy()
        return reward, end

class env():
    def __init__(self, player, ghost, device):
        self.player = player
        self.ghost = ghost
        self.device = device
        self.point_map_bak = torch.tensor(cv2.imread('./point_map.png', 0), dtype=torch.float64).unsqueeze(dim=0)
        self.point_map_bak[0][self.player.coordinates[1]][self.player.coordinates[0]] = 0
        self.wall_map = torch.tensor(cv2.imread('./wall_map.png', 0), dtype=torch.float64).unsqueeze(dim=0) / 255
        self.point_map = torch.clone(self.point_map_bak)
        self.character_map = torch.zeros((1, self.point_map.size(1), self.point_map.size(2)), dtype=torch.float64)

    def reset(self):
        self.point_map = torch.clone(self.point_map_bak)
        self.character_map = torch.zeros((1, self.point_map.size(1), self.point_map.size(2)), dtype=torch.float64)

    def draw_state(self):
        pygame.event.get()
        SCREEN.fill((0, 0, 0))

        for coords in torch.nonzero(self.wall_map):
            draw.rect(SCREEN, (59,122,87), pygame.Rect(coords[2].item() * SIZE, coords[1].item() * SIZE, SIZE, SIZE))

        point_coordinates = torch.nonzero(self.point_map)
        for coords in point_coordinates:
            draw.circle(SCREEN, (255, 191, 0), (coords[2].item() * SIZE + SIZE/2, coords[1].item() * SIZE + SIZE/2), SIZE/4)

        draw.rect(SCREEN, (255, 126, 0), pygame.Rect(self.player.coordinates[0] * SIZE, self.player.coordinates[1] * SIZE, SIZE, SIZE))
        draw.rect(SCREEN, (211, 33, 45), pygame.Rect(self.ghost.coordinates[0] * SIZE, self.ghost.coordinates[1] * SIZE, SIZE, SIZE))

        text = FONT.render("Score: " + str(self.player.points), True, (255, 255, 255))
        SCREEN.blit(text, (0, 0))
        display.update()

    def get_state(self):
        return torch.cat((self.point_map, self.wall_map, self.character_map), dim=0).unsqueeze(dim=0), \
                torch.reshape(torch.tensor(self.player.direction, dtype=torch.float64), (1, 1)), \
                torch.reshape(torch.tensor(self.ghost.direction, dtype=torch.float64), (1, 1)), \
                torch.reshape(torch.tensor(self.player.hunter, dtype=torch.float64), (1, 1))