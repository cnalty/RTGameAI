import pygame
import copy

class Player():
    movement_speed = 40

    def __init__(self, image):
        self._positions = [[320, 400], [360, 400], [400, 400]]
        self.xspeed = self.movement_speed
        self.yspeed = 0
        self.image = pygame.image.load(image)





    def update(self, key):
        if key == "U":
            self.yspeed = -1 * self.movement_speed
            self.xspeed = 0
        elif key == "D":
            self.yspeed = self.movement_speed
            self.xspeed = 0
        elif key == "L":
            self.yspeed = 0
            self.xspeed = -1 * self.movement_speed
        elif key == "R":
            self.yspeed = 0
            self.xspeed = self.movement_speed



        for i in range(len(self._positions) - 1):
            self._positions[i] = copy.deepcopy(self._positions[i + 1])

        self._positions[-1][0] += self.xspeed
        self._positions[-1][1] += self.yspeed


    def draw(self, surface):
        for i in range(len(self._positions)):
            surface.blit(self.image, tuple(self._positions[i]))

    def __len__(self):
        return len(self._positions)


    def get_pos(self):
        return self._positions

    def get_head(self):
        return self._positions[-1]

    def grow(self):
        self._positions.insert(0, self._positions[0])