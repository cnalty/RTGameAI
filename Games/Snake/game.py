import pygame
from player import Player
from random import randint
import time

class Snake():
    windowWidth = 800
    windowHeight = 800
    player = 0

    def __init__(self, input_source, move_keys, game_speed=0.25):
        pygame.init()
        self.started = True
        self.surface = pygame.display.set_mode((self.windowWidth, self.windowHeight))
        pygame.display.set_caption("Snake")
        self.started = False
        self.running = True
        self.player = Player("snake.png")
        self.apple = Apple("food.png", self.new_pos())
        self.inputs = input_source
        self.closeness = 1000
        self.moves = 0
        self.move_keys = move_keys
        self.game_speed = game_speed

    def ate_apple(self):
        if self.apple.get_pos() == tuple(self.player.get_head()):
            self.apple = Apple("food.png", self.new_pos())
            return True
        else:
            return False

    def close_apple(self):
        a_pos = self.apple.get_pos()
        p_pos = self.player.get_head()

        return ((a_pos[0] - p_pos[0])**2 + (a_pos[1] - p_pos[1])**2)**(0.5)


    def new_pos(self):
        overlap = True
        while overlap:
            x = randint(1, self.windowWidth / 40 - 1) * 40
            y = randint(1, self.windowWidth / 40 - 1) * 40
            overlap = False
            for pos in self.player.get_pos():
                if [x, y] == pos:
                    overlap = True
                    break
        return (x, y)

    def is_dead(self):
        pos = self.player.get_pos()
        head = self.player.get_head()
        for i in range(len(pos) - 1):
            if pos[i] == head:
                return True
            if pos[i][0] <= 0 or pos[i][0] >= self.windowWidth or \
                pos[i][1] <= 0 or pos[i][1] >= self.windowHeight:
                    return True
        return False

    def get_image(self):
        pygame.surfarray.array3d(self.surface)

    def game_loop(self):
        while(self.running):
            pygame.event.pump()
            input = self.inputs(self.get_image())

            # K_LEFT = 276
            # K_RIGHT = 275
            # K_UP = 273
            # K_DOWN = 274
            # K_ESCAPE = 27
            has_input = False
            for key in self.move_keys.keys():
                if input[key]:
                    has_input = True
                    if self.move_keys[key] == "E":
                        self.running = False
                    else:
                        self.player.update(self.move_keys[key])
                    break
            if not has_input:
                self.player.update("S")

            self.closeness = min(self.close_apple(), self.closeness)

            if self.ate_apple():
                self.player.grow()

            self.surface.fill((0,0,0))
            self.player.draw(self.surface)
            self.apple.draw(self.surface)
            pygame.display.flip()
            if self.is_dead():
                self.running = False
            self.moves += 1
            pygame.display.update()
            time.sleep(self.game_speed)

        pygame.quit()
        return len(self.player), self.closeness, self.moves








class Apple():
    def __init__(self, image, pos):
        # image is a path to a file
        # pos is a tuple (x, y)
        self.image = pygame.image.load(image)
        self.position = pos

    def draw(self, surface):
        surface.blit(self.image, self.position)

    def get_pos(self):
        return self.position