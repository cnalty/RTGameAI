import pygame
from Games.Snake.player import Player
from random import randint
import time
import copy

class Snake():
    windowWidth = 800
    windowHeight = 800
    player = 0

    def __init__(self, input_source, move_keys, snake_im, food_im, game_speed=0.25):
        pygame.init()
        self.started = True
        self.surface = pygame.display.set_mode((self.windowWidth, self.windowHeight))
        #pygame.display.set_caption("Snake")
        self.started = False
        self.running = True
        self.player = Player(snake_im)
        self.apple = Apple(food_im, self.new_pos())
        self.inputs = input_source
        self.closeness = 1000
        self.moves_away = 0
        self.moves = 0
        self.hunger = 0
        self.move_keys = move_keys
        self.game_speed = game_speed
        self.food_im = food_im

    def ate_apple(self):
        if self.apple.get_pos() == tuple(self.player.get_head()):
            self.apple = Apple(self.food_im, self.new_pos())
            return True
        else:
            return False

    def close_apple(self):
        a_pos = self.apple.get_pos()
        p_pos = self.player.get_head()

        return ((a_pos[0] - p_pos[0])**2 + (a_pos[1] - p_pos[1])**2)**(0.5)

    def see_apple(self):

        app_pos = self.apple.get_pos()
        head = self.player.get_head()
        if head[0] == app_pos[0]:
            return True
        if head[1] == app_pos[1]:
            return True
        return False

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
            if pos[i][0] < 0 or pos[i][0] >= self.windowWidth or \
                pos[i][1] < 0 or pos[i][1] >= self.windowHeight:
                    return True
        return False

    def get_image(self):
        return pygame.surfarray.array3d(self.surface)

    def look(self, dir, move_size, body, apple, surface):

        pos = list(body[-1])
        dist = 0
        max = 20
        import pygame
        while pos[0] >= 0 and pos[0] < 800 and pos[1] >= 0 and pos[1] < 800:

            pygame.draw.circle(surface, (255, 0, 0), pos, 3)
            dist += 1
            pos[0] += dir[0] * move_size
            pos[1] += dir[1] * move_size
            for i in range(len(body) - 1):
                if body[i][0] == pos[0] and body[i][1] == pos[1]:

                    pygame.draw.circle(surface, (0, 255, 0), pos, 3)
                    return [1 / (dist + 1), 1, 0]
            if apple[0] == pos[0] and apple[1] == pos[1]:
                pygame.draw.circle(surface, (0, 0, 255), pos, 3)
                return [1 / (dist + 1), 0, 1]
        if dist == 0:
            pygame.draw.circle(surface, (255, 255, 255), pos, 3)
        return [1 / (dist + 1), 0, 0]

    def game_loop(self):
        while(self.running):
            if self.hunger > 120:
                break
            pygame.event.pump()
            input = self.inputs(self.get_image(), self.player.get_pos(), self.apple.get_pos(), self.surface)
            saw_apple = self.see_apple()
            start_dist = self.dist(self.player.get_head(), self.apple.get_pos())
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
            self.hunger += 1
            if self.ate_apple():
                self.player.grow()
                self.hunger = 0
            if self.is_dead():
                break

            self.surface.fill((0,0,0))
            self.player.draw(self.surface)
            self.apple.draw(self.surface)
            #self.draw_lines()
            move_dirs = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (-1, -1), (1, 1)]
            for dir in move_dirs:
                self.look(dir, 40, self.player.get_pos(), self.apple.get_pos(), self.surface)
            pygame.display.flip()

            self.moves += 1
            end_dist = self.dist(self.player.get_head(), self.apple.get_pos())
            if end_dist > start_dist and saw_apple:
                self.moves_away += 1
            pygame.display.update()
            #pygame.display.iconify()
            time.sleep(self.game_speed)

        pygame.quit()
        return len(self.player), self.closeness, self.moves, self.moves_away


    def draw_lines(self):
        start = copy.deepcopy(self.player.get_head())
        start[0] += 17
        start[1] += 17

        pygame.draw.line(self.surface, (225, 0, 0), start, (0, start[1]))
        pygame.draw.line(self.surface, (225, 0, 0), start, (800, start[1]))
        pygame.draw.line(self.surface, (225, 0, 0), start, (start[0], 0))
        pygame.draw.line(self.surface, (225, 0, 0), start, (start[0], 800))


    def hypot(self, a):
        return (a ** 2 + a ** 2) ** 0.5

    def dist(self, a, b):
        xd = a[0] - b[0]
        yd = a[1] - b[1]
        return (xd**2 + yd**2)**0.5


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
