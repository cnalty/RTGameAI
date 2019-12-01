import pygame as pg
from pygame.math import Vector2 as vec
import random
import math
import time
import copy

class World():
    WIDTH = 400
    HEIGHT = 500
    SPACING = 150
    SPEED = -9

    def __init__(self, steering):
        pg.init()
        self.steering = steering
        self.surface = pg.display.set_mode((World.WIDTH, World.HEIGHT))
        self.flap = Flappy(World.WIDTH / 2, World.HEIGHT / 2)
        self.pipes = [Pipe(World.HEIGHT, World.WIDTH)]
        self.scoring_pipe = 0
        self.running = True

    def update(self):
        press = self.steering(self.dist_pipe())
        self.flap.update(press)
        for pipe in self.pipes:
            pipe.update(World.SPEED)
        if self.pipes[self.scoring_pipe].top.x + self.pipes[self.scoring_pipe].top.w < World.WIDTH / 2:
            self.flap.score += 1
            self.scoring_pipe += 1

        if self.pipes[-1].top.x + self.pipes[-1].top.w < World.WIDTH - World.SPACING:
            self.pipes.append(Pipe(World.HEIGHT, World.WIDTH))
        if self.pipes[0].top.x + self.pipes[0].top.w < 0:
            self.scoring_pipe -= 1
            self.pipes.pop(0)

    def draw(self):
        self.surface.fill((255, 255, 255))
        self.flap.draw(self.surface)
        for pipe in self.pipes:
            pipe.draw(self.surface)

        pg.display.flip()


    def game_loop(self):
        while self.running:
            pg.event.pump()
            self.update()
            for pipe in self.pipes:
                if self.collision(pipe):
                    self.running = False
            if self.hit_ground():
                self.running = False
            self.draw()
            self.flap.score += 1
            pg.display.update()
            time.sleep(0.05)
        return self.flap.score


    def collision(self, pipe):
        test_x = self.flap.pos[0]
        test_y1 = self.flap.pos[1]
        test_y2 = self.flap.pos[1]
        if self.flap.pos[0] < pipe.top.x:
            test_x = pipe.top.x
        elif self.flap.pos[0] > pipe.top.x + pipe.top.w:
            test_x =  pipe.top.x + pipe.top.w
        if self.flap.pos[1] < pipe.bot.y:
            test_y1 = pipe.bot.y
        if self.flap.pos[1] > pipe.top.y + pipe.top.h:
            test_y2 = pipe.top.y + pipe.top.h
        distx = self.flap.pos[0] - test_x
        disty1 = self.flap.pos[1] - test_y1
        disty2 = self.flap.pos[1] - test_y2

        dist1 = math.sqrt(distx * distx + disty1 * disty1)
        dist2 = math.sqrt(distx * distx + disty2 * disty2)
        if dist1 < Flappy.SIZE or dist2 < Flappy.SIZE:
            return True
        return False

    def hit_ground(self):
        if self.flap.pos[1] + Flappy.SIZE > World.HEIGHT:
            return True
        return False


    def look(self):
        start_vec = vec(self.pipes[0].top.w, 0)
        directions = [start_vec.rotate(90), start_vec.rotate(45), start_vec, start_vec.rotate(315), start_vec.rotate(270)]
        dists = [self.look_dir(dir) for dir in directions]
        #print(dists)
        return dists

    def look_dir(self, dir):
        pt = copy.deepcopy(self.flap.pos)
        board = pg.Rect(0, 0, World.WIDTH, World.HEIGHT)
        dist = 0
        while board.collidepoint(pt[0], pt[1]):
            dist += 1
            pt += dir
            for pipe in self.pipes:
                if pipe.top.collidepoint(pt[0], pt[1]) or pipe.bot.collidepoint(pt[0], pt[1]):
                    return 1 / dist
        if dist == 0:
            dist = 1
        return 1 / dist

    def dist_pipe(self):
        delta_x = self.pipes[self.scoring_pipe].top.x + self.pipes[self.scoring_pipe].top.w - self.flap.pos[0]
        delta_y = self.pipes[self.scoring_pipe].bot.y - self.flap.pos[1]
        #print(delta_x, delta_y)

        return delta_x / (World.WIDTH / 2), delta_y / (World.HEIGHT / 2)


class Flappy():
    MAX = 17
    G = 2
    SIZE = int(20)
    def __init__(self, x, y):
        self.pos = vec(x, y)
        self.vel = 0
        self.score = 0

    def update(self, pressed):
        if pressed and self.vel > -1:
            self.vel = -1 * Flappy.MAX
        else:
            self.vel += Flappy.G
        if self.vel > Flappy.MAX:
            self.vel = Flappy.MAX
        elif self.vel < -1 * Flappy.MAX:
            self.vel = -1 * Flappy.MAX
        self.pos[1] += self.vel

    def draw(self, surface):
        pg.draw.circle(surface, (255, 0, 0), (int(self.pos[0]), int(self.pos[1])), Flappy.SIZE)

class Pipe():
    SPACING = math.ceil(Flappy.SIZE * 8.5)
    def __init__(self, height, width):
        self.width = int(width / 6.5)
        self.min = int(.8 * height)
        self.max = int(.2 * height)
        self.top, self.bot = self.create_pipes(width, height)

    def create_pipes(self, start, floor):
        top_start = random.randint(self.max, self.min - Pipe.SPACING)
        bot_start = top_start + Pipe.SPACING
        top = pg.Rect(start, 0, self.width, top_start)
        bot = pg.Rect(start, bot_start, self.width, floor)
        return top, bot

    def update(self, speed):
        self.top = self.top.move(speed, 0)
        self.bot = self.bot.move(speed, 0)

    def draw(self, surface):
        #print(self.top)
        pg.draw.rect(surface, (0, 255, 0), self.top)
        pg.draw.rect(surface, (0, 255, 0), self.bot)


def test_game():
    def test_steer(thing):
        return False
    game = World(test_steer)
    game.game_loop()

if __name__ == "__main__":
    test_game()