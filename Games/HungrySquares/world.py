import pygame as pg
from pygame.math import Vector2 as vec
import random
import time

class World():
    WIDTH = 800
    HEIGHT = 800
    FPS = 30
    NUM_FOOD = 50
    NUM_ANIMAL = 25

    def __init__(self, steering):
        pg.init()
        self.steering = steering
        self.surface = pg.display.set_mode((World.WIDTH, World.HEIGHT))
        self.clock = pg.time.Clock()
        self.food = pg.sprite.Group()
        self.animals = pg.sprite.Group()
        self.running = True
        self.world_age = 0
        for _ in range(World.NUM_ANIMAL):
            Animal(self.animals)
        for _ in range(World.NUM_FOOD):
            Food(self.food)

    def draw_world(self):
        self.surface.fill((255, 255, 255))
        self.food.draw(self.surface)
        self.animals.draw(self.surface)
        pg.display.flip()

    def update_world(self):
        for i, anim in zip(range(len(self.animals)), self.animals):
            dir = self.steering[i](self.look(anim))
            anim.update(dir)

        self.food.update()
        for anim in self.animals:
            for fd in self.food:
                if pg.sprite.collide_rect(anim, fd):
                    anim.life += 5
                    anim.food_ate += 1
                    self.food.remove(fd)
                    self.food.add(Food(self.food))

    def game_loop(self):
        while self.running:
            pg.event.pump()
            #self.clock.tick(World.FPS)
            self.update_world()
            self.draw_world()
            pg.display.update()
            self.world_age += 1
            if self.world_age > 500:
                self.running = False
            #time.sleep(0.001)


        pg.quit()
        return [x.food_ate for x in self.animals]


    def look(self, animal):
        directions = []
        look_dirs = [90, 75, 60, 45, 30, 15, 0, 360 - 15, 360 - 30, 360 - 45, 360 - 60, 360 - 75, 360 - 90]
        for dir in look_dirs:
            directions.extend(self.look_dir(dir, animal))
        return directions

    def look_dir(self, dir, animal):
        board = pg.Rect(0, 0, World.WIDTH, World.HEIGHT)
        pt = animal.rect.center
        mv_size = vec(Food.SIZE / 2, Food.SIZE / 2).magnitude()
        look_v = animal.vel.rotate(dir)
        look_v.scale_to_length(mv_size)
        dist = 1
        while board.collidepoint(pt[0], pt[1]):
            #print("looking")
            #pg.draw.circle(self.surface, (255, 0, 0), (int(pt[0]), int(pt[1])), 3)
            #pg.display.update()
            dist += 1
            pt += look_v
            for food in self.food:
                if food.rect.collidepoint(pt[0], pt[1]):
                    #print("see food")
                    return [1 / dist, 1]

        return [1 / dist, 0]





class Animal(pg.sprite.Sprite):
    MAX_SPEED = 3
    TURN_FORCE = 16
    SIZE = int(15)
    def __init__(self, group):
        self.groups = group
        pg.sprite.Sprite.__init__(self, self.groups)
        self.image = pg.Surface((Animal.SIZE, Animal.SIZE))
        self.image.fill((0,0,0))
        self.rect = self.image.get_rect()
        self.pos = vec(random.randint(0, World.WIDTH), random.randint(0, World.HEIGHT))
        self.vel = vec(Animal.MAX_SPEED, 0).rotate(random.uniform(0, 360))
        self.acc = vec(0, 0)
        self.rect.center = self.pos
        self.life = 30
        self.food_ate = 0

    def steer(self, direction):
        if direction == "L":
            self.vel = self.vel.rotate(Animal.TURN_FORCE)
        elif direction == "R":
            self.vel = self.vel.rotate(360 - Animal.TURN_FORCE)

    def update(self, direction):
        self.steer(direction)
        self.pos += self.vel

        self.life -= 1
        board = pg.Rect(0, 0, World.WIDTH, World.HEIGHT)
        if not board.collidepoint(self.pos[0], self.pos[1]):
            self.vel *= -1
            if self.pos[0] > World.WIDTH:
                self.pos[0] = World.WIDTH
            elif self.pos[0] < 0:
                self.pos[0] = 0
            if self.pos[1] > World.HEIGHT:
                self.pos[1] = World.HEIGHT
            elif self.pos[1] < 0:
                self.pos[1] = 0

        self.rect.center = self.pos


class Food(pg.sprite.Sprite):
    SIZE = 10
    def __init__(self, group):
        self.groups = group
        pg.sprite.Sprite.__init__(self, self.groups)
        self.image = pg.Surface((Food.SIZE, Food.SIZE))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.pos = vec(random.randint(0, World.WIDTH), random.randint(0, World.HEIGHT))
        self.rect.center = self.pos

