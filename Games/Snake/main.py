import pygame
from game import Snake

def buttons(image, head, apple):
    return pygame.key.get_pressed()

def main():
    keys = {pygame.K_LEFT: "L", pygame.K_RIGHT: "R", pygame.K_UP: "U", pygame.K_DOWN: "D", pygame.K_ESCAPE: "E"}
    game = Snake(buttons, keys)
    print(game.game_loop())


if __name__ == "__main__":
    main()