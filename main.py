import pygame
import random
import neat
import os

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Game settings
GRAVITY = 0.5
FLAP_STRENGTH = -10
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150
BIRD_WIDTH = 40
BIRD_HEIGHT = 30

# Bird class
class Bird:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.alive = True

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        if self.y + BIRD_HEIGHT > SCREEN_HEIGHT or self.y < 0:
            self.alive = False

    def draw(self, screen):
        pygame.draw.rect(screen, BLUE, (self.x, self.y, BIRD_WIDTH, BIRD_HEIGHT))

# Pipe class
class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(50, SCREEN_HEIGHT - PIPE_GAP - 50)
        self.passed = False

    def update(self):
        self.x -= 5

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, (self.x, 0, PIPE_WIDTH, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT))

# Game function
def game(genomes, config):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    birds = []
    nets = []
    ge = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        genome.fitness = 0
        ge.append(genome)

    pipes = [Pipe(SCREEN_WIDTH + i * 300) for i in range(3)]

    score = 0
    running = True

    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i, bird in enumerate(birds):
            if bird.alive:
                pipe_ind = 0 if bird.x < pipes[0].x + PIPE_WIDTH else 1
                inputs = (
                    bird.y,
                    bird.velocity,
                    pipes[pipe_ind].height,
                    pipes[pipe_ind].height + PIPE_GAP,
                    pipes[pipe_ind].x - bird.x,
                )
                output = nets[i].activate(inputs)
                if output[0] > 0.5:
                    bird.flap()

                bird.update()
                bird.draw(screen)

                ge[i].fitness += 0.1

                for pipe in pipes:
                    if bird.x + BIRD_WIDTH > pipe.x and bird.x < pipe.x + PIPE_WIDTH:
                        if bird.y < pipe.height or bird.y + BIRD_HEIGHT > pipe.height + PIPE_GAP:
                            bird.alive = False
                            ge[i].fitness -= 1

        for pipe in pipes:
            pipe.update()
            pipe.draw(screen)
            if pipe.x + PIPE_WIDTH < birds[0].x and not pipe.passed:
                pipe.passed = True
                score += 1
                for g in ge:
                    g.fitness += 5

        for bird in birds:
            if not bird.alive:
                nets.remove(nets[birds.index(bird)])
                ge.remove(ge[birds.index(bird)])
                birds.remove(bird)

        if not birds:
            running = False

        pygame.display.update()
        clock.tick(30)

    pygame.quit()
    print(f"Score: {score}")

# Run the NEAT algorithm
def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(game, 50)
    return winner

# Absolute configuration path
config_path = "C:/Users/energ/Documents/Code/Flappy-GEN/config-feedforward.txt"

# Run the genetic algorithm
winner = run_neat(config_path)
