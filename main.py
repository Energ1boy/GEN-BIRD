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
PIPE_GAP = 150  # Increased pipe gap
PIPE_DISTANCE = 300  # Increased distance between pipes
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

    def get_mask(self):
        return pygame.mask.from_surface(pygame.Surface((BIRD_WIDTH, BIRD_HEIGHT)))

    def get_position(self):
        return (self.x, self.y, self.x + BIRD_WIDTH, self.y + BIRD_HEIGHT)

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

    def get_rects(self):
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom_rect = pygame.Rect(self.x, self.height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - (self.height + PIPE_GAP))
        return (top_rect, bottom_rect)

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

    pipes = []
    pipes.append(Pipe(SCREEN_WIDTH + PIPE_DISTANCE))  # Initial pipes
    pipes.append(Pipe(SCREEN_WIDTH + 2 * PIPE_DISTANCE))
    pipes.append(Pipe(SCREEN_WIDTH + 3 * PIPE_DISTANCE))

    score = 0
    running = True

    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Manage pipes
        for i in range(len(pipes)-1, -1, -1):
            pipes[i].update()
            pipes[i].draw(screen)

            # Check collision with birds
            for bird in birds:
                for rect in pipes[i].get_rects():
                    bird_rect = pygame.Rect(bird.x, bird.y, BIRD_WIDTH, BIRD_HEIGHT)
                    if bird_rect.colliderect(rect):
                        bird.alive = False
                        ge[birds.index(bird)].fitness -= 1

            if pipes[i].x < -PIPE_WIDTH:
                pipes.pop(i)

        if pipes[-1].x < SCREEN_WIDTH - PIPE_DISTANCE:  # Spawn a new pipe ahead of the last pipe
            pipes.append(Pipe(SCREEN_WIDTH + PIPE_DISTANCE))

        # Manage birds
        for i, bird in enumerate(birds):
            if bird.alive:
                closest_pipe = None
                closest_dist = float('inf')
                for pipe in pipes:
                    if pipe.x > bird.x:
                        dist = pipe.x - bird.x
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_pipe = pipe

                if closest_pipe:
                    inputs = (
                        bird.y,
                        bird.velocity,
                        closest_pipe.height,
                        closest_pipe.height + PIPE_GAP,
                        closest_pipe.x - bird.x,
                    )
                    output = nets[i].activate(inputs)
                    if output[0] > 0.5:
                        bird.flap()

                    bird.update()
                    bird.draw(screen)

                    ge[i].fitness += 0.1

        # Remove dead birds
        for bird in birds:
            if not bird.alive:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        # Exit game if no birds are alive
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
