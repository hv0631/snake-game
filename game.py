import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE
TEXT_COLOR = (255, 255, 255)
RED = (255, 0, 0)

# Color Options
COLORS = {
    "Green": (0, 255, 0),
    "Red": (255, 0, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Purple": (128, 0, 128),
    "Cyan": (0, 255, 255),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
}

# Difficulty Levels
DIFFICULTY = {
    "Easy": (5, 0),
    "Medium": (10, 5),
    "Hard": (15, 10),
}

# Game Modes
GAME_MODES = {
    "Classic": {"description": "Traditional snake game", "food_value": 1, "special_features": None},
    "Time Attack": {"description": "Score as much as possible in 60 seconds", "food_value": 1, "special_features": "timer"},
    "Survival": {"description": "Moving obstacles make the game harder", "food_value": 1, "special_features": "moving_obstacles"}
}

# Directions
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.font = pygame.font.Font(None, 30)
        
    def draw(self):
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=5)
        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
    def is_hovered(self, pos):
        return self.rect.collidepoint(pos)
        
    def update(self, pos):
        if self.is_hovered(pos):
            self.current_color = self.hover_color
        else:
            self.current_color = self.color

class SnakeGame:
    def __init__(self, player_name, snake_color, bg_color, speed, hurdles_count, game_mode):
        self.snake = [(5, 5)]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.running = True
        self.score = 0
        self.player_name = player_name
        self.snake_color = snake_color
        self.bg_color = bg_color
        self.speed = speed
        self.game_mode = game_mode
        self.hurdles = self.place_hurdles(hurdles_count)
        self.moving_obstacles = []
        self.food = self.place_food()
        self.start_time = pygame.time.get_ticks()
        self.time_left = 60  # 60 seconds for Time Attack mode
        
        # Set up moving obstacles for Survival mode
        if game_mode == "Survival":
            self.setup_moving_obstacles(3)  # Start with 3 moving obstacles

    def setup_moving_obstacles(self, count):
        for _ in range(count):
            # Position, direction, and speed
            obstacle = {
                'pos': (random.randint(0, ROWS - 1), random.randint(0, COLS - 1)),
                'dir': random.choice([UP, DOWN, LEFT, RIGHT]),
                'speed': random.randint(10, 20)  # Lower is faster
            }
            # Make sure obstacles don't spawn on the snake
            while obstacle['pos'] in self.snake:
                obstacle['pos'] = (random.randint(0, ROWS - 1), random.randint(0, COLS - 1))
            self.moving_obstacles.append(obstacle)

    def place_food(self):
        while True:
            food = (random.randint(0, ROWS - 1), random.randint(0, COLS - 1))
            if food not in self.snake and food not in self.hurdles:
                return food

    def place_hurdles(self, count):
        hurdles = set()
        while len(hurdles) < count:
            hurdle = (random.randint(0, ROWS - 1), random.randint(0, COLS - 1))
            if hurdle not in self.snake:
                hurdles.add(hurdle)
        return hurdles

    def update_moving_obstacles(self):
        for obstacle in self.moving_obstacles:
            # Only move the obstacle occasionally based on its speed
            if random.randint(1, obstacle['speed']) == 1:
                # Randomly change direction sometimes
                if random.randint(1, 10) == 1:
                    obstacle['dir'] = random.choice([UP, DOWN, LEFT, RIGHT])
                
                # Calculate new position
                new_pos = (obstacle['pos'][0] + obstacle['dir'][0], 
                           obstacle['pos'][1] + obstacle['dir'][1])
                
                # If new position is outside the grid or occupied by a hurdle, choose a new direction
                if not (0 <= new_pos[0] < ROWS and 0 <= new_pos[1] < COLS) or new_pos in self.hurdles:
                    obstacle['dir'] = random.choice([UP, DOWN, LEFT, RIGHT])
                    continue
                
                # Update position
                obstacle['pos'] = new_pos

    def move(self):
        if not self.running:
            return
            
        # Update game timer for Time Attack mode
        if self.game_mode == "Time Attack":
            current_time = pygame.time.get_ticks()
            elapsed_seconds = (current_time - self.start_time) // 1000
            self.time_left = max(0, 60 - elapsed_seconds)
            if self.time_left <= 0:
                self.running = False
                return

        # Update moving obstacles for Survival mode
        if self.game_mode == "Survival":
            self.update_moving_obstacles()
            
        if (self.next_direction[0] * -1, self.next_direction[1] * -1) != self.direction:
            self.direction = self.next_direction
        
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        # Check collision with walls, snake body, hurdles, or moving obstacles
        if (new_head in self.snake or 
            new_head in self.hurdles or 
            not (0 <= new_head[0] < ROWS and 0 <= new_head[1] < COLS) or
            any(new_head == obstacle['pos'] for obstacle in self.moving_obstacles)):
            self.running = False
            return
        
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.place_food()
            self.score += 1
            
            # In Survival mode, add a new moving obstacle every 5 points
            if self.game_mode == "Survival" and self.score % 5 == 0:
                self.setup_moving_obstacles(1)
        else:
            self.snake.pop()

    def draw(self):
        screen.fill(self.bg_color)
        
        # Draw the snake
        for segment in self.snake:
            pygame.draw.rect(screen, self.snake_color, 
                            (segment[1] * GRID_SIZE, segment[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw the food
        pygame.draw.rect(screen, RED, 
                        (self.food[1] * GRID_SIZE, self.food[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw hurdles
        for hurdle in self.hurdles:
            pygame.draw.rect(screen, (200, 200, 200), 
                            (hurdle[1] * GRID_SIZE, hurdle[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw moving obstacles
        for obstacle in self.moving_obstacles:
            pygame.draw.rect(screen, (255, 128, 0),  # Orange color for moving obstacles
                            (obstacle['pos'][1] * GRID_SIZE, obstacle['pos'][0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"{self.player_name} - Score: {self.score}", True, TEXT_COLOR)
        screen.blit(score_text, (10, 10))
        
        # Draw timer for Time Attack mode
        if self.game_mode == "Time Attack":
            timer_text = font.render(f"Time: {self.time_left}s", True, TEXT_COLOR)
            screen.blit(timer_text, (WIDTH - 150, 10))
        
        # Draw mode indicator
        mode_text = font.render(f"Mode: {self.game_mode}", True, TEXT_COLOR)
        screen.blit(mode_text, (10, HEIGHT - 30))
        
        pygame.display.update()

    def game_over_screen(self):
        font = pygame.font.Font(None, 50)
        screen.fill((0, 0, 0))
        game_over_text = font.render("Game Over!", True, RED)
        score_text = font.render(f"{self.player_name} - Final Score: {self.score}", True, TEXT_COLOR)
        mode_text = font.render(f"Mode: {self.game_mode}", True, TEXT_COLOR)
        restart_text = font.render("Press SPACE to restart", True, TEXT_COLOR)
        restart_text1 = font.render("Press ESC to quit", True, TEXT_COLOR)
        
        screen.blit(game_over_text, (WIDTH // 3, HEIGHT // 3))
        screen.blit(score_text, (WIDTH // 4, HEIGHT // 2))
        screen.blit(mode_text, (WIDTH // 4, HEIGHT // 2 + 50))
        screen.blit(restart_text, (WIDTH // 4, HEIGHT // 2 + 100))
        screen.blit(restart_text1, (WIDTH // 4, HEIGHT // 2 + 150))
        
        pygame.display.update()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                        return True  # Restart
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            pygame.time.delay(100)

def mode_selection_screen():
    screen.fill(COLORS["Black"])
    font_large = pygame.font.Font(None, 50)
    font_small = pygame.font.Font(None, 30)
    
    title = font_large.render("Select Game Mode", True, TEXT_COLOR)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
    
    # Create buttons for each game mode
    buttons = []
    y_position = 150
    
    for i, (mode, details) in enumerate(GAME_MODES.items()):
        button = Button(WIDTH // 4, y_position, WIDTH // 2, 60, mode, (50, 50, 150), (100, 100, 200))
        buttons.append(button)
        
        description = font_small.render(details["description"], True, TEXT_COLOR)
        screen.blit(description, (WIDTH // 2 - description.get_width() // 2, y_position + 70))
        
        y_position += 120
    
    back_button = Button(WIDTH // 4, y_position, WIDTH // 2, 40, "Back", (150, 50, 50), (200, 100, 100))
    buttons.append(back_button)
    
    pygame.display.update()
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, button in enumerate(buttons):
                    if button.is_hovered(mouse_pos):
                        if i < len(GAME_MODES):  # Game mode buttons
                            return list(GAME_MODES.keys())[i]
                        else:  # Back button
                            return None
        
        # Update button hover states
        for button in buttons:
            button.update(mouse_pos)
            button.draw()
        
        pygame.display.update()
    
    return None

def difficulty_selection_screen():
    screen.fill(COLORS["Black"])
    font_large = pygame.font.Font(None, 50)
    
    title = font_large.render("Select Difficulty", True, TEXT_COLOR)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
    
    # Create buttons for each difficulty
    buttons = []
    y_position = 150
    
    for difficulty in DIFFICULTY.keys():
        button = Button(WIDTH // 4, y_position, WIDTH // 2, 60, difficulty, (50, 50, 150), (100, 100, 200))
        buttons.append(button)
        y_position += 100
    
    back_button = Button(WIDTH // 4, y_position, WIDTH // 2, 40, "Back", (150, 50, 50), (200, 100, 100))
    buttons.append(back_button)
    
    pygame.display.update()
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, button in enumerate(buttons):
                    if button.is_hovered(mouse_pos):
                        if i < len(DIFFICULTY):  # Difficulty buttons
                            return list(DIFFICULTY.keys())[i]
                        else:  # Back button
                            return None
        
        # Update button hover states
        for button in buttons:
            button.update(mouse_pos)
            button.draw()
        
        pygame.display.update()
    
    return None

def color_selection_screen():
    screen.fill(COLORS["Black"])
    font_large = pygame.font.Font(None, 50)
    
    title = font_large.render("Select Snake Color", True, TEXT_COLOR)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
    
    # Create buttons for each color
    buttons = []
    color_buttons = {}
    y_position = 150
    cols = 2
    button_width = WIDTH // 3
    button_height = 40
    button_margin = 20
    
    for i, (color_name, color_value) in enumerate(COLORS.items()):
        if color_name == "White" or color_name == "Black":  # Skip background colors
            continue
            
        row = i // cols
        col = i % cols
        x = WIDTH // 4 + col * (button_width + button_margin)
        y = y_position + row * (button_height + button_margin)
        
        button = Button(x, y, button_width, button_height, color_name, color_value, color_value)
        buttons.append(button)
        color_buttons[button] = color_name
    
    back_y = y_position + (len(COLORS) // cols + 1) * (button_height + button_margin)
    back_button = Button(WIDTH // 4, back_y, WIDTH // 2, 40, "Back", (150, 50, 50), (200, 100, 100))
    buttons.append(back_button)
    
    pygame.display.update()
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons:
                    if button.is_hovered(mouse_pos):
                        if button == back_button:
                            return None
                        else:
                            return color_buttons[button]
        
        # Update button hover states
        for button in buttons:
            button.update(mouse_pos)
            button.draw()
        
        pygame.display.update()
    
    return None

def main_menu():
    screen.fill(COLORS["Black"])
    font_large = pygame.font.Font(None, 60)
    font_small = pygame.font.Font(None, 30)
    
    title = font_large.render("Snake Game", True, TEXT_COLOR)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))
    
    start_button = Button(WIDTH // 4, 200, WIDTH // 2, 60, "Start Game", (50, 150, 50), (100, 200, 100))
    options_button = Button(WIDTH // 4, 280, WIDTH // 2, 60, "Options", (50, 50, 150), (100, 100, 200))
    quit_button = Button(WIDTH // 4, 360, WIDTH // 2, 60, "Quit", (150, 50, 50), (200, 100, 100))
    
    version_text = font_small.render("v1.0", True, TEXT_COLOR)
    screen.blit(version_text, (WIDTH - 50, HEIGHT - 30))
    
    pygame.display.update()
    
    # Default settings
    player_name = "Player"
    snake_color = COLORS["Green"]
    bg_color = COLORS["Black"]
    difficulty = "Easy"
    game_mode = "Classic"
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.is_hovered(mouse_pos):
                    # First select game mode
                    selected_mode = mode_selection_screen()
                    if selected_mode:
                        game_mode = selected_mode
                        # Then select difficulty
                        selected_difficulty = difficulty_selection_screen()
                        if selected_difficulty:
                            difficulty = selected_difficulty
                            speed, hurdles_count = DIFFICULTY[difficulty]
                            # Start the game
                            clock = pygame.time.Clock()
                            game = SnakeGame(player_name, snake_color, bg_color, speed, hurdles_count, game_mode)
                            
                            while game.running:
                                for game_event in pygame.event.get():
                                    if game_event.type == pygame.QUIT:
                                        pygame.quit()
                                        sys.exit()
                                    elif game_event.type == pygame.KEYDOWN:
                                        if game_event.key == pygame.K_UP and game.direction != DOWN:
                                            game.next_direction = UP
                                        elif game_event.key == pygame.K_DOWN and game.direction != UP:
                                            game.next_direction = DOWN
                                        elif game_event.key == pygame.K_LEFT and game.direction != RIGHT:
                                            game.next_direction = LEFT
                                        elif game_event.key == pygame.K_RIGHT and game.direction != LEFT:
                                            game.next_direction = RIGHT
                                
                                game.move()
                                game.draw()
                                clock.tick(speed)
                            
                            restart = game.game_over_screen()
                            if restart:
                                # Restart at main menu
                                screen.fill(COLORS["Black"])
                                pygame.display.update()
                            
                elif options_button.is_hovered(mouse_pos):
                    # Change snake color
                    selected_color = color_selection_screen()
                    if selected_color:
                        snake_color = COLORS[selected_color]
                
                elif quit_button.is_hovered(mouse_pos):
                    pygame.quit()
                    sys.exit()
        
        # Update button hover states
        start_button.update(mouse_pos)
        options_button.update(mouse_pos)
        quit_button.update(mouse_pos)
        
        # Draw buttons
        screen.fill(COLORS["Black"])
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))
        start_button.draw()
        options_button.draw()
        quit_button.draw()
        screen.blit(version_text, (WIDTH - 50, HEIGHT - 30))
        
        pygame.display.update()

if __name__ == "__main__":
    main_menu()