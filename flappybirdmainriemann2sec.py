import pygame
import sys
import random

import socket
import struct
import threading
import time
import numpy as np
import mne
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import joblib

from pygame import mixer
#from fighterwithbotwithbci import Fighter
#from sfgamewithbotandbcicontrol import game2
from huaqiaogamewithbciriemann2sec import Game3
from Calibration2sec import BCICalibrator
#from subway import subway_surfers_game

ip_address = "192.168.31.112"
server_port = 8712
n_chan = 18
sample_rate = 250
trial_samples = 625  # 4.5 seconds of data at 250 Hz

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Main Menu")

# Colors
WHITE = (255, 255, 255)
BLUE = (135, 206, 250)
BLACK = (0, 0, 0)
GREEN = (0, 255,0)

# Load images
BG = pygame.image.load("assets/images.png")
button_image = pygame.image.load("assets/betterwood.png").convert_alpha()

# Scale images
BUTTON_WIDTH, BUTTON_HEIGHT= 160,70
BG=pygame.transform.scale(BG, (SCREEN_WIDTH, SCREEN_HEIGHT)) 
button_image=pygame.transform.scale(button_image, (BUTTON_WIDTH, BUTTON_HEIGHT)) 

def get_font(size): 
    return pygame.font.Font("assets/font.ttf", size)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to 2D numpy array.
    
    Parameters:
        data (numpy.ndarray): Input array of shape (n_channels, n_samples)
        lowcut (float): Low cutoff frequency (Hz)
        highcut (float): High cutoff frequency (Hz)
        fs (float): Sampling frequency (Hz)
        order (int): Filter order
        
    Returns:
        numpy.ndarray: Filtered data with same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
    
    return filtered_data

class DataClient:
    def __init__(self, ip_address, server_port, n_chan, sample_rate, trial_samples):
        self.ip_address = ip_address
        self.server_port = server_port
        self.n_chan = n_chan
        self.sample_rate = sample_rate
        self.trial_samples = trial_samples
        self.current_eeg = None
        self.command = None  # Current command
        self.processing_flag = True  # Flag to indicate if a command is being processed

        # Calculate buffer size for one trial
        self.bytes_per_sample = 4  # 32-bit float
        self.buffer_size = self.n_chan * self.trial_samples * self.bytes_per_sample

        # Initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip_address, self.server_port))

        # Thread for receiving data
        self.receive_thread = threading.Thread(target=self.receive_data)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        # Lock for synchronization
       # self.lock = threading.Lock()

    def receive_data(self):
        while True:
            try:
                # Initialize an empty buffer
                data = b''
                while len(data) < self.buffer_size:
                    # Receive data in chunks
                    packet = self.sock.recv(self.buffer_size - len(data))
                    if not packet:
                        break  # No more data
                    data += packet

                if len(data) == self.buffer_size:
                    num_samples = self.trial_samples
                    eeg_data = struct.unpack(f"{num_samples * self.n_chan}f", data)
                    self.rawcurrent_eeg = np.array(eeg_data).reshape(self.n_chan, num_samples)
                    self.current_eeg = bandpass_filter(self.rawcurrent_eeg,0.7,35,250,5)
                    
                    print(self.current_eeg.shape) #OUTPUT 18,1125 , when fetching for add_calibration_data , probably just add an axis to it

               #  self.lock.acquire()
               # self.processing_flag = False # maybe need to put at calibrator
                # self.processing_flag =False after processing

            except Exception as e:
                print(f"Error receiving data: {e}")
                break

          #  self.lock.release()

    def close(self):
        self.sock.close()

class Button():
	def __init__(self, image, pos, text_input, font, base_color, hovering_color):
		self.image = image
		self.x_pos = pos[0]
		self.y_pos = pos[1]
		self.font = font
		self.base_color, self.hovering_color = base_color, hovering_color
		self.text_input = text_input
		self.text = self.font.render(self.text_input, True, self.base_color)
		if self.image is None:
			self.image = self.text
        
		self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
		self.text_rect = self.text.get_rect(center=(self.x_pos, self.y_pos))
        

	def update(self, screen):
		if self.image is not None:
			screen.blit(self.image, self.rect)
		screen.blit(self.text, self.text_rect)

	def checkForInput(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def changeColor(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			self.text = self.font.render(self.text_input, True, self.hovering_color)
		else:
			self.text = self.font.render(self.text_input, True, self.base_color)

# Game loop

def subway_surfers_game(use_bci=False):
   # global command_counter, command_threshold, client, bci_calibrator
    
    # Initialize pygame if not already initialized
    if not pygame.get_init():
        pygame.init()
    
    # Game constants
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 700
    FPS = 60
    PLAYER_SPEED = 5
    OBSTACLE_SPEED =2
    LANE_POSITIONS = [200, 400, 600]  # Three lanes for movement

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GRAY = (100, 100, 100)
    BLUE = (0, 0, 255)
    GOLD = (255, 215, 0)

    # Create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Subway Surfers")
    clock = pygame.time.Clock()

    class Player:
        def __init__(self):
            self.width = 50
            self.height = 70
            self.x = LANE_POSITIONS[1] - self.width // 2  # Start in middle lane
            self.y = SCREEN_HEIGHT - 100
            self.color = BLUE
            self.current_lane = 1  # 0: left, 1: middle, 2: right
            self.is_jumping = False
            self.jump_height = 100
            self.jump_progress = 0
            
        def draw(self, surface):
            pygame.draw.rect(surface, self.color, (self.x, self.y - self.jump_progress, self.width, self.height))
            
        def move_left(self):
            if self.current_lane > 0:
                self.current_lane -= 1
                self.x = LANE_POSITIONS[self.current_lane] - self.width // 2
                
        def move_right(self):
            if self.current_lane < 2:
                self.current_lane += 1
                self.x = LANE_POSITIONS[self.current_lane] - self.width // 2
                
        def jump(self):
            if not self.is_jumping:
                self.is_jumping = True
                self.jump_progress = 0
                
        def update_jump(self):
            if self.is_jumping:
                self.jump_progress += 5
                if self.jump_progress >= self.jump_height:
                    self.is_jumping = False
                    self.jump_progress = 0

    class Obstacle:
        def __init__(self, lane):
            self.width = 30
            self.height = 45
            self.lane = lane
            self.x = LANE_POSITIONS[lane] - self.width // 2
            self.y = -self.height
            self.color = RED
            
        def draw(self, surface):
            pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))
            
        def update(self):
            self.y += OBSTACLE_SPEED
            return self.y > SCREEN_HEIGHT
            
        def collides_with(self, player):
            if (player.x < self.x + self.width and
                player.x + player.width > self.x and
                player.y < self.y + self.height and
                player.y + player.height > self.y):
                return True
            return False

    class Coin:
        def __init__(self, lane):
            self.width = 15
            self.height = 15
            self.lane = lane
            self.x = LANE_POSITIONS[lane] - self.width // 2
            self.y = -self.height
            self.color = GOLD
            
        def draw(self, surface):
            pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))
            
        def update(self):
            self.y += OBSTACLE_SPEED
            return self.y > SCREEN_HEIGHT
            
        def collected_by(self, player):
            if (player.x < self.x + self.width and
                player.x + player.width > self.x and
                player.y < self.y + self.height and
                player.y + player.height > self.y):
                return True
            return False

    def draw_lanes(surface):
        for pos in LANE_POSITIONS:
            pygame.draw.line(surface, GRAY, (pos, 0), (pos, SCREEN_HEIGHT), 2)

    def show_score(surface, score, font):
        score_text = font.render(f"Score: {score}", True, WHITE)
        surface.blit(score_text, (10, 10))

    def show_game_over(surface, font):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black
        surface.blit(overlay, (0, 0))
        
        game_over_text = font.render("Game Over!", True, WHITE)
        restart_text = font.render("Press R to restart", True, WHITE)
        quit_text = font.render("Press Q to quit", True, WHITE)
        
        surface.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2 - 50))
        surface.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, SCREEN_HEIGHT//2))
        surface.blit(quit_text, (SCREEN_WIDTH//2 - quit_text.get_width()//2, SCREEN_HEIGHT//2 + 50))

    # BCI setup if needed
    command_counter = 0
    command_threshold = 5
    client = None
    bci_calibrator = None
    
    if use_bci:
        client = DataClient(ip_address, server_port, n_chan, sample_rate, trial_samples)
        bci_calibrator = BCICalibrator(n_chan=n_chan)   

        # Loading screen
        loading_font = pygame.font.SysFont(None, 50)
        loading_text = loading_font.render("Loading...", True, BLACK)
        screen.fill(WHITE)
        screen.blit(loading_text, (SCREEN_WIDTH // 2 - 70, SCREEN_HEIGHT // 2))
        pygame.display.flip()

        while command_counter < command_threshold:
            if client.current_eeg is not None:
                command_counter += 1
                print(f"Ignoring command {command_counter}/{command_threshold}")
                time.sleep(1)

        def show_calibration_screen(text, duration, label):
            font = pygame.font.SysFont(None, 50)
            client.current_eeg = None
            start_time = time.time()
            
            while time.time() - start_time < duration:
                screen.fill(BLACK)
                text_surf = font.render(text, True, WHITE)
                text_rect = text_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2-50))
                screen.blit(text_surf, text_rect)
                
                progress = (time.time() - start_time)/duration
                pygame.draw.rect(screen, WHITE, (SCREEN_WIDTH//2-100, SCREEN_HEIGHT//2+50, 200, 20), 1)
                pygame.draw.rect(screen, GREEN, (SCREEN_WIDTH//2-100, SCREEN_HEIGHT//2+50, 200*progress, 20))
                
                if client.current_eeg is not None:
                    bci_calibrator.add_calibration_data_game2(client.current_eeg, label)   #change this if needed
                    client.current_eeg = None
                
                pygame.display.flip()
                time.sleep(0.1)
     
        # Run calibration
       #  show_calibration_screen("Imagine Left Hand", 90, 'action')
        # show_calibration_screen("Relax (Empty State)", 90, 'empty') 
        show_calibration_screen("Imagine Left Hand", 90, 'lefthand')
        show_calibration_screen("Imagine Right Hand", 90, 'righthand')
        
        # Train classifier
        try:
            bci_calibrator.train_game2()                                        #chagne this if needed too
            screen.fill(BLACK) 
            font = pygame.font.SysFont(None, 50)
            text = font.render("Calibration Complete!", True, GREEN)
            screen.blit(text, (SCREEN_WIDTH//2-150, SCREEN_HEIGHT//2))
            pygame.display.flip()
            time.sleep(2)
        except Exception as e:
            print(f"Calibration failed: {e}")
            client.close()
            return

    # Main game function
    def run_game():
        player = Player()
        obstacles = []
        coins = []
        score = 0
        obstacle_timer = 0
        coin_timer = 0
        game_over = False
        font = pygame.font.SysFont(None, 36)
        
        # For BCI control
        bci_command = None
        client.current_eeg = None
        
        while True:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if use_bci:
                        client.close()
                    pygame.quit()
                    return "quit"
                
                if game_over:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            client.data = None
                            return "restart"
                        elif event.key == pygame.K_q:
                            return "quit"
                else:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                            player.move_left()
                        elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                            player.move_right()
                        elif event.key == pygame.K_w or event.key == pygame.K_UP:
                            player.jump()
            
            # BCI control
            if use_bci and client.current_eeg is not None:
                prediction = bci_calibrator.predict(client.current_eeg)
                if prediction == 0:  # Left
                    player.move_left()
                elif prediction == 1:  # Right
                    player.move_right()
                client.current_eeg = None
            
            if game_over:
                screen.fill(BLACK)
                draw_lanes(screen)
                player.draw(screen)
                show_score(screen, score, font)
                show_game_over(screen, font)
                pygame.display.flip()
                clock.tick(FPS)
                continue
            
            # Game logic
            player.update_jump()
            
            # Spawn obstacles
            obstacle_timer += 1
            if obstacle_timer >= random.randint(5*FPS,5*FPS+20):
                lane = random.randint(0, 2)
                obstacles.append(Obstacle(lane))
                obstacle_timer = 0
                
            # Spawn coins
      #      coin_timer += 1
        #    if coin_timer >= random.randint(60, 120):
       #         lane = random.randint(0, 2)
       #         coins.append(Coin(lane))
      #          coin_timer = 0
            
            # Update obstacles
            for obstacle in obstacles[:]:
                if obstacle.update():
                    obstacles.remove(obstacle)
                    score += 1
                elif obstacle.collides_with(player):
                    game_over = True
            
            # Update coins
      #      for coin in coins[:]:
      #          if coin.update():
        #            coins.remove(coin)
       #         elif coin.collected_by(player):
        #            coins.remove(coin)
       #             score += 5
            
            # Drawing
            screen.fill(BLACK)
            draw_lanes(screen)
            
            for obstacle in obstacles:
                obstacle.draw(screen)
                
     #       for coin in coins:
     #           coin.draw(screen)
                
            player.draw(screen)
            show_score(screen, score, font)
            
            pygame.display.flip()
            clock.tick(FPS)

    # Start the game loop
    result = run_game()
    while result == "restart":        
        result = run_game()
    
    # Clean up
    if use_bci:
        client.close()
    pygame.display.quit()
    menu_for_game()
    return "menu"

def test_input_mode():               #modified into read_me
    font = pygame.font.SysFont(None, 30)
    title_font = pygame.font.SysFont(None, 50)
   
    running = True
    while running:
        MENU_MOUSE_POS = pygame.mouse.get_pos()
        screen.fill(WHITE)  # Clear screen with blue background
        screen.blit(BG, (0, 0))
    # Clear screen and show initial instructions
        title = title_font.render("Instructions", True, BLACK)
        instruction1 = font.render("During the Calibration process:", True, BLACK)
        instruction2 = font.render("1.Relax", True, BLACK)
        instruction3 = font.render("2.Maintain consistent imagination", True, BLACK)
        instruction4 = font.render("1.Try to replicate what you imagined", True, BLACK)
        instruction5 = font.render("in calibration when playing", True, BLACK)
        instruction6 = font.render("During the gameplay:", True, BLACK)
        QUIT_BUTTON = Button(image=button_image, pos=(200, 510), 
            text_input="Return", font=get_font(36), base_color="Black", hovering_color="White")
        
        for button in [QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(screen)
        
        screen.blit(title, (95, 50))
        screen.blit(instruction1, (25, 125))
        screen.blit(instruction2, (25, 175))
        screen.blit(instruction3, (25, 225))
        screen.blit(instruction6, (25, 275))
        screen.blit(instruction4, (25, 325))
        screen.blit(instruction5, (45, 375))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    main_menu()    # just change this function to another game_select_menu and then game select menu give option for game1 and game 2

        pygame.display.update()

def main_menu():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    while True:
        screen.blit(BG, (0, 0))
        #screen.fill(BLUE)
        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(50).render("MAIN MENU", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(200, 70))

        PLAY_BUTTON = Button(image=button_image, pos=(200, 210), 
                            text_input="PLAY", font=get_font(36), base_color="Black", hovering_color="White")
        OPTIONS_BUTTON = Button(image=button_image, pos=(200, 360), 
                            text_input="ReadMe", font=get_font(36), base_color="Black", hovering_color="White")
        QUIT_BUTTON = Button(image=button_image, pos=(200, 510), 
                            text_input="QUIT", font=get_font(36), base_color="Black", hovering_color="White")

        screen.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, OPTIONS_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    menu_for_game()    # just change this function to another game_select_menu and then game select menu give option for game1 and game 2
                if OPTIONS_BUTTON.checkForInput(MENU_MOUSE_POS):
                    test_input_mode()  # readme
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()
        
def menu_for_game():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    while True:
        screen.blit(BG, (0, 0))
        #screen.fill(BLUE)
        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(35).render("Choose Your Game", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(200, 70))

     #   First_BUTTON = Button(image=button_image, pos=(200, 120), 
     #                       text_input="FlappyBird", font=get_font(20), base_color="Black", hovering_color="White")
        Second_BUTTON = Button(image=button_image, pos=(200, 210), 
                            text_input="SubwayGame", font=get_font(25), base_color="Black", hovering_color="White")
        Third_BUTTON = Button(image=button_image, pos=(200, 360), 
                            text_input="HuaQiaoExplore", font=get_font(20), base_color="Black", hovering_color="White")
        Return_BUTTON = Button(image=button_image, pos=(200, 510), 
                              text_input="Return", font=get_font(36), base_color="Black", hovering_color="White")                    

        screen.blit(MENU_TEXT, MENU_RECT)

        for button in [Second_BUTTON, Third_BUTTON,Return_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(screen)           

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
           #     if First_BUTTON.checkForInput(MENU_MOUSE_POS):
          #          game1()    # just change this function to another game_select_menu and then game select menu give option for game1 and game 2
                if Second_BUTTON.checkForInput(MENU_MOUSE_POS):
                    subway_surfers_game(use_bci= True)
                if Third_BUTTON.checkForInput(MENU_MOUSE_POS):
                    game3 = Game3()
                    game3.run()
                if Return_BUTTON.checkForInput(MENU_MOUSE_POS):     
                    main_menu()    
     
        pygame.display.update()

# Run the menu
main_menu()

