from settings import *
from pytmx.util_pygame import load_pygame
from os.path import join

from sprites import Sprite,BorderSprite
from playerwithbciriemann import Player,Character
from groups import AllSprites
from support import *
from Calibration2sec import BCICalibrator

import socket
import struct
import threading
import time
import numpy as np
import joblib
from scipy import signal

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


class Game3:
    def __init__(self):
        self.ip_address = "192.168.31.112"
        self.server_port = 8712
        self.n_chan = 18
        self.sample_rate = 250
        self.trial_samples = 625  # 4.5 seconds of data at 250 Hz
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        self.ZOOM_FACTOR = 3
        self.command_counter = 0
        self.command_threshold = 5  
        self.camera_surface = pygame.Surface((WINDOW_WIDTH // self.ZOOM_FACTOR, WINDOW_HEIGHT // self.ZOOM_FACTOR))
        pygame.display.set_caption('Lets Explore Hua Qiao University')
        self.clock = pygame.time.Clock()



        #groups
        self.all_sprites = AllSprites()
        self.collision_sprites = pygame.sprite.Group()
        self.character_sprites = pygame.sprite.Group()


        self.import_assets()
        self.setup(self.tmx_maps['world'],'house')
    
    def import_assets(self):
        self.tmx_maps = {'world': load_pygame(join('Desktop','finalyear','Flappy_bird_test','huaqiaomap(finetuned).tmx'))}

        self.overworld_frames = {
            'characters' : all_character_import('Desktop','finalyear','Flappy_bird_test','Character')
        }
        
    
    def setup(self, tmx_map, player_start_pos):
        #Base layer 
        for x,y, surf in tmx_map.get_layer_by_name('Tile Layer 1').tiles():
            Sprite((x * TILE_SIZE, y*TILE_SIZE), surf, self.all_sprites)
        #Player Starting Pos
        self.player = Player(
            pos= (640,630), 
            frames= self.overworld_frames['characters']['jock_white_male_spritesheet'],
            groups= self.all_sprites,
            collision_sprites = self.collision_sprites
        )
        for obj in tmx_map.get_layer_by_name('Marker'):
            if obj.name == 'Entities':
                Character(
                    pos = (obj.x,obj.y),
                    frames = self.overworld_frames['characters'][obj.properties['model']],
                    groups = (self.all_sprites,self.collision_sprites,self.character_sprites)
                )
                

        #Building Layer
        for x,y, surf in tmx_map.get_layer_by_name('buildings').tiles():
            Sprite((x * TILE_SIZE, y*TILE_SIZE), surf, self.all_sprites)
        #deco layer    
        for x,y, surf in tmx_map.get_layer_by_name('Decorations').tiles():
            Sprite((x * TILE_SIZE, y*TILE_SIZE), surf, self.all_sprites)
        
        #collision layer
        for obj in tmx_map.get_layer_by_name('BetterCollision'):
            BorderSprite(
                (obj.x,obj.y),
                pygame.Surface((obj.width, obj.height)),
                (self.collision_sprites))
    def show_calibration_screen(self, text, duration, label):
        font = pygame.font.SysFont(None, 50)
        start_time = time.time()
        self.client.current_eeg = None
        while time.time() - start_time < duration:
            # Display UI
            self.display_surface.fill((0, 0, 0))
            text_surf = font.render(text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2-50))
            self.display_surface.blit(text_surf, text_rect)
            
            # Progress bar
            progress = (time.time() - start_time)/duration
            pygame.draw.rect(self.display_surface,(255, 255, 255), (WINDOW_WIDTH//2-100, WINDOW_HEIGHT//2+50, 200, 20), 1)
            pygame.draw.rect(self.display_surface, (0, 255,0), (WINDOW_WIDTH//2-100, WINDOW_HEIGHT//2+50, 200*progress, 20))
            
            # Process data
            if self.client.current_eeg is not None:
                self.bci_calibrator.add_calibration_data_game3(self.client.current_eeg, label) #change here if needed,added function in calibration
                self.client.current_eeg = None
            
            pygame.display.flip()
            time.sleep(0.1)
    

    def run(self):
        self.client = DataClient(self.ip_address,self.server_port,self.n_chan,self.sample_rate,self.trial_samples)
        self.bci_calibrator = BCICalibrator(self.n_chan)


        loading_font = pygame.font.SysFont(None, 50)
        loading_text = loading_font.render("Loading...", True, (255, 255, 255))
        self.display_surface.fill((0, 0, 0))
        self.display_surface.blit(loading_text, (WINDOW_WIDTH // 2 - 70, WINDOW_HEIGHT // 2))
        pygame.display.flip()

        while self.command_counter < self.command_threshold:
            if self.client.current_eeg is not None:
                # client.processing_flag = True
                self.command_counter += 1
                print(f"Ignoring command {self.command_counter}/{self.command_threshold}")
                time.sleep(1)  # Simulate delay 
                
        # self.show_calibration_screen("Relax (Empty State)", 90, 'empty') #change here if want other action   
        # self.show_calibration_screen("Imagine Left Hand", 90, 'action')  
        self.show_calibration_screen("Imagine Left Hand", 90, 'lefthand')
        self.show_calibration_screen("Imagine Right Hand", 90, 'righthand')
        self.show_calibration_screen("Imagine  Feet", 90, 'feet')
        self.show_calibration_screen("Imagine Tongue", 90, 'tongue')

        try:
            self.bci_calibrator.train_game3()              #change this if needed
            # Show completion message
            self.display_surface.fill((0, 0, 0))
            font = pygame.font.SysFont(None, 50)
            text = font.render("Calibration Complete!", True, (0, 255,0))
            self.display_surface.blit(text, (WINDOW_WIDTH//2-150, WINDOW_HEIGHT//2))
            pygame.display.flip()
            time.sleep(2)
        except Exception as e:
            print(f"Calibration failed: {e}")
            self.client.close()
            return
        
        running = True
        self.client.current_eeg = None
        while running:
            dt = self.clock.tick() / 1000
            #event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                   self.client.close()
                   from flappybirdmainriemann2sec import menu_for_game as mainmenulink
                   mainmenulink()
                   running = False        

            if self.client.current_eeg is not None:
                prediction = self.bci_calibrator.predict(self.client.current_eeg)
                if prediction == 0:  # Action detected
                    # Handle movement based on prediction
                    command = 'A'  # Example: move forward
                elif prediction ==1: 
                    command = 'D'     
                elif prediction ==2: 
                    command = 'S'     
                elif prediction ==3: 
                    command = 'W'     
                self.player.update(dt, command)
                self.client.current_eeg = None       

            # game logic
            self.all_sprites.update(dt)
            self.display_surface.fill('black')
            # Calculate camera offset
            camera_offset = (
                self.player.rect.centerx - (WINDOW_WIDTH // (2 * self.ZOOM_FACTOR)),
                self.player.rect.centery - (WINDOW_HEIGHT // (2 * self.ZOOM_FACTOR))
            )

            # Clamp camera offset to map boundaries
            map_width = self.tmx_maps['world'].width * TILE_SIZE
            map_height = self.tmx_maps['world'].height * TILE_SIZE

            camera_offset = (
                max(0, min(camera_offset[0], map_width - (WINDOW_WIDTH // self.ZOOM_FACTOR))),
                max(0, min(camera_offset[1], map_height - (WINDOW_HEIGHT // self.ZOOM_FACTOR)))
            )

            self.camera_surface.fill('black')
            self.all_sprites.draw(self.camera_surface, camera_offset)
            scaled_surface = pygame.transform.scale(self.camera_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)) 
            self.display_surface.blit(scaled_surface, (0, 0))       

            pygame.display.update()
        self.client.close()

                     
if __name__ == '__main__':
    game3 = Game3()
    game3.run()