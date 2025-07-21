from settings import *

class Entity(pygame.sprite.Sprite):
    def __init__(self,pos,frames, groups):
        super().__init__(groups)

        #graphics
        self.frame_index, self.frames= 0, frames
        #self.starting_animation = 'idle'

        #movements
        self.direction = vector()
        self.speed = 50

        #sprite setup
        self.image = self.frames[self.get_state()][self.frame_index]
        self.rect = self.image.get_frect(center = pos)
        self.hitbox = self.rect.inflate(-self.rect.width / 2, -2)
    
    def animate(self,dt):
        self.frame_index += ANIMATION_SPEED * dt
        self.image = self.frames[self.get_state()][int(self.frame_index % len(self.frames[self.get_state()]))]

    def get_state(self):
        moving = bool(self.direction)
        return 'walk' if moving else 'idle'

class Character(Entity):
    def __init__(self, pos, frames, groups):
        super().__init__(pos, frames, groups)



class Player(Entity):
    def __init__(self, pos,frames, groups,collision_sprites):
        super().__init__(pos,frames,groups)
        self.collision_sprites = collision_sprites

    def update_direction_from_bci(self, command):
        input_vector2 = vector()
        if command == 'W':
            input_vector2.y -=10
        if command == 'S':
            input_vector2.y +=10
        if command == 'A':
            input_vector2.x -=10  
        if command == 'D':
            input_vector2.x +=10
        self.direction = input_vector2
     #   self.direction = input_vector2.normalize() if input_vector2 else input_vector2 

    def input(self):
        keys = pygame.key.get_pressed()
        input_vector = vector()
        if keys[pygame.K_w]:
            input_vector.y -= 1
        if keys[pygame.K_s]:
            input_vector.y += 1
        if keys[pygame.K_a]:
            input_vector.x -= 1  
        if keys[pygame.K_d]:
            input_vector.x += 1

        self.direction = input_vector.normalize() if input_vector else input_vector 

    def move(self,dt):
        self.rect.centerx += self.direction.x * self.speed * dt
        self.hitbox.centerx = self.rect.centerx
        self.collision('horizontal')

        self.rect.centery += self.direction.y * self.speed * dt
        self.hitbox.centery = self.rect.centery
        self.collision('vertical')

    def collision(self, axis):
        for sprite in self.collision_sprites:
            if sprite.hitbox.colliderect(self.hitbox):
                if axis == 'horizontal':
                    if self.direction.x > 0:
                        self.hitbox.right = sprite.hitbox.left
                    if self.direction.x < 0:
                        self.hitbox.left = sprite.hitbox.right
                    self.rect.centerx = self.hitbox.centerx
                else:
                    if self.direction.y >0:
                        self.hitbox.bottom = sprite.hitbox.top
                    if self.direction.y <0:
                        self.hitbox.top = sprite.hitbox.bottom
                    self.rect.centery = self.hitbox.centery    
    
    def update(self, dt,bci_command=None):
        self.input()
        if bci_command:
            self.update_direction_from_bci(bci_command)
        self.move(dt)
        self.animate(dt)       
               
