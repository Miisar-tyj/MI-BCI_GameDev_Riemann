from settings import *

class AllSprites(pygame.sprite.Group):
    def __init__(self):
        super().__init__()

    def draw(self, surface, camera_offset=(0, 0)):
        for sprite in self.sprites():
            # Adjust sprite position based on camera offset
            offset_pos = sprite.rect.topleft[0] - camera_offset[0], sprite.rect.topleft[1] - camera_offset[1]
            surface.blit(sprite.image, offset_pos)

#class AllSprites(pygame.sprite.Group):
#    def __init__(self):
 #       super().__init__()
  #      self.display_surface = pygame.display.get_surface()
   #     self.offset = vector()

    #def draw(self,surface, player_center):
     #   self.offset.x = -(player_center[0] - WINDOW_WIDTH / 2)
      #  self.offset.y = -(player_center[1] - WINDOW_HEIGHT / 2)
       # for sprite in self:
        #    self.display_surface.blit(sprite.image, sprite.rect.topleft + self.offset)    
