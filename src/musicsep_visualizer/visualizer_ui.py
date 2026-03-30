import pygame
from OpenGL.GL import *
from typing import Optional

class Sidebar:
    def __init__(self, width_px=250, bg_color=(20, 20, 25), text_color=(220, 220, 220)):
        self.width_px = width_px
        self.bg_color = bg_color
        self.text_color = text_color
        
        self.texture = None
        self.font = None
        self.surface = None
        self._needs_update = True
        self.content_text = "Hello world"
        
        self.hook_names = []
        self.active_index = 0
        self.item_rects = [] # List of (rect, index)
        self.show_play_button = False
        self.play_button_rect: Optional[pygame.Rect] = None
        
        # Cache for window size to detect resizes
        self.last_win_size = (0, 0)

    def _init_resources(self):
        if self.font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            # Try to get a nice monospaced font, fallback to system default
            self.font = pygame.font.SysFont('Consolas', 16) # Or 'Courier New', 'Monospace'
            if not self.font:
                 self.font = pygame.font.SysFont(None, 20)
        
        if self.texture is None:
            self.texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            # UI textures need linear filtering for smoother text, or nearest for retro look
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def cleanup(self):
        if self.texture:
            try:
                glDeleteTextures([self.texture])
            except Exception:
                pass
            self.texture = None

    def update_content(self, info_dict):
        """
        Update the sidebar content.
        info_dict: Dictionary or string to display.
        """
        # For now, just simplistic text handling as requested + extension
        new_text = str(info_dict)
        if new_text != self.content_text:
            self.content_text = new_text
            self._needs_update = True

    def set_hooks(self, names, active_index):
        if self.hook_names != names or self.active_index != active_index:
            self.hook_names = names
            self.active_index = active_index
            self._needs_update = True

    def set_play_button_visible(self, visible: bool):
        if self.show_play_button != visible:
            self.show_play_button = visible
            self._needs_update = True

    def handle_click(self, x, y):
        """
        Check if a click at (x, y) hit any interactive sidebar element.
        Returns:
            ("play_waveform", None) if play button was clicked
            ("select_hook", index) if a hook list item was clicked
            None for no hit
        """
        # x, y are local to the sidebar surface (0,0 at top-left)
        if self.play_button_rect and self.play_button_rect.collidepoint(x, y):
            return ("play_waveform", None)

        for rect, idx in self.item_rects:
            if rect.collidepoint(x, y):
                return ("select_hook", idx)

        return None

    def _update_surface_and_texture(self, win_w, win_h):
        # Create surface if needed or resized
        if self.surface is None or self.surface.get_size() != (self.width_px, win_h):
            self.surface = pygame.Surface((self.width_px, win_h))
            self._needs_update = True
        
        if self._needs_update:
            self.surface.fill(self.bg_color)
            self.item_rects = []
            self.play_button_rect = None
            
            # Simple text rendering
            # Split lines if it's a multiline string
            lines = self.content_text.split('\n')
            y_offset = 20
            x_offset = 15
            
            # Title
            title_surf = self.font.render("Status", True, (255, 200, 100))
            self.surface.blit(title_surf, (x_offset, y_offset))
            y_offset += 30
            
            for line in lines:
                text_surf = self.font.render(line, True, self.text_color)
                self.surface.blit(text_surf, (x_offset, y_offset))
                y_offset += 20

            y_offset += 12
            button_h = 34
            button_rect = pygame.Rect(15, y_offset, self.width_px - 30, button_h)
            if self.show_play_button:
                pygame.draw.rect(self.surface, (50, 88, 120), button_rect, border_radius=6)
                pygame.draw.rect(self.surface, (98, 148, 194), button_rect, width=1, border_radius=6)
                button_label = self.font.render("Play Waveform", True, (240, 245, 250))
                label_rect = button_label.get_rect(center=button_rect.center)
                self.surface.blit(button_label, label_rect)
                self.play_button_rect = button_rect
            y_offset += button_h

            # Draw Hook List
            y_offset += 20
            list_title = self.font.render("Tensors (Click to select)", True, (255, 200, 100))
            self.surface.blit(list_title, (x_offset, y_offset))
            y_offset += 30

            for i, name in enumerate(self.hook_names):
                is_active = (i == self.active_index)
                color = (255, 255, 255) if is_active else (150, 150, 150)
                if is_active:
                    # Draw highlight background
                    bg_rect = pygame.Rect(5, y_offset - 2, self.width_px - 10, 20)
                    pygame.draw.rect(self.surface, (50, 50, 80), bg_rect)
                
                # Truncate if too long
                display_name = name
                if len(display_name) > 25:
                    display_name = display_name[:22] + "..."

                item_surf = self.font.render(display_name, True, color)
                item_rect = self.surface.blit(item_surf, (x_offset, y_offset))
                
                # Store full width rect for easier clicking
                click_rect = pygame.Rect(0, y_offset, self.width_px, 20)
                self.item_rects.append((click_rect, i))
                
                y_offset += 20
                
            # Upload to texture
            # "RGB" format, flip=True (1) because OpenGL is bottom-left origin
            data = pygame.image.tostring(self.surface, "RGB", True)
            
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width_px, win_h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
            
            self._needs_update = False

    def draw(self, display_size):
        """
        Draws the sidebar on the left side of the screen.
        Returns the width of the sidebar in pixels, so the main view can offset itself.
        """
        self._init_resources()
        
        win_w, win_h = display_size
        if win_w == 0 or win_h == 0:
            return 0
            
        self._update_surface_and_texture(win_w, win_h)
        
        # Calculate NDC coordinates
        # Screen X spans [-1, 1]
        # Pixel width to NDC width: (px / win_w) * 2
        ndc_w = (self.width_px / win_w) * 2
        
        x1 = -1.0
        x2 = -1.0 + ndc_w
        y1 = -1.0
        y2 = 1.0
        
        # Save previous state if necessary (mostly handled by caller resetting identity)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glColor3f(1.0, 1.0, 1.0)
        
        # Draw Quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x1, y1) # Bottom Left
        glTexCoord2f(1, 0); glVertex2f(x2, y1) # Bottom Right
        glTexCoord2f(1, 1); glVertex2f(x2, y2) # Top Right
        glTexCoord2f(0, 1); glVertex2f(x1, y2) # Top Left
        glEnd()
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return self.width_px
