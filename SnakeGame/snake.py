import cv2
import numpy as np
from random import randint
from random import choice


class SnakePart:
  def __init__(self, front, x, y):
    self.front = front
    self.x = x
    self.y = y

  def move(self):
    # Moves by following the part in front of it
    self.x = self.front.x
    self.y = self.front.y

class Head:
  def __init__(self, direction, x, y):
    self.direction = direction
    self.x = x
    self.y = y

  def move(self):
    # Checks what its current direction is and moves accordingly
    if self.direction == 0:
        self.x += 1
    elif self.direction == 1:
        self.y += 1
    elif self.direction == 2:
        self.x -= 1
    elif self.direction == 3:
        self.y -= 1

def display():

  # Create a blank image
  board = np.zeros([BOARD_SIZE, BOARD_SIZE, 3])

  # Color the snake green
  for part in snake:
    board[part.y, part.x] = [0, 255, 0]
  
  # Color the apple red
  board[appley, applex] = [0, 0, 255]
  
  # Display board
  cv2.imshow("Snake Game", np.uint8(board.repeat(CELL_SIZE, 0).repeat(CELL_SIZE, 1)))
  key = cv2.waitKey(int(1000/SPEED))
  
  # Return the key pressed. It is -1 if no key is pressed. 
  return key
  
def win_focus():
  # Ugly trick to get the window in focus.
  # Opens an image in fullscreen and then back to normal window
  cv2.namedWindow("Snake Game", cv2.WINDOW_AUTOSIZE);
  board = np.zeros([BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE, 3])
  cv2.imshow("Snake Game", board);
  cv2.setWindowProperty("Snake Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
  cv2.waitKey(2000)
  cv2.setWindowProperty("Snake Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)


if __name__ == '__main__' : 

  # Size of each cell in the board game
  CELL_SIZE = 20
  # Number of cells along the width in the game
  BOARD_SIZE = 45
  # Change SPEED to make the game go faster
  SPEED = 12
  # After eating an apple the snake grows by GROWTH units
  GROWTH = 3

  
  # Variable to track if apple is eaten
  eaten = True
  # Variable to check if the game should quit
  quit = False
  # Variable to track growth. 
  grow = 0
  
  # Array for storing snake
  snake = []

  #snake's head starts at the center of the board. 
  head = Head(0, int((BOARD_SIZE - 1)/2), int((BOARD_SIZE - 1)/2))
  snake.append(head)

  # Start the game by printing instructions
  print('w = top, a = left, s = down, d = right')
  # Ugly trick to bring the window in focus
  win_focus()

  while True:

    # Checks if the apple is eaten and generates a new one
    if eaten:
      # Create a list of all possible locations
      s = list(range(0, BOARD_SIZE ** 2))
      # Delete cells that are part of the snake
      for part in snake:
          s.remove(part.x * BOARD_SIZE + part.y)
      
      # Randomly pick from one of the remaining cells    
      a = choice(s)
      applex = int(a/BOARD_SIZE)
      appley = a % BOARD_SIZE
      
      eaten = False

    # Makes and displays the board
    key = display()

    
    # Gets key presses and moves accordingly
    # 8 and 27 are delete and escape keys
    # Arrow keys are tricky in OpenCV. So we use
    # keys 'w', 'a','s','d' for movement. 
    # w = top, a = left, s = down, d = right


    if key == 8 or key == 27:
      break
    elif key == ord("d") :
      head.direction = 0
    elif key == ord("s") :
      head.direction = 1
    elif key == ord("a") :
      head.direction = 2
    elif key == ord("w") :
      head.direction = 3

    # Moving the snake
    for part in snake[::-1]:
      part.move()    

    # Collision rules

    if head.x < 0 or head.x > BOARD_SIZE - 1 or head.y < 0 or head.y > BOARD_SIZE - 1:
      break
        
    for part in snake[1:]:
      if head.x == part.x and head.y == part.y:
        quit = True
        break
        
    if quit:
      break
      
    # The snake grows graduallly over multiple frames    
    if grow > 0:
      snake.append(SnakePart(snake[-1], subx, suby))
      grow -= 1
      
    # Grows the snake when it eats an apple
    if applex == head.x and appley == head.y:
      subx = snake[-1].x
      suby = snake[-1].y
      eaten = True
      grow += GROWTH

