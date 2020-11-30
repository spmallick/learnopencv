import cv2
import numpy as np
from random import choice

speed = 1

board = np.uint8(np.zeros([20, 10, 3]))

quit = False
place = False
drop = False
switch = False
hold = ""
flag = 0
border = np.uint8(127 - np.zeros([20, 1, 3]))
border_ = np.uint8(127 - np.zeros([1, 34, 3]))
next = choice(["I", "T", "L", "J", "Z", "S", "O"])
score = 0

def get_info(piece):
    if piece == "I":
        coords = np.array([[0, 3], [0, 4], [0, 5], [0, 6]])
#        color = [255, 255, 0]
        color = [255, 155, 15]
    elif piece == "T":
        coords = np.array([[1, 3], [1, 4], [1, 5], [0, 4]])
#        color = [255, 0, 127]
        color = [138, 41, 175]
    elif piece == "L":
        coords = np.array([[1, 3], [1, 4], [1, 5], [0, 5]])
#        color = [0, 127, 255]
        color = [2, 91, 227]
    elif piece == "J":
        coords = np.array([[1, 3], [1, 4], [1, 5], [0, 3]])
#        color = [255, 0, 0]
        color = [198, 65, 33]
    elif piece == "S":
        coords = np.array([[1, 5], [1, 4], [0, 3], [0, 4]])
#        color = [0, 0, 255]
        color = [55, 15, 215]
    elif piece == "Z":
        coords = np.array([[1, 3], [1, 4], [0, 4], [0, 5]])
#        color = [0, 255, 0]
        color = [1, 177, 89]
    else:
        coords = np.array([[0, 4], [0, 5], [1, 4], [1, 5]])
#        color = [0, 255, 255]
        color = [2, 159, 227]
        
    return coords, color

while not quit:
    if switch:
        hold, current = current, hold
        switch = False
    else:
        current = next
        next = choice(["I", "T", "L", "J", "Z", "S", "O"])
    
    if flag > 0:
        flag -= 1
    
    if hold == "":
        held = np.array([[0, 0]]), [0, 0, 0]
    else:
        held = get_info(hold)
    
    next_ = get_info(next)
    
    coords, color = get_info(current)
    if current == "I":
        top_left = [-2, 3]
            
    if not np.all(board[coords[:,0], coords[:,1]] == 0):
        break
    
    while True:
        dummy = board.copy()
        dummy[coords[:,0], coords[:,1]] = color
        right = np.uint8(np.zeros([20, 10, 3]))
        right[next_[0][:,0] + 2, next_[0][:,1]] = next_[1]
        left = np.uint8(np.zeros([20, 10, 3]))
        left[held[0][:,0] + 2, held[0][:,1]] = held[1]
        dummy = np.concatenate((border, left, border, dummy, border, right, border), 1)
        dummy = np.concatenate((border_, dummy, border_), 0)
        dummy = dummy.repeat(20, 0).repeat(20, 1)
        dummy = cv2.putText(dummy, str(score), (520, 200), cv2.FONT_HERSHEY_DUPLEX, 1, [0, 0, 255], 2)
        
        cv2.imshow("Tetris", dummy)
        key = cv2.waitKey(int(1000/speed))
        
        dummy = coords.copy()
        
        if key == ord("a"):
            if np.min(coords[:,1]) > 0:
                coords[:,1] -= 1
                if current == "I":
                    top_left[1] -= 1
        elif key == ord("d"):
            if np.max(coords[:,1]) < 9:
                coords[:,1] += 1
                if current == "I":
                    top_left[1] += 1
        elif key == ord("j") or key == ord("l"):
            if current != "I" and current != "O":
                if coords[1,1] > 0 and coords[1,1] < 9:
                    arr = coords[1] - 1 + np.array([[[x, y] for y in range(3)] for x in range(3)])
                    pov = coords - coords[1] + 1
            elif current == "I":
                arr = top_left + np.array([[[x, y] for y in range(4)] for x in range(4)])
                pov = np.array([np.where(np.logical_and(arr[:,:,0] == pos[0], arr[:,:,1] == pos[1])) for pos in coords])
                pov = np.array([k[0] for k in np.swapaxes(pov, 1, 2)])
            
            if current != "O":
                if key == ord("j"):
                    arr = np.rot90(arr, -1)
                else:
                    arr = np.rot90(arr)
                coords = arr[pov[:,0], pov[:,1]]
                
        elif key == ord("w"):
            drop = True
        elif key == ord("i"):
            if flag == 0:
                if hold == "":
                    hold = current
                else:
                    switch = True
                flag = 2
                break
        elif key == 8 or key == 27:
            quit = True
            break
            
        if np.max(coords[:,0]) < 20 and np.min(coords[:,0]) >= 0:
            if not (current == "I" and (np.max(coords[:,1]) >= 10 or np.min(coords[:,1]) < 0)):
                if not np.all(board[coords[:,0], coords[:,1]] == 0):
                    coords = dummy.copy()
            else:
                coords = dummy.copy()
        else:
            coords = dummy.copy()
        
        if drop:
            while not place:
                if np.max(coords[:,0]) != 19:
                    for pos in coords:
                        if not np.array_equal(board[pos[0] + 1, pos[1]], [0, 0, 0]):
                            place = True
                            break
                else:
                    place = True
                
                if place:
                    break
                
                coords[:,0] += 1
                score += 1
                if current == "I":
                    top_left[0] += 1
                    
            drop = False
            
        else:
            if np.max(coords[:,0]) != 19:
                for pos in coords:
                    if not np.array_equal(board[pos[0] + 1, pos[1]], [0, 0, 0]):
                        place = True
                        break
            else:
                place = True
            
        if place:
            for pos in coords:
                board[tuple(pos)] = color
            place = False
            break
        
        coords[:,0] += 1
        if key == ord("s"):
            score += 1
        if current == "I":
            top_left[0] += 1
    
    lines = 0
    
    for line in range(20):
        if np.all([np.any(pos != 0) for pos in board[line]]):
            lines += 1
            board[1:line+1] = board[:line]
            
    if lines == 1:
        score += 40
    elif lines == 2:
        score += 100
    elif lines == 3:
        score += 300
    elif lines == 4:
        score += 1200

print(score)
