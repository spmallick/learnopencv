import cv2
import numpy as np
from utils import *

movie_csv = "movies-list-short.csv"
canvas = "blank-canvas.png"

movies_data = read_from_csv(movie_csv)

movie, movie_info = get_movie_info(movies_data)

print(movie)

hints,labels = select_hints(movie_info)

img = get_canvas(canvas)

char_rects = get_char_coords(movie)

img = draw_blank_rects(movie,char_rects,img)

cv2.namedWindow("Hangman", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hangman",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

cv2.imshow("Hangman",img)

chars_entered = []

incorrect_attempts = 0

img_copy = img.copy()

while 1:
    img = img_copy.copy()
    img = draw_hint(img,hints,labels,incorrect_attempts)
    if incorrect_attempts >= 6:
        img = draw_lost(img)
        break
    elif check_all_chars_found(movie, chars_entered):
        img = draw_won(img)
        break
    else:
        letter = cv2.waitKey(0) & 0xFF
        if letter < 65 or letter > 122 or (letter > 90 and letter < 97):
            img = draw_invalid(img)
            cv2.imshow("Hangman",img)
            continue
        else:
            letter = chr(letter).upper()
        if letter in chars_entered:
            img = draw_reuse(img)
            img = draw_used_chars(img,chars_entered,letter)
            cv2.imshow("Hangman",img)
            continue
        else:
            chars_entered.append(letter)
            if letter in movie:
                img = draw_right(img)
                img = displayLetter(img,letter,movie,char_rects)
                img_copy = displayLetter(img_copy,letter,movie,\
                        char_rects)
            else:
                img = draw_wrong(img,incorrect_attempts)
                incorrect_attempts += 1
    img = draw_used_chars(img,chars_entered,letter)
    img = draw_hangman(img,incorrect_attempts)
    img_copy = draw_used_chars(img_copy,chars_entered,letter)
    img_copy = draw_hangman(img_copy,incorrect_attempts)
    cv2.imshow("Hangman",img)
    #cv2.waitKey(0)

img = revealMovie(movie,img,char_rects)
cv2.imshow("Hangman",img)
cv2.waitKey(0)

cv2.destroyAllWindows()
