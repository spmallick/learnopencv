import cv2
import numpy as np
import sys

def read_from_csv(csv_f):
    with open(csv_f,'r') as f:
        movie_data = {}
        for line in f.readlines():
            line_split = line.strip().split(",")
            year = line_split[-1].split("|")
            keywords = line_split[-2].split("|")
            tagline = line_split[-3].split("|")
            director = line_split[-4].split("|")
            cast = line_split[-5].split("|")
            movie = line_split[0].upper()
            movie_data[movie] = [year,keywords,tagline,director,cast]
    return movie_data

def get_movie_info(movies_data):
    movies_list = list(movies_data.keys())
    movie = np.random.choice(movies_list,1)[0].upper()
    movie_info = movies_data[movie]
    return movie,movie_info

def select_hints(movie_info):
    # We will randomly select 3 types of
    # hints to display
    hints_index = list(np.random.choice(5,3,replace=False))
    hints = []
    hints_labels = ["Release Year","Keyword","Tagline","Director","Cast"]
    labels = []
    for hint_index in hints_index:
        hint = np.random.choice(movie_info[hint_index],1)[0].upper()
        hints.append(hint)
        labels.append(hints_labels[hint_index].upper())
    return hints,labels

def get_canvas(canvas_file):
    img = cv2.imread(canvas_file,1)
    return img

def draw_wrong(img,incorrect_attempts):
    cv2.putText(img,"WRONG {}/6".format(incorrect_attempts+1),(380,40),\
            cv2.FONT_HERSHEY_SIMPLEX,1,\
            (0,0,255),2)
    return img

def draw_hint(img,hints,labels,incorrect_attempts):
    x,y = 20,30
    if incorrect_attempts == 0:
        return img
    elif incorrect_attempts <= 1:
        index = 0
    elif incorrect_attempts <= 3:
        index = 1
    elif incorrect_attempts <= 6:
        index = 2
    cv2.putText(img,"HINT: {}".format(labels[index]),(x,y),\
            cv2.FONT_HERSHEY_SIMPLEX,0.6,\
            (255,0,255),1)
    cv2.putText(img,"{}".format(hints[index]),(x,y+30),\
            cv2.FONT_HERSHEY_SIMPLEX,0.6,\
            (255,0,255),1)
    return img

def draw_right(img):
    cv2.putText(img,"RIGHT",(380,40),\
            cv2.FONT_HERSHEY_SIMPLEX,0.7,\
            (0,255,0),2)
    return img

def draw_lost(img):
    cv2.putText(img,"YOU LOST",(380,40),\
            cv2.FONT_HERSHEY_SIMPLEX,0.7,\
            (0,0,255),2)
    return img

def draw_won(img):
    cv2.putText(img,"YOU WON",(380,40),\
            cv2.FONT_HERSHEY_SIMPLEX,0.7,\
            (0,255,0),2)
    return img

def draw_invalid(img):
    cv2.putText(img,"INVALID INPUT",(300,40),\
            cv2.FONT_HERSHEY_SIMPLEX,0.7,\
            (0,0,255),2)
    return img

def draw_reuse(img):
    cv2.putText(img,"ALREADY USED",(300,40),\
            cv2.FONT_HERSHEY_SIMPLEX,0.7,\
            (0,0,255),2)
    return img

def draw_used_chars(img,chars_entered,letter):
    cv2.putText(img,"Letters used:",(300,80),\
            cv2.FONT_HERSHEY_SIMPLEX,0.5,\
            (0,0,0),1)
    y = 120
    x = 350
    count = 0
    for i in chars_entered:
        if count == 10:
           x += 50
           y = 120
        if i==letter:
           cv2.putText(img,i,(x,y),\
                cv2.FONT_HERSHEY_SIMPLEX,0.5,\
                (0,0,255),1)
        else:
           cv2.putText(img,i,(x,y),\
                cv2.FONT_HERSHEY_SIMPLEX,0.5,\
                (0,0,0),1)
        y += 20
        count += 1
    return img

def get_char_coords(movie):
    x_coord = 100
    y_coord = 400

    char_ws = []
    char_hs = []

    for i in movie:
        char_width, char_height = cv2.getTextSize(i,\
                cv2.FONT_HERSHEY_SIMPLEX,1,2)[0]
        char_ws.append(char_width)
        char_hs.append(char_height)

    max_char_h = max(char_hs)
    max_char_w = max(char_ws)

    char_rects = []

    for i in range(len(char_ws)):
        rect_coord = [(x_coord,y_coord-max_char_h),\
                (x_coord+max_char_w,y_coord)]
        char_rects.append(rect_coord)
        x_coord = x_coord + max_char_w

    return char_rects

def draw_blank_rects(movie,char_rects,img):

    for i in range(len(char_rects)):
        top_left, bottom_right = char_rects[i]
        if not movie[i].isalpha() or \
                ord(movie[i]) < 65 or \
                ord(movie[i]) > 122 or \
                (ord(movie[i]) > 90 and \
                ord(movie[i]) < 97):
            cv2.putText(img,movie[i],(top_left[0],\
                    bottom_right[1]),\
                    cv2.FONT_HERSHEY_SIMPLEX,\
                    1,(0,0,255),2)
            continue
        cv2.rectangle(img,top_left,\
                bottom_right,\
                (0,0,255),thickness=1,\
                lineType = cv2.LINE_8)

    return img

def check_all_chars_found(movie, chars_entered):
    chars_to_be_checked = [i for i in movie if i.isalpha()]
    for i in chars_to_be_checked:
        if i not in chars_entered:
            return False
    return True

def draw_circle(img):
    cv2.circle(img,(190,160),40,(0,0,0),thickness=2,\
            lineType=cv2.LINE_AA)
    return img

def draw_back(img):
    cv2.line(img,(190,200),(190,320),\
            (0,0,0),thickness=2,\
            lineType=cv2.LINE_AA)
    return img

def draw_left_hand(img):
    cv2.line(img,(190,240),(130,200),\
            (0,0,0),thickness=2,\
            lineType=cv2.LINE_AA)
    return img

def draw_right_hand(img):
    cv2.line(img,(190,240),(250,200),\
            (0,0,0),thickness=2,\
            lineType=cv2.LINE_AA)
    return img

def draw_left_leg(img):
    cv2.line(img,(190,320),(130,360),\
            (0,0,0),thickness=2,\
            lineType=cv2.LINE_AA)
    return img

def draw_right_leg(img):
    cv2.line(img,(190,320),(250,360),\
            (0,0,0),thickness=2,\
            lineType=cv2.LINE_AA)
    return img

def draw_hangman(img,num_tries):
    if num_tries==1:
        return draw_circle(img)
    elif num_tries==2:
        return draw_back(img)
    elif num_tries==3:
        return draw_left_hand(img)
    elif num_tries==4:
        return draw_right_hand(img)
    elif num_tries==5:
        return draw_left_leg(img)
    elif num_tries==6:
        return draw_right_leg(img)
    else:
        return img

def revealMovie(movie,img,char_rects):
    #img = cv2.imread(canvas_file,1)
    for i in range(len(movie)):
        top_left, bottom_right = char_rects[i]
        cv2.putText(img,movie[i],(top_left[0],bottom_right[1]),\
                cv2.FONT_HERSHEY_SIMPLEX,\
                1,(0,255,0),2)
    return img

def displayLetter(img,letter,movie,char_rects):
    for i in range(len(movie)):
        if movie[i]==letter:
            top_left, bottom_right = char_rects[i]
            cv2.putText(img, movie[i],\
                    (top_left[0],bottom_right[1]),\
                    cv2.FONT_HERSHEY_SIMPLEX,\
                    1,(255,0,0),2)
    return img
