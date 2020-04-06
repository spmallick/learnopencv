# Hangman-OpenCV
This repository contains the code for Hangman game made using OpenCV library in Python.

## Outline
1. Use **pandas** library to process the **IMDb database**.
- Remove the columns which are not required. This has been done by removing the columns which are required from the list of all column headings. We are keeping **`release_year`**, **`cast`**, **`director`**, **`keywords`** and **`tagline`** as 5 features of a movie.
- All the rows having at least one missing value are removed using `dropna` function from Pandas.
- We only keep the rows which have english alpha numerical movie title.
- We remove all the rows which have `,` in the movie title or any of the 5 features. This has been done to make sure that the commas are present only to separate the data across columns in CSV file.
- We only keep movies with titles less than or equal to 20 and tagline length less than or equal to 30. 
- Save the final database to a CSV file using `to_csv` function from Pandas.

2. Read the dataset from the CSV file and store it in dictionary format: `movie_title:[year,list of keywords, tagline, director, list of cast]`- **`read_from_csv`**

3. Get a random movie from the list of movies and get all the information (5 features) for that movie - **`get_movie_info`**

4. From the 5 movie features, select any 3 features. If the features have list of keywords and/or list of cast, randomly select one from the list - **`select_hints`**

5. Read the hangman template - **`get_canvas`**

6. Get the points where each blank rectanglular box will be drawn depending on the maximum width and height among all the characters present in the movie name - **`get_char_coords`**

7. Draw the blank rectangular boxes - **`draw_blank_rects`**

8. While the number of incorrect attempts is less than 6 and the game hasn't been won or lost yet:
- Take character input from user
- If the character is invalid (not an alphabet), display `INVALID CHARACTER`
- Else if the character has already been entered, display `ALREADY USED`
- Else if the character is NOT present in movie title, display `WRONG`, increment number of incorrect attempts, display another body part in Hangman, and display the character in `Letters used`
- Else, display `CORRECT`, display the letters in the blank rectangles and display the character in `Letters used`
- If all characters in movie have been guessed, break the loop and display `YOU WON`
- If number of incorrect attempts are equal to or more than 6, break the loop and display `YOU LOST`

9. Reveal the movie name at the end of the game.

10. Wait for the user to press a key and quit.

**The entire game will run on full screen by default**.

## Instructions
`python main.py`

## Requirements
1. OpenCV 3.4
2. Pandas
3. Numpy



# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
