# Universal Sentence Encoder

**This repository contains code for [Universal Sentence Encoder](https://www.learnopencv.com/universal-sentence-encoder/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/qa21gf9k1z88wlu/AAC2YN02UF4lC8MgHs-22AtDa?dl=1)

## Instructions
First install the dependencies.

```
pip3 install --quiet "tensorflow>=1.7"
pip3 install --quiet tensorflow-hub
pip3 install --quiet seaborn
```

To run the **message encoder example**, use:
`python3 embedMessages.py`

To run the **Semantic Similarity Analysis on Avengers:Infinity Warcast**, use:
`cd Avengers-Similarity-Analysis`

`python3 process-script.py`
This will process the raw script and remove all the text contained in brackets.

Next, we have to extract the dialogues of the characters.

`python3 get-character-lines.py`

Finally, use the Universal Sentence Encoder for running Semantic Similarity Analysis.

`python3 universal-sentence-encoder.py`

This will display and save the Similarity Matrix as **Avenger-semantic-similarity.png**.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
