# Quora Dataset Duplicate Search
First make sure that the following python modules are installed:

```
pip3 install --quiet "tensorflow>=1.7"
pip3 install --quiet tensorflow-hub
pip3 install --quiet seaborn
```

Next, to run the **Semantic Similarity Analysis** using **Universal Sentence Encoder** use the following:

`python Quora-Duplicate-Search.py`

Enter the number of lines to read from the CSV file for analysis (for example, 1000).

The duplicate sentences found will be written in the file `similarity-results.txt`.  
