# FlairDetectionReddit
Automatic Flair Detection on [r/India](https://www.reddit.com/r/india "r/India") subreddit - A Data Science approach

### Description
The flair detector classifies given reddit post links from r/India into one of 11 flairs: 'AskIndia', 'Non-Political', 'Scheduled', 'Photography', 'Science/Technology', 'Politics', 'Business/Finance', 'Policy/Economy', 'Sports', 'Food', 'Coronavirus'.

It has been developed using Python and deployed using Flask.

The [live demo](https://reddit-flair-detector-shash.herokuapp.com/ "live demo") has been deployed on Heroku.

The training set consisted of 70 examples from each flair. On a testing set of 30 examples from each flair, the model achieved 71.8% accuracy. Detailed analysis is available in the Jupyter Notebooks.

### GitHub Structure
Data - Consists of 4 Jupyter Notebooks, the final model and the training/validation dataset used.
- getdata.ipynb - Data extraction from Reddit (Part I);
- EDA and Preprocessing.ipynb - Exploratory Data Analysis (Part 2);
- Model.ipynb - Cleaning text data and trying different classification algorithms (Part 3);
- Final Model - Logistic Regression.ipynb - Final file to dump a pickle model (Part 4);

Website - Consists of the website materials required for deploying to Heroku.

requirements.txt - Requirements for running the website and the Jupyter Notebooks

example.txt - A sample txt file to exemplify how to do batch url predictions.

### Reproducing the Project

- Clone the repository
- Run pip install -r requirements.txt in your shell
- You can then run the python notebook code cells in the Data folder
- To run the flask app go in the Website folder:
```bash
$ export FLASK_APP=hello.py
$ flask run
```

If you face any problems feel free to contact me. I would appreciate any feedback :D

### References
**Part 1** - Data Extraction
- https://praw.readthedocs.io/en/latest/
- https://towardsdatascience.com/scraping-reddit-data-1c0af3040768

**Part 2** - Exploratory Data Analysis
- https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
- https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
- https://realpython.com/python-keras-text-classification/

**Part 3** - Trying different classification algorithms
- https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
- https://medium.com/text-classification-algorithms/text-classification-algorithms-a-survey-a215b7ab7e2d

**Part 4, 5** - Building a Web Application and deploying on Heroku
- https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b


------------

