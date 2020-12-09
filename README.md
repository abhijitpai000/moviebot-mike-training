# Movibot Mike - Model Training
This repository holds the model training part of the moviebot-mike application from the [moviebot-mike](https://github.com/abhijitpai000/moviebot-mike) repo.

## Data Source
I'm using [The Indian Movie Database](https://www.kaggle.com/pncnmnp/the-indian-movie-database?select=2010-2019) dataset from Kaggle for the Movie Recommendation part of the project.


## Setup instructions

1. Clone/Fork this repo

2. Navigate to cloned path.

3. Setup virtual envinorment.

    `pip install virtualenv`

    `virtualenv myenv`

4. Activate it by running & Install project requirements.

    `myenv/Scripts/activate`

    `pip install -r requirements.txt`
    
## Train Model using Terminal.

1. For Movie recommendation system training dowload the .zip file from [The Indian Movie Database](https://www.kaggle.com/pncnmnp/the-indian-movie-database?select=2010-2019) link and place it in 'movie_recsys/datasets' directory.

2. Run the following command in the terminal.

    `python run.py`
    
3. Check for following files after Step 2.
* 'learned_parameters.pth' in the intent_classifier/datasets directory.
* 'movie_data.csv' & 'cosine_sim_scores.csv' in the movie_recsys/datasets directory.
    
