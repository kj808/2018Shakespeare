# Shakespeare

Shakespeare is well known for his plays. In all the created plays, multiple characters exist with various lines. Using text and line data from Shakespeare's plays, can the character be classified? To further breakdown the problem, certain characters appear in only certain plays. This provides better dependency for character detection. Furthermore, characters only appear in certain scenes. By parsing the act/scene line, it's possible to take this into consideration. Lastly, certain terms are more frequent among certain players compared to other players. With these features, it is possible to correctly classify.

Location: Data Science Graduate Course at KU \\
Dates: Sep 20, 2018 to Sep 24, 2018

## Goal
* Determine preparedness of Massachuesettes high school students for U.S. universities
* Determine reliability of  Massachuesettes' Accountability and Assistance level as an indicator of preparedness

## Data
1. [Shakespeare's Plays](https://www.kaggle.com/kingburrito666/shakespeare-plays). A list of Shakespeare's plays with lines and the character who delivered those lines. 


## Process Techniques
In this project...
* I performed data rangling on the two data sets (i.e., normalized features, combined data sets, removing unrelevant entries, randomizing entries) 
* I split lines into training, testing and validation based on the 80/20 rule
* I built two classifiers are used: Random Forest and Naive Bayes. 

## Results
For determining players based on lines in a play, the best features include the scene, act, player number and term frequencies. In terms of model, since these features consist of frequency data and discrete values, random forests work best especially considering the amount of rows. Naive Bayes may work, but the current features need readjusting to all frequency data instead of a combination of both.

## Repository Contents

| Directory | Description |
| --- | ----------- |
| Data | Contains all of the datasets used in this project. |
| Libraries | If libraries are used, the exact distribution will be located here. Includes library, library name, and library version. |
| Models | Models generated for the project such as machine learning models. |
| Notebooks | Notebooks used for visualing the data. |
| Reports | The resulting reports on this project. |
| Src | Source scripts and other helper files located here. |


