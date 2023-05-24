# Using pretrained CNNs for image classification

## 1.	Contributions
Parts of the ```generate_data()``` function were written with the help of classmates, while the ```req_functions.py``` contains a plotting function written by Ross. Beyond that, much of the work within the main script is inspired by the in-class notebooks for the weeks in which the assignment was given. ChatGPT was also used for understanding and correcting errors, though the only code contributed was for the label mapping on line 182-3 as I encountered issues in converting ```y_test``` to one-hot encoding.

## 2.	Assignment Description
"In the previous assignments involving classification, we were performing a kind of simple feature extraction on images by making them greyscale and flattening them to a single vector. This vector of pixel values was then used as the input for some kind of classification model.

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which trains a classifier on this dataset using a *pretrained CNN like VGG16*
- Save the training and validation history plots
- Save the classification report"

## 3.	Methods
The image metadata is loaded into a dataframe each for train, test, and validation data. With this, data generated is performed by creating slight alterations of the images themselves by rotation and mirroring.

The pretrained base model ```VGG16``` is then loaded without its top layers, and new top layers are defined. With learning rate set and the model compiled, the model is trained using the training and validation data for 50 epochs, with early stopping enabled if the model does not improve for three epochs in a row. The training history is plotted and a classification report is made based on test data predictions, both of which are saved to the ```out``` folder.

## 4.	Usage
Download the dataset from the link in Assignment Description and place it into the ```data``` directory. The script will expect the subfolder containing the images to be named ```dataset```.

Before running the script, the requisite packages must be installed. To do this, ensure that the current directory is ```assignment3-pretrained-cnns-nikolaimh``` and run ```bash setup.sh``` from the command line, which will update ```pip``` and install the packages in ```requirements.txt```.

From the directory above the ```data``` folder, execute ```python src/indo_fash.py``` from command line to run the script. To run the script on a fraction of the full dataset, comment lines 47-49 back in before running and determine the fraction there. The number of epochs is determined on line 156, currently set to 50.

## 5.	Discussion
In this assignment, the final training was regretfully performed on only 10% of the dataset and with fewer fully-connected layers than I would have liked. This is due mostly to external time constraints as I used UCloud to train my models and server time with the larger vCPUs has been scarce in the past few weeks, with no few crashes and stalls. Still, the fault does ultimately lie with me - I should frankly have made backup plans for this eventuality.

However, despite the limitations this approach, the model's training accuracy did reach 60%, with validation accuracy lagging a bit below that. For a spread of 15 separate classes, and with the limitations on the complexities of the model's new top layers, this is not a terrible result for a type of data (Indian fashion) that the base model was likely never trained on.

![Training History](https://github.com/AU-CDS/assignment3-pretrained-cnns-nikolaimh/blob/main/out/fashion_plot.png)

Conversely with the classification report below, which shows utterly abysmal test predictions in all categories save two, a fact which did not change no matter how much I tried to rerun it. While the issue could lie with the smaller sample size taken for the aforementioned reasons, I suspect the root of the problem is to be found in my label mapping; I had trouble converting the ```str``` class labels to the corresponding one-hot encoding because I could not figure out the order in which the model assigned label maps, as it did not appear to be resolved by my efforts to create a label conversion map. A basic issue, but there it is.

In spite of the exceptionally poor test predictions, the model did seem decently functional, though it would no doubt have been improved by more top layers,  the full dataset size, and a more proficient programmer than me to write it all.

|                    |  precision  |  recall | f1-score |  support|
|--------------------|----------|---------|---------|---------|
|              blouse|       0.84   |   0.86  |    0.85   |     44|
|           sherwanis|       0.00   |   0.00  |    0.00   |     48|
|         mojaris_men|       0.07   |   0.01  |    0.02   |     71|
|         dhoti_pants|       0.03   |   0.02  |    0.02   |     54|
|         women_kurta|       0.00   |   0.00  |    0.00   |     49|
|leggings_and_salwars|       0.44   |   0.67  |    0.53   |     54|
|       nehru_jackets|       0.06   |   0.12  |    0.08   |     41|
|           kurta_men|       0.00   |   0.00  |    0.00   |     53|
|               saree|       0.00   |   0.00  |    0.00   |     46|
|          petticoats|       0.00   |   0.00  |    0.00   |     43|
|               gowns|       0.03   |   0.04  |    0.04   |     45|
|            palazzos|       0.00   |   0.00  |    0.00   |     47|
|             lehenga|       0.00   |   0.00  |    0.00   |     52|
|            dupattas|       0.00   |   0.00  |    0.00   |     61|
|       mojaris_women|       0.02   |   0.05  |    0.03   |     42|
||
|            accuracy|              |         |    0.11   |    750|
|           macro avg|       0.10   |   0.12  |    0.11   |    750|
|        weighted avg|       0.10   |   0.11  |    0.10   |    750|
