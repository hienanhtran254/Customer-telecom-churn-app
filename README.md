# Customer telecom churn

For telecom businesses, it is important to maintain existing customers with their service. Therefore, it is the key challenge for the companies to identify and predict customer churn rate so they would be able to point out customer problems and resolve them to ensure that customers will continue using their service.

## 1. Step to design interactive app 
### Model building:
- Performed hyperparameter tuning by using library Caret for Decision Tree and Random Forest models.
- Created makeTree and makeForest functions to train data and predicted them in useTree and useForest functions. 
- Calculated score of accuracy, TPR, and TNR based on the previous prediction. 
### Server logic function:
- Rendered data table as a preview
- Hyperparameter tuning (“number" and “repeats" control as input, and output would be tuning results and variables important plot)
- Build decision tree and random forest model (input would be “features" and “controls" of each models, and output would be the accuracy score and plot of each models)
### User interface part:
- Hello Shiny! (a welcome page, which guide users which steps they should take )
- About (an introduction about the dataset)
- Hyperparameter (tuning hyperparameter to search for the ideal model architecture)
- Model (Decision Tree, Random Forest) (shows results of accuracy and final plots)

## 2. Explore analysis
The most influential variables in the decision tree are international plan, total minutes per day (domestic), total charge per day (domestic) and total international minutes. While the total minutes per day (domestic), total charge per day (domestic) and number of times that customer service calls are three most important variables for random forest. Moreover, in the random forest, we tuned that the best number of features used at each split is 18 ( 3 folds, repeated 5 times), which are all the features. This indicates that all the features are relatively uncorrelated to each other.

For some certain parameters, the training accuracy of the tree is 93.45%, while that of the random forest is 92.59%. This matches the fact that trees have lower bias compared to the corresponding forests. Moreover, the testing accuracy of the tree is 87.92%, while that of the random forest is 91.28%. This indicates that forests have lower variance compared to the corresponding trees. In this case, we focus on the customer who wants to discontinue the service or true positive rate. The accuracy and positive rate of the test result are higher in the random forest, which are 91.28% and 84.62%,  than the single decision tree, which are 87.92% and 70%. Therefore, we recommend using the model of random forest.
