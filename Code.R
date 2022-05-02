
#setwd("/Users/hienanh/Desktop/Cass/T3-SMM636 Machine Learning/GC")

# LOAD PACKAGES
library(methods)       # large scale analytics
library(rpart)         # creating decision trees
library(rpart.plot)    # plotting decision trees
library(randomForest)  # creating random forest
library(shiny)         # user interface creation
library(shinyWidgets)  # better Shiny UI widgets
library(caret)         # for splitting data
library(DT)            # build DataTables in R

# DATA
## Import data
raw_data <- read.csv('telecom_churn_sample.csv')
data <- data.frame(raw_data)

## check missing values
sum(is.na(data))

## remove columns
data <- subset(data, select = -c(X,State))
colnames(data)

## changing characteristic variables into numerical 
data$International.plan = as.factor(data$International.plan)
data$Voice.mail.plan = as.factor(data$Voice.mail.plan)
data$Churn = as.factor(data$Churn)

## create training/test split
set.seed(345)
train.index=createDataPartition(data[,ncol(data)],p=0.7,list=FALSE)
train=data[train.index,]
test=data[-train.index,]

# VARIABLE LISTS
variable_list = list(
  "Account length"        = "Account.length",     
  "Area code"             = "Area.code",     
  "International plan"    = "International.plan",     
  "Voice mail plan"       = "Voice.mail.plan",     
  "Number vmail messages" = "Number.vmail.messages",
  "Total day minutes"     = "Total.day.minutes",
  "Total day calls"       = "Total.day.calls",     
  "Total day charge"      = "Total.day.charge" ,
  "Total evening minutes" = "Total.eve.minutes", 
  "Total evening calls"   = "Total.eve.calls", 
  "Total evening charge"  = "Total.eve.charge"  ,  
  "Total night minutes"   = "Total.night.minutes"  ,
  "Total night calls"     = "Total.night.calls"  ,
  "Total night charge"    = "Total.night.charge"  ,  
  "Total international minutes" = "Total.intl.minutes"    ,
  "Total international calls"   = "Total.intl.calls"  , 
  "Total international charge"  = "Total.intl.charge"   ,
  "Customer service calls" = "Customer.service.calls"
)

variable_list_default = list("Total.day.minutes", 
                             "Total.eve.minutes",
                             "International.plan",
                             "Total.night.minutes",
                             "Total.intl.minutes",
                             "Customer.service.calls"
)

# BUILD FUNCTION
## use Caret to tune paramter
fitControl = function(number, repeats){
  fitControl=trainControl(
    method = "repeatedcv",
    number = number,
    repeats = repeats)
  return(fitControl)
}

tuneTree = function(model_vars, fitControl){
  f = paste("Churn ~ ", paste(model_vars, collapse = " + "))
  set.seed(113)
  tuneFit=train(as.formula(f),
                data=data.frame(train),
                method='rpart2',
                metric="Accuracy",
                preProcess=c("center", "scale"),
                tuneGrid=data.frame(.maxdepth=seq(1,30,1)), 
                trControl=fitControl,
                tuneLength=5)
  return(tuneFit)
}

resultsTuneTree = function(tuneTree) {
  para = c("Complexity Parameter")
  tuneResults = c(tuneTree$finalModel$control$cp)
  output = data.frame ("Parameters" = para, "Results" = tuneResults)
  return(output)
}

tuneForest = function(model_vars, fitControl){
  f = paste("Churn ~ ", paste(model_vars, collapse = " + "))
  set.seed(113)
  tuneFit=train(as.formula(f),
                data=data.frame(train),
                method="rf",
                preProcess=c("center", "scale"),
                metric="Accuracy",
                tuneGrid=data.frame(.mtry=seq(1,20,1)),
                trControl=fitControl,
                tuneLength=5)
  return(tuneFit)
}

resultsTuneForest = function(tuneForest) {
  para = c("Number of features to consider at each split point")
  tuneResults = c(tuneForest$finalModel$mtry)  
  output = data.frame("Parameters" = para, "Results" = tuneResults)
  return(output)
}

## build tree and forest 
makeTree = function(model_vars, min_split, min_bucket, max_depth,cp) {
  train_dat = train
  f = paste("Churn ~ ", paste(model_vars, collapse = " + "))
  tree = rpart(
    as.formula(f),
    method = "class", 
    data = train_dat,
    parms = list(split = "gini"),  
    minsplit = min_split,
    minbucket = min_bucket,
    maxdepth = max_depth,
    cp = cp  
  )
  return(tree)
}

useTree = function(tree, filename) {
  data = filename
  prediction = predict(tree, filename, type = "class")
  results = as.data.frame(prediction)
  results$truth = data$Churn
  return(results)
}

makeForest = function(model_vars, ntree, mtry) {
  train_data = train
  f = paste("Churn ~ ", paste(model_vars, collapse = " + "))
  forest = randomForest(as.formula(f), 
                        data = train_data, 
                        ntree = ntree, 
                        mtry=mtry)
  return(forest)
}

useForest = function(forest, filename) {
  data = filename
  prediction = predict(forest, filename)
  results = as.data.frame(prediction)
  results$truth = data$Churn
  return(results)
}

useForest2 = function(forest, filename) {
  data = filename
  prediction = forest$predicted
  results = as.data.frame(prediction)
  results$truth = data$Churn
  return(results)
}

## Calculate score
calcScores = function(results) {
  results = table(results)
  
  # calculate the scores 
  accuracy = round(100 * (results[1] + results[4]) / sum(results), 2)
  true_neg = round(100 * results[1] / sum(results[1, ]), 2)
  true_pos = round(100 * results[4] / sum(results[2, ]), 2)
  
  return(list(
    paste(c("Overall Accuracy: ",   accuracy, "%"), collapse = ""),
    paste(c("True Positive Rate: ", true_pos, "%"), collapse = ""),
    paste(c("True Negative Rate: ", true_neg, "%"), collapse = "")
  ))
}

resultsTable = function(results) {
  data = table(results)
  Outcomes = c("Predicted Not Leave", "Predicted Leave", "Total")
  c1 = c(data[, 1], sum(data[, 1]))    
  c2 = c(data[, 2], sum(data[, 2]))   
  c3 = c(sum(data[, 1]), sum(data[2, ]), sum(data))
  
  # turn these columns back into a dataframe 
  output = data.frame(Outcomes)
  output$"Actually Not Leave" = c1
  output$"Actually Leave"     = c2
  output$"Total"             = c3
  
  return(output)
}

# SERVER LOGIC FUNCTION
server = function(input, output, session) {
  # DATA PREVIEW
  # data table
  output$mytable = DT::renderDataTable({data})
  
  # HYPERPARAMETER
  fit = eventReactive(
    eventExpr = input$tunePara,
    valueExpr = fitControl(input$numbers, input$repeats))
  # decision tree
  hyper_tree = eventReactive(
    eventExpr = input$tunePara,
    valueExpr = tuneTree(
      model_vars = input$features0, fit() )
  )
  output$para_tree_tabels = renderTable(resultsTuneTree(hyper_tree()), 
                                        striped = TRUE) 
  output$varImp_tree_plot = renderPlot(plot(varImp(hyper_tree())))
  # random forest
  hyper_forest = eventReactive(
    eventExpr = input$tunePara,
    valueExpr = tuneForest(
      model_vars = input$features0, fit() )
  )
  output$para_forest_tabels = renderTable(resultsTuneForest(hyper_forest()), 
                                          striped = TRUE)
  output$varImp_forest_plot = renderPlot(plot(varImp(hyper_forest())))
  
  # DECISION TREES
  # INPUT EVENT REACTIONS
  # reconstruct the tree every time createModel is pressed
  tree = eventReactive(
    eventExpr = input$createModel,
    valueExpr = makeTree(
      model_vars = input$features, 
      input$min_split, input$min_bucket, input$max_depth, input$cp
    )
  )
  # regenerate training results every time createModel is pressed for a tree
  training_results = eventReactive(
    eventExpr = input$createModel,
    valueExpr = useTree(tree(), train)
  )
  # regenerate test results every time createModel is pressed for a tree
  test_results = eventReactive(
    eventExpr = input$testModel,
    valueExpr = useTree(tree(), test)
  )
  
  # OUTPUT DISPLAY PREP
  # assessment scores are each collapsed to display on a new line
  output$training_scores = renderText(
    paste(calcScores(training_results()), collapse = "\n")
  )
  output$test_scores = renderText(
    paste(calcScores(test_results()), collapse = "\n")
  )
  
  # tables of outcome breakdows are static widgets
  output$training_table = renderTable(
    resultsTable(training_results()),
    align = "lccc",   
    striped = TRUE
  )
  output$test_table = renderTable(
    resultsTable(test_results()),
    align = "lccc",   
    striped = TRUE
  )
  
  # frame for a plot of the decision tree
  output$tree_plot = renderPlot(
    prp(
      tree(), roundint = FALSE,
      extra = 0, branch = 1, varlen = 0,
      # colours True terminals in red, False terminals in blue
      box.col = c("cornflowerblue", "tomato")[tree()$frame$yval]
    )
  )
  
  # RANDOM FOREST 
  # INPUT EVENT REACTIONS
  # reconstruct the forest every time createModel is pressed
  forest = eventReactive(
    eventExpr = input$createModel2,
    valueExpr = makeForest(
      model_vars = input$features2,input$ntree, input$mtry)
  )
  # regenerate training results every time createModel is pressed for a forest
  training_results2 = eventReactive(
    eventExpr = input$createModel2,
    valueExpr = useForest2(forest(), train)
  )
  # regenerate test results every time createModel is pressed for a forest
  test_results2 = eventReactive(
    eventExpr = input$testModel2,
    valueExpr = useForest(forest(), test)
  )
  
  # OUTPUT DISPLAY PREP
  # assessment scores are each collapsed to display on a new line
  output$training_scores2 = renderText(
    paste(calcScores(training_results2()), collapse = "\n")
  )
  output$test_scores2 = renderText(
    paste(calcScores(test_results2()), collapse = "\n")
  )
  
  # tables of outcome breakdows are static widgets
  output$training_table2 = renderTable(
    resultsTable(training_results2()),
    align = "lccc",   
    striped = TRUE
  )
  output$test_table2 = renderTable(
    resultsTable(test_results2()),
    align = "lccc",   
    striped = TRUE
  )
  
  # frame for a plot of the random forest
  output$forest_plot = renderPlot(plot(forest() ))
}

# USER INTERFACE FUNCTION
ui = navbarPage(
  "Telecom Churn Customers Filtering",
  tabPanel("Hello Shiny!",
           titlePanel("Welcome!"),
           helpText(
             "Welcome to the Shiny app! ",
             br(), br(),
             "Here you can have a play with some of the variables in the dataset. ",
             "First and foremost, please go through the 'About' tab to have a ",
             "peak at what the datasets is about. Then, go to the 'Hyperparametes' ",
             "tab to tune for the best parameters. Try adding and removing featuers ",
             "and changing the hyperparameters to see how they affect ",
             "the results. Last but not least, choose the model of your prefered choice ",
             "at tab 'Model'. As long as you have choosen the model, ",
             "simply press the big blue button to generate the ",
             "model and see how well it does. Try the hyperparameters ",
             "that you have tune before to see how they affect ",
             "the results. Once you're happy with your results from the training",
             "data press the big red button to run your model with the test data.",
             br(), br(),
             "Both the interface and the underlying analysis are written in",
             tags$a(
               href = "https://en.wikipedia.org/wiki/R_(programming_language)",
               "the R programming language"
             ),
             "which is the industry standard for statistical analysis.",
             br(), br(),
             "Now let's try it on your own.",
             br(), br(),
             h4("HAVE FUN!")
           ) ),
  tabPanel("About",
           h2("Telecom Churn"),
           br(),
           
           h3("Description"),
           h5(
             "The Orange Telecom's Churn Dataset, which consists of ",
             "cleaned customer activity data (features), along with a churn ",
             "label specifying whether a customer canceled the subscription, ",
             "will be used to develop predictive models. "),
           br(),
           h3("Information"),
           h5(
             "Each row represents a customer, each column contains customer's attributes.",
             br(),br(),
             "The data set includes information about:",br(),br(),
             "- Customers who left within the last month - the column is called Churn",br(),
             "- Length of customers account",br(),
             "- Code of customers area ",br(),
             "- Planning of customers for voice mail and international",br(),
             "- Numbers of voice mail messages",br(),
             "- Total minutes, calls, charge at day, evening, night and through international",
           ),
           br(),
           h2("Data Preview"),
           DT::dataTableOutput("mytable"),
           helpText(
             "Source:",
             tags$a(
               href = "https://www.kaggle.com/taranenkodaria/churn-in-telecom-s-dataset/data",
               "Kaggle"
             )
           )
  ),
  tabPanel("Hyperparameter",
           titlePanel("Hyperparameter Tuning"),
           sidebarLayout(
             sidebarPanel(
               h2("Input"),
               helpText("Press 'Tuning' button to see tuning results. (It will took a while)"),
               actionButton(
                 inputId = "tunePara",
                 label = "Tuning",
                 class = "btn-primary" ),
               br(),
               
               h3("Model Features"),
               helpText(
                 "The below controls allow you to add and remove specific",
                 "features from your model. Use the dropdown picker to add" ,
                 "and remove these features from your model."
               ),
               pickerInput(
                 inputId = "features0",
                 label = NULL,  
                 choices = variable_list,
                 selected = variable_list,
                 options = list(`actions-box` = TRUE),
                 multiple = TRUE
               ),
               br(), 
               h3("Training Control"),
               h4("Numbers"),
               helpText("The number of folds or number of resampling iterations."),
               sliderInput(
                 inputId = "numbers",
                 label = NULL,  
                 min = 2,     
                 max = 10,    
                 value = 3     
               ),
               br(),
               h4("Repeats"),
               helpText("The number of complete sets of folds to compute."),
               sliderInput(
                 inputId = "repeats",
                 label = NULL,  
                 min = 1,     
                 max = 10,    
                 value = 5     
               )
             ),
             mainPanel(
               h2("Fitting results"),
               h3("Hyperparameters"),
               helpText(
                 "Parameters which define the model architecture are ",
                 "referred to as hyperparameters and thus this process of ",
                 "searching for the ideal model architecture."),
               fluidRow(
                 label = NULL,
                 column(6,
                        h4("Decision Tree"),
                        tableOutput("para_tree_tabels")
                 ),
                 column(6,
                        h4("Random Forest"),
                        tableOutput("para_forest_tabels")
                 ),
               ),
               br(),br(),
               h3("Variables Importance"),
               helpText(
                 "An important variable is a variable that is used as a ",
                 "primary or surrogate splitter in the tree. The variable ",
                 "with the highest improvement score is set as the most ",
                 "important variable, and the other variables are ranked accordingly."),
               h4("Decision Tree"),
               plotOutput("varImp_tree_plot"),
               br(),
               h4("Random Forest"),
               plotOutput("varImp_forest_plot")
             )
           )
  ),
  navbarMenu("Model",
             tabPanel("Decision Tree",
                      titlePanel("Decision Tree"),
                      helpText(
                        "A decision tree is a supervised machine learning algorithm that",
                        "can be used for both classification and regression problems. ",
                        "A decision tree is simply a series of sequential decisions made ",
                        "to reach a specific result."),
                      br(),
                      
                      # partition the rest of the page into controls and output
                      sidebarLayout(
                        sidebarPanel(
                          h2("The Controls"),
                          br(),
                          
                          actionButton(
                            inputId = "createModel",
                            label = "Create Model",
                            class = "btn-primary" 
                          ),
                          helpText(
                            "Pressing this blue button creates a decision tree model for",
                            "the training data, based on the features you've selected and",
                            "the chosen hyperparameter values. Use this while you're still",
                            "developing your model; if you change the selected features",
                            "just press it again to regenerate the model."
                          ),
                          br(),
                          actionButton(
                            inputId = "testModel",
                            label = "Test Model",
                            class = "btn-danger" 
                          ),
                          helpText(
                            " Press the above red button to run it on the test data.", 
                            " Only use this once you're happy and ready to assess your model."
                          ),
                          br(),
                          
                          h3("Model Features"),
                          helpText(
                            "The below controls allow you to add and remove specific",
                            "features from your model. Use the dropdown picker to add" ,
                            "and remove these features from your model."
                          ),
                          pickerInput(
                            inputId = "features",
                            label = NULL,  
                            choices = variable_list,
                            selected = variable_list_default,
                            options = list(`actions-box` = TRUE),
                            multiple = TRUE
                          ),
                          br(),
                          
                          h3("Hyperparameter"),
                          helpText(
                            "These controls are for setting the hyperparameter values",
                            "which partly control the structure of the decision tree."
                          ),
                          br(),
                          h4("Minimum Split"),
                          helpText(
                            "If at a given node N is below this value, that node cannot",
                            "be split any further: it is a terminal node of the tree."
                          ),
                          sliderInput(
                            inputId = "min_split",
                            label = NULL,  
                            min = 2,     
                            max = 30,    
                            value = 20     
                          ),
                          br(),
                          h4("Minimum Bucket Size"),
                          helpText(
                            "If creating a given split would cause N₁ or N₂ to fall below",
                            "this minimum, then that split isn't made part of the",
                            "decision tree."
                          ),
                          sliderInput(
                            inputId = "min_bucket",
                            label = NULL, 
                            min = 1,       
                            max = 30,     
                            value = 7     
                          ),
                          br(),
                          h4("Complexity Parameter"),
                          helpText(
                            "The minimum improvement in the model needed at each node"
                          ),
                          sliderInput(
                            inputId = "cp",
                            label = NULL,  
                            min = 0,      
                            max = 0.2,
                            value = 0.01 ,
                            step = 0.01     
                          ),
                          br(),
                          h4("Maximum Tree Depth"),
                          helpText(
                            "Control the maximum depth that the decision tree can reach."
                          ),
                          sliderInput(
                            inputId = "max_depth",
                            label = NULL, 
                            min = 2,       
                            max = 30,     
                            value = 4     
                          )
                        ),
                        
                        mainPanel(
                          fluidRow(
                            label = NULL,
                            column(6,
                                   h2("Training Results"),
                                   helpText(
                                     "These are the measures of how good your model was",
                                     "when it was ran on the training data set."
                                   ),
                                   # training accuracy, true positive, and true negative
                                   tagAppendAttributes(
                                     textOutput("training_scores"),
                                     style = "white-space: pre-wrap; font-size: 17px;"
                                   ),
                                   br(),
                                   tableOutput("training_table")
                            ),
                            column(6,
                                   h2("Test Results"),
                                   helpText(
                                     "These are the measures of how good your model was",
                                     "when it was ran on the test data set. "
                                   ),
                                   # test accuracy, true positive, and true negative
                                   tagAppendAttributes(
                                     textOutput("test_scores"),
                                     style = "white-space: pre-wrap; font-size: 17px;"
                                   ),
                                   br(),
                                   tableOutput("test_table")
                            )
                          ),
                          # plot of the decision tree
                          h2("Decision Tree"),
                          helpText("This is a graphical depiction of the decision tree."),
                          plotOutput(outputId = "tree_plot")
                        )
                      ) 
             ), 
             tabPanel("Random forest",
                      titlePanel("Random forest"),
                      helpText(
                        "The decision tree algorithm is quite easy to understand and ",
                        "interpret. But often, a single tree is not sufficient for ",
                        "producing effective results. This is where the Random Forest",
                        "algorithm comes. Random Forest is a tree-based machine learning",
                        "algorithm that leverages the power of multiple decision trees",
                        "for making decisions."),
                      br(),
                      
                      # partition the rest of the page into controls and output
                      sidebarLayout(
                        sidebarPanel(
                          h2("The Controls"),
                          br(),
                          
                          actionButton(
                            inputId = "createModel2",
                            label = "Create Model",
                            class = "btn-primary" 
                          ),
                          helpText(
                            "Pressing this blue button creates a random forest model for",
                            "the training data, based on the features you've selected and",
                            "the chosen hyperparameter values. Use this while you're still",
                            "developing your model; if you change the selected features",
                            "just press it again to regenerate the model."
                          ),
                          br(),
                          actionButton(
                            inputId = "testModel2",
                            label = "Test Model",
                            class = "btn-danger" 
                          ),
                          helpText(
                            " Press the above red button to run it on the test data.", 
                            " Only use this once you're happy and ready to assess your model."
                          ),
                          br(),
                          
                          h3("Model Features"),
                          helpText(
                            "The below controls allow you to add and remove specific",
                            "features from your model. Use the dropdown picker to add" ,
                            "and remove these features from your model."
                          ),
                          pickerInput(
                            inputId = "features2",
                            label = NULL,  
                            choices = variable_list,
                            selected = variable_list_default,
                            options = list(`actions-box` = TRUE),
                            multiple = TRUE
                          ),
                          br(),
                          
                          h3("Hyperparameter"),
                          helpText(
                            "These controls are for setting the hyperparameter values",
                            "which partly control the structure of the random forest."
                          ),
                          br(),
                          h4("Number of features to consider at each split point"),
                          helpText(
                            "The number of variables randomly sampled as candidates at each split."
                          ),
                          sliderInput(
                            inputId = "mtry",
                            label = NULL,  
                            min = 1,     
                            max = 20,    
                            value = 18     
                          ),
                          br(),
                          h4("Number of Trees to Grow"),
                          helpText(
                            "The number of decision trees you'd like to grow."
                          ),
                          sliderInput(
                            inputId = "ntree",
                            label = NULL,  
                            min = 100,     
                            max = 500,    
                            value = 500     
                          )
                        ),
                        
                        mainPanel(
                          fluidRow(
                            label = NULL,
                            column(6,
                                   h2("Training Results"),
                                   helpText(
                                     "These are the measures of how good your model was",
                                     "when it was ran on the training data set."
                                   ),
                                   # training accuracy, true positive, and true negative
                                   tagAppendAttributes(
                                     textOutput("training_scores2"),
                                     style = "white-space: pre-wrap; font-size: 17px;"
                                   ),
                                   br(),
                                   tableOutput("training_table2")
                            ),
                            column(6,
                                   h2("Test Results"),
                                   helpText(
                                     "These are the measures of how good your model was",
                                     "when it was ran on the test data set. "
                                   ),
                                   # test accuracy, true positive, and true negative
                                   tagAppendAttributes(
                                     textOutput("test_scores2"),
                                     style = "white-space: pre-wrap; font-size: 17px;"
                                   ),
                                   br(),
                                   tableOutput("test_table2")
                            )
                          ),
                          # plot of the random forest
                          h2("Random Forest"),
                          helpText("This is a graphical depiction of the random forest." ),
                          plotOutput(outputId = "forest_plot")
                        )
                      )
             )
  )
)

shinyApp(ui = ui, server = server)

