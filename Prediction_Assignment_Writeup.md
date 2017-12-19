# Prediction Assignment Writeup
Juan Tenopala  
19 de diciembre de 2017  

#Prediction Assignment Writeup

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Objective
To use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, to predict the manner in which they did the exercise.

## Data
The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Introduction
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
### The model
The outcome variable is classe, a factor variable with 5 levels. For this data set, participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions: Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.
Prediction evaluations will be based on maximizing the accuracy and minimizing the out-of-sample error. All other available variables after cleaning will be used for prediction.
In this case I will use a Random Forest algorithm because the features of the algorithm are important for classification and prediction.

## Code and Results
### Packages, Libraries and seed

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.4.2
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
set.seed(1234)
```

### Load the data sets

```r
trainingset <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
testingset <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))
dim(trainingset)
```

```
## [1] 19622   160
```

```r
dim(testingset)
```

```
## [1]  20 160
```

### Preprocessing
Delete columns with all "NA" and the variables that are irrelevant to the analysis.

```r
trainingset<-trainingset[,colSums(is.na(trainingset)) == 0]
testingset <-testingset[,colSums(is.na(testingset)) == 0]
trainingset   <-trainingset[,-c(1:7)]
testingset <-testingset[,-c(1:7)]
```

### Exploratory data analysis

```r
subsamples <- createDataPartition(y=trainingset$classe, p=0.7, list=FALSE)
subTraining <- trainingset[subsamples, ] 
subTesting <- trainingset[-subsamples, ]
dim(subTraining)
```

```
## [1] 13737    53
```

```r
dim(subTesting)
```

```
## [1] 5885   53
```

```r
head(subTraining)
```

```
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
## 2      1.41       8.07    -94.4                3         0.02         0.00
## 3      1.42       8.07    -94.4                3         0.00         0.00
## 4      1.48       8.05    -94.4                3         0.02         0.00
## 5      1.48       8.07    -94.4                3         0.02         0.02
## 6      1.45       8.06    -94.4                3         0.02         0.00
## 7      1.42       8.09    -94.4                3         0.02         0.00
##   gyros_belt_z accel_belt_x accel_belt_y accel_belt_z magnet_belt_x
## 2        -0.02          -22            4           22            -7
## 3        -0.02          -20            5           23            -2
## 4        -0.03          -22            3           21            -6
## 5        -0.02          -21            2           24            -6
## 6        -0.02          -21            4           21             0
## 7        -0.02          -22            3           21            -4
##   magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm
## 2           608          -311     -128      22.5    -161              34
## 3           600          -305     -128      22.5    -161              34
## 4           604          -310     -128      22.1    -161              34
## 5           600          -302     -128      22.1    -161              34
## 6           603          -312     -128      22.0    -161              34
## 7           599          -311     -128      21.9    -161              34
##   gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
## 2        0.02       -0.02       -0.02        -290         110        -125
## 3        0.02       -0.02       -0.02        -289         110        -126
## 4        0.02       -0.03        0.02        -289         111        -123
## 5        0.00       -0.03        0.00        -289         111        -123
## 6        0.02       -0.03        0.00        -289         111        -122
## 7        0.00       -0.03        0.00        -289         111        -125
##   magnet_arm_x magnet_arm_y magnet_arm_z roll_dumbbell pitch_dumbbell
## 2         -369          337          513      13.13074      -70.63751
## 3         -368          344          513      12.85075      -70.27812
## 4         -372          344          512      13.43120      -70.39379
## 5         -374          337          506      13.37872      -70.42856
## 6         -369          342          513      13.38246      -70.81759
## 7         -373          336          509      13.12695      -70.24757
##   yaw_dumbbell total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 2    -84.71065                   37                0            -0.02
## 3    -85.14078                   37                0            -0.02
## 4    -84.87363                   37                0            -0.02
## 5    -84.85306                   37                0            -0.02
## 6    -84.46500                   37                0            -0.02
## 7    -85.09961                   37                0            -0.02
##   gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## 2             0.00             -233               47             -269
## 3             0.00             -232               46             -270
## 4            -0.02             -232               48             -269
## 5             0.00             -233               48             -270
## 6             0.00             -234               48             -269
## 7             0.00             -232               47             -270
##   magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## 2              -555               296               -64         28.3
## 3              -561               298               -63         28.3
## 4              -552               303               -60         28.1
## 5              -554               292               -68         28.0
## 6              -558               294               -66         27.9
## 7              -551               295               -70         27.9
##   pitch_forearm yaw_forearm total_accel_forearm gyros_forearm_x
## 2         -63.9        -153                  36            0.02
## 3         -63.9        -152                  36            0.03
## 4         -63.9        -152                  36            0.02
## 5         -63.9        -152                  36            0.02
## 6         -63.9        -152                  36            0.02
## 7         -63.9        -152                  36            0.02
##   gyros_forearm_y gyros_forearm_z accel_forearm_x accel_forearm_y
## 2            0.00           -0.02             192             203
## 3           -0.02            0.00             196             204
## 4           -0.02            0.00             189             206
## 5            0.00           -0.02             189             206
## 6           -0.02           -0.03             193             203
## 7            0.00           -0.02             195             205
##   accel_forearm_z magnet_forearm_x magnet_forearm_y magnet_forearm_z
## 2            -216              -18              661              473
## 3            -213              -18              658              469
## 4            -214              -16              658              469
## 5            -214              -17              655              473
## 6            -215               -9              660              478
## 7            -215              -18              659              470
##   classe
## 2      A
## 3      A
## 4      A
## 5      A
## 6      A
## 7      A
```

```r
head(subTesting)
```

```
##    roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x
## 1       1.41       8.07    -94.4                3         0.00
## 21      1.60       8.10    -94.4                3         0.02
## 22      1.57       8.09    -94.4                3         0.02
## 23      1.56       8.10    -94.3                3         0.02
## 25      1.53       8.11    -94.4                3         0.03
## 26      1.55       8.09    -94.4                3         0.02
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 1          0.00        -0.02          -21            4           22
## 21         0.00        -0.02          -20            1           20
## 22         0.02        -0.02          -21            3           21
## 23         0.00        -0.02          -21            4           21
## 25         0.00         0.00          -19            4           21
## 26         0.00         0.00          -21            3           22
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 1             -3           599          -313     -128      22.5    -161
## 21           -10           607          -304     -129      20.9    -161
## 22            -2           604          -313     -129      20.8    -161
## 23            -4           606          -311     -129      20.7    -161
## 25            -8           605          -319     -129      20.7    -161
## 26           -10           601          -312     -129      20.7    -161
##    total_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x
## 1               34        0.00        0.00       -0.02        -288
## 21              34        0.03       -0.02       -0.02        -288
## 22              34        0.03       -0.02       -0.02        -289
## 23              34        0.02       -0.02       -0.02        -290
## 25              34       -0.02       -0.02        0.00        -289
## 26              34       -0.02       -0.02       -0.02        -290
##    accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z
## 1          109        -123         -368          337          516
## 21         111        -124         -375          337          513
## 22         111        -123         -372          338          510
## 23         110        -123         -373          333          509
## 25         109        -123         -370          340          512
## 26         108        -123         -366          346          511
##    roll_dumbbell pitch_dumbbell yaw_dumbbell total_accel_dumbbell
## 1       13.05217      -70.49400    -84.87394                   37
## 21      13.38246      -70.81759    -84.46500                   37
## 22      13.37872      -70.42856    -84.85306                   37
## 23      13.35451      -70.63995    -84.64919                   37
## 25      13.05217      -70.49400    -84.87394                   37
## 26      12.80060      -70.31305    -85.11886                   37
##    gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
## 1                 0            -0.02             0.00             -234
## 21                0            -0.02             0.00             -234
## 22                0            -0.02             0.00             -233
## 23                0            -0.02             0.00             -234
## 25                0            -0.02             0.00             -234
## 26                0            -0.02            -0.02             -233
##    accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
## 1                47             -271              -559               293
## 21               48             -269              -554               299
## 22               48             -270              -554               301
## 23               48             -270              -557               294
## 25               47             -271              -555               290
## 26               46             -271              -563               294
##    magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
## 1                -65         28.4         -63.9        -153
## 21               -72         26.9         -63.9        -151
## 22               -65         27.0         -63.9        -151
## 23               -69         26.9         -63.8        -151
## 25               -68         27.1         -63.7        -151
## 26               -72         27.0         -63.7        -151
##    total_accel_forearm gyros_forearm_x gyros_forearm_y gyros_forearm_z
## 1                   36            0.03            0.00           -0.02
## 21                  36            0.03           -0.03           -0.02
## 22                  36            0.02           -0.03           -0.02
## 23                  36            0.02           -0.02           -0.02
## 25                  36            0.05           -0.03            0.00
## 26                  36            0.03            0.00            0.00
##    accel_forearm_x accel_forearm_y accel_forearm_z magnet_forearm_x
## 1              192             203            -215              -17
## 21             194             208            -214              -11
## 22             191             206            -213              -17
## 23             194             206            -214              -10
## 25             191             202            -214              -14
## 26             190             203            -216              -16
##    magnet_forearm_y magnet_forearm_z classe
## 1               654              476      A
## 21              654              469      A
## 22              654              478      A
## 23              653              467      A
## 25              667              470      A
## 26              658              462      A
```


```r
plot(subTraining$classe, col="blue", main="5 levels in variable classe of the subTraining data set", xlab="classe levels", ylab="Frequency")
```

![](Prediction_Assignment_Writeup_files/figure-html/unnamed-chunk-5-1.png)<!-- -->
From the graph above, we can see that each level frequency is within the same order of magnitude of each other. Level A is the most frequent with more than 4000 occurrences while level D is the least frequent with about 2500 occurrence

### Prediction model

```r
model <- randomForest(classe ~. , data=subTraining, method="class")
subprediction <- predict(model, subTesting, type="class")
prediction <- predict(model, testingset, type="class")
prediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

