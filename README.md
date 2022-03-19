# ml_project2-no_more_bugs

CS-433 Machine learning project2 team repo

# Project structure

```
- datasets
  - test_data.txt
  - train_neg_full.txt
  - train_pos_full.txt
- LSTM
  - Map_Dataset.py
  - RNN_LSTM.py
  - Run.py                  # train and predict using different methods and create output .csv
  - Text_Processing.py
- SVM
  - Run.py                  # train and predict using different methods and create output .csv
- Transformer
  - Run.py                  # train and predict using different methods and create output .csv
```

Data files are too big to push. We ignored them when committing.

# How to run the code

1. Download the data files from [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files), rename the folder to datasets. (see project structure)

2. If you want to test LSTM:
  
  - cd to /LSTM
  
  - run the following command:

   ```
   python Run.py
   
   ```
  
3. If you want to test SVM:
  
  - cd to /SVM
  
  - run the following command:

   ```
   python Run.py
   
   ```
 4. If you want to test Transformer:
  
  - cd to /Transformer
  
  - if you want to test "early stop" trick:
    - in Run.py set variable "is_early_stop" = True
    
  - if you want to test "sweep" models:
    - in Run.py set variable "is_sweep" = True
  
  - in Run.py set variable "model_sequence" with different value to test different models
    - value 1 for "bert" model
    - value 2 for "roberta" model
    - value 3 for "xlnet" model
  
  - run the following command:

   ```
   python Run.py
   
   ```

5. After training and testing finish, you should find a prediction.csv
