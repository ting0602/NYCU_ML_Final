# NYCU_ML_Final2022
**NYCU ML Final Project**

**Kaggle topic：Tabular Playground Series - Aug 2022**
* [spec](https://docs.google.com/presentation/d/15d4W_8GFks4Mqmf4kvmTxYC8tJv-KNg6c8rQrlccEWM/edit#slide=id.g61dd2f3d9d_2_83)
* [kaggle competition link](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview)

**Outline**
* [Tabular Playground Series](#tabular-playground-series)
  * [Introduction](#competition-introduction)
  * [Data Description](#data-description)
  * [Files](#files)
* [Code](#code)
  * [Downloaded Files](#downloaded-files)
  * [Data Processing](#data-processing)
  * [Training](#training)
  * [Inference](#inference)
* [Usage](#usage)
  * [Clone Repository](#1-clone-repository)
  * [Install Packages](#2-install-packages)
  * [Run Code](#3-run-code)
  * [Result](#4-result)
* [Reference](#reference)

## Tabular Playground Series
### Competition Introduction
The August 2022 edition of the Tabular Playground Series is an opportunity to help the fictional company Keep It Dry improve its main product Super Soaker. The product is used in factories to absorb spills and leaks.

The company has just completed a large testing study for different product prototypes. Can you use this data to build a model that predicts product failures?

### Data Description
This data represents the results of a large product testing study. For each product_code a number of product attributes (fixed for the code) as well as a number of measurement values for each individual product, representing various lab testing methods. Each product is used in a simulated real-world environment experiment, and and absorbs a certain amount of fluid (loading) to see whether or not it fails.

The task is to use the data to predict individual product failures of new codes with their individual lab test results.

### Files
* train.csv - the training data, which includes the target failure
* test.csv - the test set; your task is to predict the likelihood each id will experience a failure
* sample_submission.csv - a sample submission file in the correct format

## Code
The baseline accuracy is 0.58990 on private score.

Using this code can achieve accuracy is 0.59097 on private score.

### Downloaded Files
* The directory structure:
  * ./NYCU_ML_FINAL
    * /.gitingore
    * /109550119_final.pdf
    * /109550119_Final.ipynb
    * /109550119_Final_inference.ipynb
    * /109550119_Final_train.ipynb
    * /README.md
    * /requirements.txt
    * /submission.csv
    * /input
        * /new_test.csv
        * /new_train.csv
        * /sample_submission.csv
        * /test.csv
        * /train.csv
    * /models
        * /model1-1.pkl
        * /model1-2.pkl
        * ...
        * /model1-5.pkl
        * /model2-1.pkl
        * /model3-5.pkl
      
 * Files
   * 109550119_Final.ipynb - complete code, including training and inference
   * 109550119_Final_train.ipynb - contain only training code
   * 109550119_Final_inference.ipynb - conations only inference code
   * models - contains all trained models
   * new_train.csv, new_test.csv - the processed data
   * submission.csv - the inference result

### Data Processing
**1. Add new features**

* m3_missing, m3_missing: measurement_3 or measurement_5 is missing (Nan)

    ``` Python
    # 合併 train & test
    data = pd.concat([train, test])
    data['m3_missing'] = data['measurement_3'].isnull().astype(np.int8)
    data['m5_missing'] = data['measurement_5'].isnull().astype(np.int8)
    data['loading'] = np.log1p(data['loading'])
    display(data[:5])
    ```
**2. Impute data**

* use HuberRegressor and KNNImputer
  
    ``` Python
    feature = [f for f in test.columns if f.startswith('measurement') or f=='loading']

    fill_dict = {
        'A': ['measurement_5','measurement_6','measurement_8'],
        'B': ['measurement_4','measurement_5','measurement_7'],
        'C': ['measurement_5','measurement_7','measurement_8','measurement_9'],
        'D': ['measurement_5','measurement_6','measurement_7','measurement_8'],
        'E': ['measurement_4','measurement_5','measurement_6','measurement_8'],
        'F': ['measurement_4','measurement_5','measurement_6','measurement_7'],
        'G': ['measurement_4','measurement_6','measurement_8','measurement_9'],
        'H': ['measurement_4','measurement_5','measurement_7','measurement_8','measurement_9'],
        'I': ['measurement_3','measurement_7','measurement_8']
    }

    for code in data.product_code.unique():
        tmp = data[data.product_code==code]
        column = fill_dict[code]
        tmp_train = tmp[column+['measurement_17']].dropna(how='any')
        tmp_test = tmp[(tmp[column].isnull().sum(axis=1)==0)&(tmp['measurement_17'].isnull())]
        print(f"code {code} has {len(tmp_test)} samples to fill nan")
        model = HuberRegressor()
        model.fit(tmp_train[column], tmp_train['measurement_17'])
        data.loc[(data.product_code==code)&(data[column].isnull().sum(axis=1)==0)&(data['measurement_17'].isnull()), 'measurement_17'] = model.predict(tmp_test[column])

        model2 = KNNImputer(n_neighbors=5)
        print(f"KNN imputing code {code}")
        data.loc[data.product_code==code, feature] = model2.fit_transform(data.loc[data.product_code==code, feature])

    display(data[:5])
    ```

**3. Add new features (after missing data imputation)**

* m3_17_avg: average of measurement_3 and measurement_17

* m3_to_16_avg:  average of measurement_3 to measurement_16

* m3_17_stdev: standard deviation of measurement_3 and measurement_17

* area: multiplication of measurement_2 and measurement_3

  ``` Python
  data['m3_17_avg'] = (data['measurement_3'] + data['measurement_17']) / 2.0
  data['m3_to_16_avg'] = (data['measurement_3'] + data['measurement_4'] + data['measurement_5'] + data['measurement_6'] + data['measurement_7'] + data['measurement_8'] + data['measurement_9'] + data['measurement_10'] + data['measurement_11'] + data['measurement_12'] + data['measurement_13'] + data['measurement_14'] + data['measurement_15'] + data['measurement_16']) / 14.0
  data['m3_17_stdev'] = data[['measurement_3', 'measurement_17']].std(axis=1)
  data['area'] = (data['attribute_2'] * data['attribute_3'])
  display(data[:5])
  ```

### Training
**1. select features**
* There are three features combined in this program as follow:
  ``` Python
  select_feature = ['m3_missing', 'm5_missing', 'measurement_1', 'measurement_2', 'loading', 'measurement_17']
  ```
  ``` Python
  select_feature = ['measurement_1', 'measurement_2', 'loading', 'measurement_17']
  ```
  ``` Python
  select_feature = ['loading', 'measurement_17', 'm3_17_avg']
  ```
* You can also modify this section to find better combinations

**2. train**
* Use Cross-Validation (K-fold) and LogisticRegression to train the models
* Calculate the importance of selected features
* variables
  * lr_oof_1：The prediction results of LogisticRegression on each validation set
  * lr_oof_2：The predicted class predicted by LogisticRegression on each validation set
  * lr_test：The predicted probability predicted by LogisticRegression on each validation set
  * lr_auc：The average AUC of LogisticRegression during the cross-validation
  * lr_acc：The average accuracy of LogisticRegression during the cross-validation
  * importance_list：A list of weight(coefficient) of features estimated by LogisticRegression

    ``` Python
    lr_oof_1 = np.zeros(len(X))
    lr_oof_2 = np.zeros(len(X))
    lr_test = np.zeros(len(test))
    lr_auc = 0
    lr_acc = 0
    importance_list = []
    model_list = ['./models/model1-1.pkl', './models/model1-2.pkl', './models/model1-3.pkl', './models/model1-4.pkl','./models/model1-5.pkl']

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print("Fold:", fold_idx+1)
        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        x_test = test.copy()

        x_train, x_val, x_test = _scale(x_train, x_val, x_test, select_feature)

        model = LogisticRegression(max_iter=1000, C=0.0001, penalty='l2', solver='newton-cg') # , class_weight='balanced'
        model.fit(x_train[select_feature], y_train)

        # model.coef_ => weight(an array of weights estimated by linear regression)
        # ravel() => return contiguous flattened array (be 1D array)
        importance_list.append(model.coef_.ravel())

        val_preds = model.predict_proba(x_val[select_feature])[:, 1]
        lr_auc += roc_auc_score(y_val, val_preds) / 5
        y_preds = model.predict(x_val[select_feature])
        lr_acc += accuracy_score(y_val, y_preds) / 5
        lr_test += model.predict_proba(x_test[select_feature])[:, 1] / 5
        lr_oof_1[val_idx] = val_preds
        lr_oof_2[val_idx] = y_preds

        with open(model_list[fold_idx], 'wb') as pkl_file:
            pickle.dump(model, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"{Fore.GREEN}{Style.BRIGHT}Average auc = {round(lr_auc, 5)}, Average acc = {round(lr_acc, 5)}{Style.RESET_ALL}")
    print(f"{Fore.RED}{Style.BRIGHT}OOF auc = {round(roc_auc_score(y, lr_oof_1), 5)}, OOF acc = {round(accuracy_score(y, lr_oof_2), 5)}{Style.RESET_ALL}")

    importance_df = pd.DataFrame(np.array(importance_list).T, index=x_train[select_feature].columns)
    importance_df['mean'] = importance_df.mean(axis=1).abs()
    importance_df['feature'] = x_train[select_feature].columns
    importance_df = importance_df.sort_values('mean', ascending=False).reset_index().head(20)
    plt.barh(importance_df.index, importance_df['mean'], color='lightgreen')
    plt.gca().invert_yaxis()
    plt.yticks(ticks=importance_df.index, labels=importance_df['feature'])
    plt.title('LogisticRegression feature importances')
    plt.show()
    ```
    
    ![output](https://user-images.githubusercontent.com/69136310/211500741-2de0963e-c6a4-42f8-9f51-6a4bdf0fba38.png)
    
### Inference
**1. Load model**
* the models are all stored in the ./models folder

  ```Python
  with open(model_list[fold_idx], 'rb') as f:
          model = pickle.load(f)
  ```
**2. Result**
* After loading the models, do the same step as in train to get ```lr_test```
* Put ```lr_test```into rankdata()

  ```Python
  submission['lr1'] = lr_test
  ```
  ```python
  submission['rank0'] = rankdata(submission['lr0'])
  submission['rank1'] = rankdata(submission['lr1'])
  submission['rank2'] = rankdata(submission['lr2'])
  ```
* Adjust weight and get the result

  ``` Python
  submission['failure'] = submission['rank0']*0.70 + submission['rank1']*0.05 + submission['rank2']*0.30
  ```
  ``` python
  submission[['id', 'failure']].to_csv('submission.csv', index=False)
  ```

## Usage
### 1. Clone Repository
```
git clone git@github.com:ting0602/NYCU_ML_Final.git
```
### 2. Install Packages
```
pip install -r requirements.txt
```
 
### 3. Run Code
You can choose to run entire code, or run training code and inference separately.
run entire code - run ```109550119_Final.ipynb```
run training code and inference separately - run ```109550119_Final_train.ipynb``` first, then use new model to run ```109550119_Final_inference.ipynb```

### 4. Result
You can check the result in ```submission.csv```.
 
## Reference
* https://www.kaggle.com/code/takanashihumbert/tps-aug22-9th-solution/notebook?fbclid=IwAR0_uaztxUxw_pHXL74TZVjN-26DG_r5UCSAROUtrjfxGa0iUzjD1ekZE3c
*	https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/343939
*	https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/342319
*	https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/342126
* https://www.kaggle.com/code/mehrankazeminia/tps22aug-logisticr-lgbm-keras
