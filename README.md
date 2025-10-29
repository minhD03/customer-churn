# Customer Churn Experiment
## 1. Introduction:
In this project, I will perform experiments in Microsoft Fabric to test out Machine Learning Models. These models will be used to predict the Customer Data to see whether a Customer is "Exited" or not. This will help deciding which customer is potential for long term profit. The experiment consists of multiple models: Random Forest Classification, CatBoost, LightGBM, etc.

## 2. Libraries Required:

These are the libraries that were neccessary to conduct the experiment. To install these libraries, enter the following code:

| Library / Tool    | Install Command            | Purpose                                                                 | What It Does                                                                                   |
|-------------------|----------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| pandas            | `pip install pandas`       | Core data manipulation and analysis                                     | Provides DataFrames for tabular data, supports filtering, grouping, merging, and time series   |
| numpy             | `pip install numpy`        | Numerical operations and array manipulation                            | Enables fast numerical computations, array broadcasting, and linear algebra operations         |
| matplotlib        | `pip install matplotlib`   | Global plot aesthetics and layout customization                         | Creates static, animated, and interactive visualizations with full control over plot elements  |
| seaborn           | `pip install seaborn`      | High-level statistical data visualization for churn patterns            | Simplifies complex plots (e.g., heatmaps, violin plots) with built-in themes and aggregation   |
| scikit-learn      | `pip install scikit-learn` | ML models, metrics, and utilities (e.g., train-test split, classifiers) | Offers a wide range of supervised/unsupervised models, preprocessing tools, and evaluation     |
| imbalanced-learn  | `pip install imbalanced-learn` | Tools for handling imbalanced datasets (e.g., SMOTE)                | Provides resampling techniques like SMOTE, Tomek Links, and ensemble balancing strategies       |
| lightgbm          | `pip install lightgbm`     | Fast gradient boosting classifier for large-scale churn datasets        | Uses histogram-based learning for efficient training on large, sparse datasets                 |
| xgboost           | `pip install xgboost`      | High-performance gradient boosting (XGBClassifier)                      | Optimized for speed and accuracy with regularization, early stopping, and tree pruning         |
| catboost          | `pip install catboost`     | Gradient boosting optimized for categorical features                    | Handles categorical variables natively, reducing preprocessing and improving accuracy          |
| mlflow            | `pip install mlflow`       | ML lifecycle management: tracking, reproducibility, deployment          | Tracks experiments, logs models and metrics, supports deployment across platforms              |

## 3. Instruction:

- Download Data and Notebook.
- Run the Notebook from the beginning cell.
- Pay attention to this configuration before running the code and modify it based on your running device.
- My exported Result can be viewed [here](https://github.com/minhD03/Customer-Churn/tree/9fb6c96741d46aa75b0769f35a3a85ea3d62dafd/Result).

```bash
DATA_ROOT = "lakehouse/default"
DATA_FOLDER = "Files"
DATA_FILE = "churn.csv"
``` 
## 5. Preview Results:
Thesea are the screenshots captured from the experiments. Further Report can be viewed in here for: [Notebook](https://github.com/minhD03/Customer-Churn/blob/0bb85d47524ff81890aff279dcfb7802b2fbd230/Code/Customer%20Churn%20-%20Nhat%20Minh%20Dang%20-%20ver%202.0.ipynb) And [Experiment Analysis](https://github.com/minhD03/Customer-Churn/blob/0bb85d47524ff81890aff279dcfb7802b2fbd230/Document/Customer%20Churn%20Experiment%20Report%20-%20Nhat%20Minh%20Dang%20-%20ver%202.0.pdf).

![alt text](https://github.com/minhD03/Customer-Churn/blob/9fb6c96741d46aa75b0769f35a3a85ea3d62dafd/Image/Random%20Forest%20With%20Max%20Depth%20of%204.png)



## 5. Conclusions:
Based on the result from Notebook and the report, I believe that LightGBM should be more preferable Random Forest Classifier, especially for Customer Churn. Unlike Random Forest, which requires careful depth tuning to avoid underfitting or overfitting, LightGBM consistently delivers high performance with fewer iterations and better scalability. The gain and split-based feature importance outputs further confirm that LightGBM leverages nuanced patterns and prioritizes impactful variables, making it ideal for production-grade modeling where interpretability, speed and precision are critical. While Random Forest remains a reliable baseline, LightGBM’s advanced architecture and adaptability make it the model of choice for high-stakes, data-rich environments.
