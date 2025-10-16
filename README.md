# Customer Churn Experiment
## 1. Introduction:
In this project, I will perform experiments in Microsoft Fabric to test out Machine Learning Models. These models will be used to predict the Customer Data to see whether a Customer is "Exited" or not. This will help deciding which customer is potential for long term profit. The experiment consists of two main models: Random Forest Classifier and LightG

## 2. Libraries Required:

These are the libraries that were neccessary to conduct the experiment. To install these libraries, enter the following code:

| Library          | Install Command              | Purpose                                               |
|------------------|------------------------------|-------------------------------------------------------|
| pandas           | ```bash pip install pandas```           | Data manipulation and analysis                        |
| numpy            | ```bash pip install numpy```             | Numerical operations                                  |
| matplotlib       | ```bash pip install matplotlib```     | Plotting and visualization                            |
| seaborn          | ```bash pip install seaborn```           | Statistical data visualization                        |
| imbalanced-learn | ```bash pip install imbalanced-learn```  | Handling imbalanced datasets (e.g., SMOTE)            |
| scikit-learn     | ```bash pip install scikit-learn```      | Machine learning models and metrics                   |
| lightgbm         | ```bash pip install lightgbm```          | Gradient boosting classifier                          |

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
## 4. Preview Results:
Thesea are the screenshots captured from the experiments. Further Report can be viewed in here for: [Notebook](https://github.com/minhD03/Customer-Churn/blob/9fb6c96741d46aa75b0769f35a3a85ea3d62dafd/Customer%20Churn%20-%20Nhat%20Minh%20Dang.ipynb) And [Experiment Analysis](https://github.com/minhD03/Customer-Churn/blob/9fb6c96741d46aa75b0769f35a3a85ea3d62dafd/Customer%20Churn%20Experiment%20Report%20-%20Nhat%20Minh%20Dang.pdf).

![alt text](https://github.com/minhD03/Customer-Churn/blob/9fb6c96741d46aa75b0769f35a3a85ea3d62dafd/Image/Random%20Forest%20With%20Max%20Depth%20of%204.png)
![alt text](https://github.com/minhD03/Customer-Churn/blob/9fb6c96741d46aa75b0769f35a3a85ea3d62dafd/Image/Random%20Forest%20With%20Max%20Depth%20of%208.png)
![alt text](https://github.com/minhD03/Customer-Churn/blob/c4391a0fb11d53ede8f9090f2015a82a2d0eda1d/Image/LightGBM.png)
![alt text](https://github.com/minhD03/Customer-Churn/blob/9fb6c96741d46aa75b0769f35a3a85ea3d62dafd/Image/Result%20Example.png)


## 5. Conclusions:
Based on the result from Notebook and the report, I believe that LightGBM should be more preferable Random Forest Classifier, especially for Customer Churn. Unlike Random Forest, which requires careful depth tuning to avoid underfitting or overfitting, LightGBM consistently delivers high performance with fewer iterations and better scalability. The gain and split-based feature importance outputs further confirm that LightGBM leverages nuanced patterns and prioritizes impactful variables, making it ideal for production-grade modeling where interpretability, speed and precision are critical. While Random Forest remains a reliable baseline, LightGBM’s advanced architecture and adaptability make it the model of choice for high-stakes, data-rich environments.
