📦 ezyml — Train and Export ML Models in 1 Line
ezyml is a lightweight Python and CLI tool to train, evaluate, and export ML models for classification, regression, clustering, and dimensionality reduction — all in a single command or function call.

🌟 Features
✅ Auto-detects task (classification / regression / clustering / PCA)
✅ Trains supported models with proper preprocessing
✅ Saves .pkl model and .json metrics
✅ Works as both a Python API and CLI tool
✅ Built-in support for 20+ ML models
✅ Optional dimensionality reduction with PCA/t-SNE
✅ Exportable model + report with 1 line

📦 Installation
pip install ezyml

💻 CLI Usage
🧠 Train a Classification Model
ezyml train 

--data data.csv 

--target label 

--model xgboost 

--output model.pkl 

--report report.json

📈 Train a Regression Model
ezyml train --data house.csv --target price --model lasso --output house_model.pkl

🔍 Clustering
ezyml train --data user_vectors.csv --model dbscan --task clustering

📉 Dimensionality Reduction (PCA)
ezyml reduce --data image_data.csv --model pca --components 2 --output pca_result.csv

🧪 Python API Usage
from ezyml import EZTrainer

Classification example
trainer = EZTrainer(data='heart.csv', target='label', model='naive_bayes')
trainer.train()
trainer.save_model('heart_model.pkl')
trainer.save_report('heart_report.json')

PCA example
trainer = EZTrainer(data='high_dim.csv', model='pca', task='dim_reduction', n_components=2)
trainer.train()
trainer.save_transformed('pca_output.csv')

🧰 Supported Tasks and Models
🧠 Classification Models
Model Name

Code ID

Logistic Regression

logistic_regression

Random Forest

random_forest

XGBoost Classifier

xgboost

SVM (Linear)

svm

Naive Bayes

naive_bayes

Gradient Boosting

gradient_boosting

Extra Trees

extra_trees

K-Nearest Neighbors

knn

📈 Regression Models
Model Name

Code ID

Linear Regression

linear_regression

Ridge Regression

ridge

Lasso Regression

lasso

ElasticNet

elasticnet

Random Forest Regr.

random_forest

XGBoost Regr.

xgboost

SVR

svr

Gradient Boosting

gradient_boosting

🔍 Clustering Models
Model Name

Code ID

KMeans

kmeans

DBSCAN

dbscan

Agglomerative Clustering

agglo

📉 Dimensionality Reduction
Method

Code ID

PCA

pca

t-SNE

tsne

📊 Metrics
Task

Metrics

Classification

Accuracy, F1, ROC AUC, Confusion Matrix

Regression

MAE, MSE, RMSE, R²

Clustering

Silhouette Score, n_clusters

PCA/t-SNE

None (returns transformed data)

🧠 API Reference: EZTrainer
EZTrainer(
data: str | pd.DataFrame,
target: str | None = None,
model: str = "random_forest",
task: str = "auto",  # or: classification, regression, clustering, dim_reduction
test_size: float = 0.2,
scale: bool = True,
n_components: int = None,  # For PCA or t-SNE
)

Methods
Method

Description

.train()

Trains the selected model

.save_model(path)

Saves the model to .pkl

.save_report(path)

Saves metrics/report as .json

.save_transformed(path)

Saves transformed data for PCA/t-SNE

.predict(X)

Returns predictions

🧰 CLI Reference

General training
ezyml train 

--data FILE.csv 

--target TARGET 

--model MODEL_NAME 

--output model.pkl 

--report metrics.json 

--task classification|regression|clustering

Dimensionality Reduction
ezyml reduce --data FILE.csv --model pca --components 2 --output reduced.csv

🧪 Examples
Classify Titanic Dataset with Extra Trees:

ezyml train --data titanic.csv --target Survived --model extra_trees --output model.pkl

Regress Housing Prices using Ridge:

ezyml train --data housing.csv --target price --model ridge --output model.pkl

Cluster Data:

ezyml train --data vectors.csv --model kmeans --task clustering

PCA:

ezyml reduce --data features.csv --model pca --components 2 --output pca_data.csv



📜 License
MIT License

👨‍💻 Author
Raktim Kalita
Machine Learning Engineer, Automator of Ideas 💡
GitHub: @raktimkalita