# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# Load the dataset
train_file_path = "/Users/jcbele/Downloads/titanic_training_data.csv"
train_df = pd.read_csv(train_file_path)

# Display basic dataset information
print("Dataset Overview:")
print(train_df.info())  # Check data types and missing values
print(train_df.head())  # Preview the first few rows

# Drop irrelevant columns
# These columns provide little predictive power or contain too many unique values
train_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Handling missing values

# 1. Age: Fill missing values using the median 
age_imputer = SimpleImputer(strategy="median")
train_df["Age"] = age_imputer.fit_transform(train_df[["Age"]])

# 2. Embarked: Fill missing values with the most frequent category (mode)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

# Encoding categorical variables

# 1. Sex: Convert "male" to 0 and "female" to 1
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})

# 2. Embarked: Encode categorical values into numeric labels
train_df["Embarked"] = LabelEncoder().fit_transform(train_df["Embarked"])

#Standardize numerical faeatures
scaler = StandardScaler()
train_df[['Age', 'Fare']] = scaler.fit_transform(train_df[['Age', 'Fare']])
## Rather than use MinMax Scaling, StandardScaler is better for handling outliers ie. ticket costs being dramatically different

# Final check for missing values
print("Missing values after preprocessing:\n", train_df.isnull().sum())

# Display the cleaned dataset
print("Cleaned Data Sample:")
print(train_df.head())

# Split dataset into features and target variable
X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================
#    Post Cleaning Data Analysis
# =====================================

#First lets inspect survival rate by gender
plt.figure(figsize=(6, 4))
sns.barplot(x=["Male", "Female"], y=train_df.groupby("Sex")["Survived"].mean(), palette="coolwarm")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Probability")
plt.xlabel("Gender")
plt.ylim(0, 1)
plt.show()
# It is quite clear that the female gender was liklier to survive the accident

#Next lets inspect the survival rate by class AND gender
plt.figure(figsize=(8, 6))
sns.heatmap(pd.crosstab(train_df["Pclass"], train_df["Sex"], values=train_df["Survived"], aggfunc="mean"), 
            annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xlabel("Sex (0 = Male, 1 = Female)")
plt.ylabel("Passenger Class")
plt.title("Survival Rate by Class and Gender")
plt.show()
# With the given heatmap, it is clear that females survived at a much higher rate than men
# The highest amount of males per class to survive was 37% of males in first class
# There is a steep dropoff in male survivors in the the next two classes
# There seems to be a direct correlation between class and survival with a greater emphasis between women in 1st and 2nd class against women in 3rd class

# =====================================
#       Decision Tree Analysis
# =====================================

# Train a Decision Tree Classifier
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

# Accuracy test score
tree_accuracy = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Accuracy:", tree_accuracy)
# The decision tree shows an accuracy of 79.89%  

# Visualizing the decision tree
plt.figure(figsize=(12, 6))
plot_tree(tree_clf, feature_names=X.columns, class_names=['Did Not Survive', 'Survived'], filled=True, fontsize=8)
plt.title("Decision Tree Classifier")
plt.show()

# Feature Importance
feature_importance = tree_clf.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(feature_importance)), feature_importance[sorted_indices], align="center")
plt.xticks(range(len(feature_importance)), np.array(feature_names)[sorted_indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Decision Tree Model")
plt.show()

# =======================================
# PROBIT CLASSIFIER (LOGISTIC REGRESSION)
# =======================================

# Train Probit Model (Logistic Regression with probit-like approximation)
probit_clf = LogisticRegression(penalty=None, solver="lbfgs")
probit_clf.fit(X_train, y_train)
y_pred_probit = probit_clf.predict(X_test)

# Accuracy Score
probit_accuracy = accuracy_score(y_test, y_pred_probit)
print("Probit Model Accuracy:", probit_accuracy)
# The probit model shows an accuracy of 81.01%


# =========================
#       Random Forest 
# =========================

# Combine multiple desicion trees to reduce overfitting and improve generalization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialize Random Forest with optimized hyperparameters
rf_clf = RandomForestClassifier(
    n_estimators=200,  # Number of trees
    max_depth=7,  # Limit depth to avoid overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    criterion="entropy",  
    random_state=42
)

# Train the model
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_clf.predict(X_test)

# Evaluate performance
print("\nConfusion Matrix - Random Forest:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report - Random Forest:\n", classification_report(y_test, y_pred_rf))

# Accuracy Score
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)
# The new combination of multiple decision trees is now 82.68% accurate at predicting survivors making it more effective than the original tree and probit model

# =========================
#       COMPARISON
# =========================

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrices
cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_probit = confusion_matrix(y_test, y_pred_probit)

# Define plot function for heatmaps
def plot_confusion_matrix(cm, title, cmap="Blues"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=["Did Not Survive", "Survived"], 
                yticklabels=["Did Not Survive", "Survived"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# Plot all confusion matrices
plot_confusion_matrix(cm_tree, "Decision Tree Confusion Matrix", cmap="Blues")
plot_confusion_matrix(cm_rf, "Random Forest Confusion Matrix", cmap="Greens")
plot_confusion_matrix(cm_probit, "Probit Model Confusion Matrix", cmap="Oranges")


# The Decision Tree correctly identified 60 surviving passengers compared to the Probit model only identifying 55
# The Probit Model correctly identified 95 passenegers that did not survive compared to the D.T. identifying only 90
# The D.T. was overall a better classifier for predicting Ture Positives whereas the Probit was a better classifier for True Negatives
# The D.T. performs slightly better due to the smaller amount of missing actual survivors (25 in D.T. vs 30 in Probit)
# In terms of insurance purposes, the model which minimizes false negatives will be preferred
# The highest quality model is the Random Forest with an accuracy of almost 83%















