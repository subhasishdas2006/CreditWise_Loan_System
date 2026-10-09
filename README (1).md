# Loan Approval Prediction Using Supervised Machine Learning

A machine learning classification project that explores applicant and
loan-related attributes to predict whether a loan application is
approved. The project is implemented in Python using Jupyter Notebook
and compares multiple supervised learning algorithms.

## Project Overview

Loan approval decisions can depend on several applicant and financial
attributes. This project follows a typical machine learning workflow:
data inspection, preprocessing, exploratory data analysis, feature
encoding, model training, and evaluation.

**Project type:** Supervised Machine Learning --- Binary Classification\
**Target variable:** `Loan_Approved`

## Objectives

-   Explore the dataset and understand its features.
-   Handle missing numerical and categorical values.
-   Visualize distributions, class balance, and relationships between
    features.
-   Convert categorical variables into numerical representations.
-   Train and evaluate supervised classification models.
-   Compare model performance using standard classification metrics.
-   Experiment with feature engineering.

## Models Implemented

-   **Logistic Regression**
-   **K-Nearest Neighbors (KNN)**
-   **Gaussian Naive Bayes**

The notebook evaluates models using metrics such as accuracy, precision,
recall, F1-score, and a confusion matrix. Review the notebook outputs
for the actual results; no performance values are assumed in this
README.

## Dataset

The file `loan_approval_data.csv` contains applicant, financial, and
loan-related features. The target column is `Loan_Approved`.

Example feature categories include: - Applicant and co-applicant
income - Age and dependents - Credit score and debt-to-income ratio -
Savings, collateral value, and loan amount - Employment, education,
marital status, and property area - Loan purpose and loan term

> **Data note:** Confirm that the dataset is appropriate to share
> publicly and contains no personal, confidential, or sensitive
> real-world applicant information before publishing it on GitHub.

## Tech Stack

-   Python
-   Jupyter Notebook
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn

## Repository Structure

``` text
loan-approval-prediction-ml/
├── Loan_Approval_Prediction.ipynb
├── loan_approval_data.csv
├── README.md
└── requirements.txt
```

If your notebook has a different filename, update the structure above
accordingly.

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR_USERNAME/loan-approval-prediction-ml.git
cd loan-approval-prediction-ml
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter

``` bash
jupyter notebook
```

Open `Loan_Approval_Prediction.ipynb` and run the cells in order. Keep
`loan_approval_data.csv` in the same directory as the notebook because
the notebook loads it by filename.

## Workflow

1.  Load and inspect the dataset.
2.  Explore missing values and descriptive statistics.
3.  Impute missing numerical and categorical values.
4.  Perform exploratory data analysis and visualizations.
5.  Encode categorical variables and scale features.
6.  Split the data into training and testing sets.
7.  Train Logistic Regression, KNN, and Gaussian Naive Bayes models.
8.  Evaluate predictions with classification metrics.
9.  Experiment with squared features for selected variables.

## Evaluation

The notebook calculates classification metrics including:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix

Use the saved notebook outputs to report verified results here if
desired.

## Limitations

-   This is an educational minor project, not a production
    loan-underwriting system.
-   Model performance depends on dataset quality, feature selection,
    preprocessing, and evaluation design.
-   Predictions should not be used as the sole basis for real financial
    decisions. Loan decisions can have significant consequences and
    require fairness, privacy, regulatory, and domain review.

## Future Improvements

-   Use a scikit-learn `Pipeline` to keep preprocessing and model
    training together.
-   Compare models with cross-validation and consistent evaluation.
-   Add class-wise metrics and explain model errors.
-   Check for data leakage and assess fairness across relevant groups.
-   Record reproducible experiment results and the final selected model.

## Author

**Subhasish Das**

------------------------------------------------------------------------

If you find this project useful, feel free to star the repository.
