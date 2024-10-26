# Human Activity Recognition Using Smartphone Data

This project utilizes machine learning to classify human physical activities based on data collected from smartphone sensors. In this experiment, 30 volunteers aged 19 to 48 performed six activities while wearing a smartphone on their waist, capturing motion data through its embedded accelerometer and gyroscope. Using this dataset, the project aims to develop a model that can predict these activities accurately.

## Project Description

Human Activity Recognition (HAR) using smartphones is a valuable application in areas like health monitoring, fitness tracking, and elderly care. This project focuses on building and comparing multiple machine learning models to classify activities into one of the following categories:
- **Walking**
- **Walking Upstairs**
- **Walking Downstairs**
- **Sitting**
- **Standing**
- **Laying**

The dataset includes sensor readings segmented into labeled activities, which serve as input features for the model training and evaluation process.

### Key Steps:

1. **Data Loading and Preprocessing**
   - Import and clean the dataset by removing duplicates and handling any missing values.
   - Examine the class distribution to address potential imbalance issues.

2. **Exploratory Data Analysis (EDA)**
   - Visualize activity distribution to understand the dataset.
   - Analyze specific features, such as `tBodyAccMag-mean`, to see how they vary across activities.
   - Use PCA and t-SNE for dimensionality reduction and visual representation of activity clusters.

3. **Model Training and Evaluation**
   - Implement several machine learning models, including Logistic Regression, SVM, Decision Tree, and Random Forest, to classify the activities.
   - Apply hyperparameter tuning with cross-validation to optimize each model’s performance.
   - Evaluate the models using accuracy, confusion matrices, and other classification metrics.

4. **Results and Interpretation**
   - Report accuracy scores and visualize confusion matrices to compare model performance.
   - Save the best model configuration for future use or deployment.

### Requirements
To run this project, you will need:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Example usages 
Fitness and Exercise Tracking
Purpose: To provide real-time activity tracking and insights for fitness enthusiasts.
Example: Fitness apps on smartphones or wearables can classify activities such as walking, jogging, or stair climbing, allowing users to track their workout type, duration, and intensity. This can help users better understand their physical activity patterns and calories burned, encouraging a healthier lifestyle.

