# Student-performance
# Student Performance Analysis

## Project Overview
This project explores the Student Performance dataset (1000 students, 8 features). The goal is to understand patterns in student scores and the impact of test preparation courses and other possible factors on the student scores.

## Session 1
- Dataset loaded successfully: 1000 rows, 8 columns, no missing values
- Math, reading, and writing scores strongly correlated(math-reading ≈ 0.89, math-writing ≈ 0.95) — performance transfers across subjects.
- Test preparation significantly improves scores:
  - Math: +5.6 points
  - Reading: +7.4 points
  - Writing: +9.9 points
- Boxplots confirm higher median and reduced spread for completed test prep
- Outliers exist but main patterns are clear

## Next Steps
- Encode categorical features
- Build regression models for predicting math, reading, and writing scores
- Evaluate model performance


## Session 2
- Feature Engineering: Applied one-hot encoding to 5 categorical columns (gender, race/ethnicity, parental level of education, lunch, test preparation course), expanding dataset to 16 binary features.

- Modeling Approach: Built separate Linear Regression models for each target (math, reading, writing scores) using 80/20 train-test split with random_state=42 for reproducibility.

- Model Performance (RMSE / R²):
    - Math	RMSE~14.2	R²=0.18	-Weak predictive power
    - Reading	RMSE~13.8	R²=0.16	-Weak predictive power
    - Writing	RMSE~13.3	R²=0.26	-Best performance, still limited
  
- Key Findings from Coefficients:
    - lunch_standard (+8 to +11 points): Strongest positive predictor across all subjects — socioeconomic status matters
    - test preparation course_none (-5 to -10 points): Not taking prep course hurts writing most (-10), math least (-5)
    - gender_male: Advantage in math (+5), disadvantage in reading/writing (-7 to -9)
    - race/ethnicity_group E: +10 points in math vs reference group A
    - Parental education effects: High school only = - 5 points vs master's degree baseline

## KEY TAKEAWAYS:
  | Demographics explain only 16-26% of score variance; Other factors (study habits, prior knowledge, school quality) play larger roles.
  
  | Test prep helps writing most, math least; Prep courses may focus on writing skills; math needs different interventions.

  | Socioeconomic status (`lunch`) consistently important; Policy interventions targeting economic barriers could help across all subjects.

  | Gender gaps vary by subject; STEM vs humanities support may need different targeting.
