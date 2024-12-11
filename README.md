# Predicting Highly-Rated Recipes: A Machine Learning Analysis of Food.com Data

**Name(s)**: Stephanie Yue & Manjot Singh Samra

**Website Link**: Food.com


```python
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
pd.options.plotting.backend = 'plotly'

# from dsc80_utils import * 
```

## Introduction


```python
# TODO
import pandas as pd
import numpy as np

# Read the data
recipes = pd.read_csv('food_data/RAW_recipes.csv')
interactions = pd.read_csv('food_data/RAW_interactions.csv')

# Left merge recipes and interactions
merged_df = recipes.merge(interactions, left_on='id', right_on='recipe_id', how='left')

# Fill ratings of 0 with np.nan
merged_df['rating'] = merged_df['rating'].replace(0, np.nan)

# Calculate average rating per recipe
avg_ratings = merged_df.groupby('recipe_id')['rating'].mean()

# Add average ratings back to recipes dataset
recipes_with_ratings = recipes.merge(avg_ratings.to_frame('avg_rating'), 
                                   left_on='id', 
                                   right_index=True, 
                                   how='left')
```


```python
import pandas as pd
import numpy as np

recipes = pd.read_csv('food_data/RAW_recipes.csv')
interactions = pd.read_csv('food_data/RAW_interactions.csv')

merged_df = recipes.merge(interactions, left_on='id', right_on='recipe_id', how='left')
merged_df['rating'] = merged_df['rating'].replace(0, np.nan)
avg_ratings = merged_df.groupby('recipe_id')['rating'].mean()
recipes_with_ratings = recipes.merge(avg_ratings.to_frame('avg_rating'), 
                                   left_on='id', 
                                   right_index=True, 
                                   how='left')

recipes_with_ratings['calories'] = recipes_with_ratings['nutrition'].apply(lambda x: eval(x)[0])

recipes_with_ratings[['name', 'id', 'calories', 'avg_rating']]
print(recipes_with_ratings[['name', 'id', 'calories', 'avg_rating']].head().to_markdown(index=False))
```

    | name                                 |     id |   calories |   avg_rating |
    |:-------------------------------------|-------:|-----------:|-------------:|
    | 1 brownies in the world    best ever | 333281 |      138.4 |            4 |
    | 1 in canada chocolate chip cookies   | 453467 |      595.1 |            5 |
    | 412 broccoli casserole               | 306168 |      194.8 |            5 |
    | millionaire pound cake               | 286009 |      878.3 |            5 |
    | 2000 meatloaf                        | 475785 |      267   |            5 |



```python
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'  # or try 'notebook'

fig1 = px.histogram(recipes_with_ratings, 
                   x='avg_rating',
                   nbins=50,
                   title='Distribution of Recipe Ratings')
fig1.show()
fig1.write_html("assets/fig1.html", include_plotlyjs="cdn")

fig2 = px.scatter(recipes_with_ratings,
                 x='calories',
                 y='avg_rating',
                 title='Recipe Calories vs Average Rating')
fig2.show()
fig2.write_html("assets/fig2.html", include_plotlyjs="cdn")
```


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_20.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_20.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Data Cleaning and Exploratory Data Analysis


```python
# First, extract calories from nutrition data
recipes_with_ratings['calories'] = recipes_with_ratings['nutrition'].apply(lambda x: eval(x)[0])

# UNIVARIATE ANALYSIS
# 1. Distribution of Ratings
fig1 = px.histogram(recipes_with_ratings, 
                   x='avg_rating',
                   nbins=50,
                   title='Distribution of Recipe Ratings')
fig1.show()
fig1.write_html("assets/fig3.html", include_plotlyjs="cdn")

# 2. Distribution of Calories
fig2 = px.histogram(recipes_with_ratings, 
                   x='calories',
                   nbins=50,
                   title='Distribution of Recipe Calories')
fig2.update_layout(xaxis_range=[0, 2000])  # Limiting x-axis for better visibility
fig2.show()
fig2.write_html("assets/fig4.html", include_plotlyjs="cdn")

# 3. Distribution of Cooking Time
fig3 = px.histogram(recipes_with_ratings, 
                   x='minutes',
                   nbins=50,
                   title='Distribution of Cooking Times')
fig3.update_layout(xaxis_range=[0, 300])  # Limiting x-axis for better visibility
fig3.show()
fig3.write_html("assets/fig5.html", include_plotlyjs="cdn")

# 4. Distribution of Number of Steps
fig4 = px.histogram(recipes_with_ratings, 
                   x='n_steps',
                   title='Distribution of Number of Steps in Recipes')
fig4.show()
fig4.write_html("assets/fig6.html", include_plotlyjs="cdn")

# BIVARIATE ANALYSIS
# 1. Calories vs Ratings Scatter Plot
fig5 = px.scatter(recipes_with_ratings,
                 x='calories',
                 y='avg_rating',
                 title='Recipe Calories vs Average Rating',
                 opacity=0.6)
fig5.update_layout(xaxis_range=[0, 2000])
fig5.show()
fig5.write_html("assets/fig7.html", include_plotlyjs="cdn")

# 2. Cooking Time vs Ratings
fig6 = px.scatter(recipes_with_ratings,
                 x='minutes',
                 y='avg_rating',
                 title='Cooking Time vs Average Rating',
                 opacity=0.6)
fig6.update_layout(xaxis_range=[0, 300])
fig6.show()
fig6.write_html("assets/fig8.html", include_plotlyjs="cdn")

# 3. Number of Steps vs Rating Box Plot
fig7 = px.box(recipes_with_ratings,
              x='n_steps',
              y='avg_rating',
              title='Number of Steps vs Rating')
fig7.show()
fig7.write_html("assets/fig9.html", include_plotlyjs="cdn")

# 4. Rating vs Minutes Box Plot
recipes_with_ratings['rating_category'] = pd.cut(recipes_with_ratings['avg_rating'],
                                               bins=[0, 2, 3, 4, 4.5, 5],
                                               labels=['1-2', '2-3', '3-4', '4-4.5', '4.5-5'],
                                               include_lowest=True)

fig8 = px.box(recipes_with_ratings,
              x='rating_category',
              y='minutes',
              title='Rating Categories vs Cooking Time')
fig8.update_layout(yaxis_range=[0, 300])
fig8.show()
fig8.write_html("assets/fig10.html", include_plotlyjs="cdn")

# Additional Analysis: Correlation Matrix
numeric_cols = ['calories', 'minutes', 'n_steps', 'avg_rating']
correlation_matrix = recipes_with_ratings[numeric_cols].corr()

fig9 = px.imshow(correlation_matrix,
                 labels=dict(color="Correlation"),
                 color_continuous_scale="RdBu",
                 title="Correlation Matrix of Numeric Variables")
fig9.show()
fig9.write_html("assets/fig11.html", include_plotlyjs="cdn")
```


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_22.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Assessment of Missingness


```python
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# First, let's check which columns have missing values
missing_vals = recipes_with_ratings.isnull().sum()
print("Columns with missing values:")
print(missing_vals[missing_vals > 0])

# Step 3: Assessment of Missingness

# Let's analyze if rating missingness depends on calories
def compute_test_statistic(data, col1, col2):
    # Returns difference in means between groups where col2 is missing vs not missing
    missing_mask = data[col2].isna()
    return data[col1][missing_mask].mean() - data[col1][~missing_mask].mean()

# Perform permutation test for missingness dependency
def permutation_test(data, col1, col2, test_statistic, n_permutations=1000):
    observed_stat = test_statistic(data, col1, col2)
    permuted_stats = []
    
    for _ in range(n_permutations):
        shuffled_col2 = data[col2].copy()
        missing_mask = shuffled_col2.isna()
        shuffled_missing = np.random.permutation(missing_mask)
        temp_data = data.copy()
        temp_data[col2] = np.where(shuffled_missing, np.nan, shuffled_col2)
        permuted_stats.append(test_statistic(temp_data, col1, col2))
    
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    return observed_stat, p_value, permuted_stats

# Test if avg_rating missingness depends on calories
observed_stat_cal, p_value_cal, permuted_stats_cal = permutation_test(
    recipes_with_ratings, 'calories', 'avg_rating', compute_test_statistic
)

print("\nMissingness Analysis - Calories:")
print(f"Observed difference in means: {observed_stat_cal:.4f}")
print(f"P-value: {p_value_cal:.4f}")

# Visualize the permutation test results
fig_perm = px.histogram(x=permuted_stats_cal, 
                       title='Permutation Test Results: Rating Missingness vs Calories',
                       labels={'x': 'Difference in Mean Calories'})
fig_perm.add_vline(x=observed_stat_cal, line_color='red', 
                   annotation_text='Observed Statistic')
fig_perm.show()

fig_perm.write_html("assets/fig12.html", include_plotlyjs="cdn")

```

    Columns with missing values:
    name                  1
    description          70
    avg_rating         2609
    rating_category    2609
    dtype: int64
    
    Missingness Analysis - Calories:
    Observed difference in means: 87.8593
    P-value: 0.0000



<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_23.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Hypothesis Testing


```python
# Hypothesis Test 1: Calorie Content Effect on Ratings
print("Hypothesis Test 1: Effect of Calorie Content on Ratings")
print("Null: High-calorie recipes and low calorie recipes have the same average rating")
print("Alternative: High calorie recipes have a different average rating than low calorie recipes")

# Calculate median calories and split groups
median_calories = recipes_with_ratings['calories'].median()
high_cal_ratings = recipes_with_ratings[recipes_with_ratings['calories'] > median_calories]['avg_rating']
low_cal_ratings = recipes_with_ratings[recipes_with_ratings['calories'] <= median_calories]['avg_rating']

# Calculate observed difference in means for calories
observed_diff_cal = high_cal_ratings.mean() - low_cal_ratings.mean()

# Perform permutation test for calories
n_permutations = 1000
permuted_diffs_cal = []

for _ in range(n_permutations):
    shuffled_ratings = recipes_with_ratings['avg_rating'].sample(frac=1).reset_index(drop=True)
    high_cal_perm = shuffled_ratings[recipes_with_ratings['calories'] > median_calories]
    low_cal_perm = shuffled_ratings[recipes_with_ratings['calories'] <= median_calories]
    perm_diff = high_cal_perm.mean() - low_cal_perm.mean()
    permuted_diffs_cal.append(perm_diff)

# Calculate p-value for calories
p_value_cal = np.mean(np.abs(permuted_diffs_cal) >= np.abs(observed_diff_cal))

print("\nCalorie Test Results:")
print(f"Observed difference in means (high cal - low cal): {observed_diff_cal:.4f}")
print(f"P-value: {p_value_cal:.4f}")

# Hypothesis Test 2: Preparation Time Effect on Ratings
print("\nHypothesis Test 2: Effect of Preparation Time on Ratings")
print("Null: Recipes with longer and shorter preparation times have the same average rating")
print("Alternative: Recipes with longer preparation times have a different average rating than those with shorter preparation times")

# Calculate median preparation time and split groups
median_time = recipes_with_ratings['minutes'].median()
long_prep_ratings = recipes_with_ratings[recipes_with_ratings['minutes'] > median_time]['avg_rating']
short_prep_ratings = recipes_with_ratings[recipes_with_ratings['minutes'] <= median_time]['avg_rating']

# Calculate observed difference in means for preparation time
observed_diff_time = long_prep_ratings.mean() - short_prep_ratings.mean()

# Perform permutation test for preparation time
permuted_diffs_time = []

for _ in range(n_permutations):
    shuffled_ratings = recipes_with_ratings['avg_rating'].sample(frac=1).reset_index(drop=True)
    long_prep_perm = shuffled_ratings[recipes_with_ratings['minutes'] > median_time]
    short_prep_perm = shuffled_ratings[recipes_with_ratings['minutes'] <= median_time]
    perm_diff = long_prep_perm.mean() - short_prep_perm.mean()
    permuted_diffs_time.append(perm_diff)

# Calculate p-value for preparation time
p_value_time = np.mean(np.abs(permuted_diffs_time) >= np.abs(observed_diff_time))

print("\nPreparation Time Test Results:")
print(f"Observed difference in means (long prep - short prep): {observed_diff_time:.4f}")
print(f"P-value: {p_value_time:.4f}")

# Visualize results for both tests
fig1 = px.histogram(x=permuted_diffs_cal,
                   title='Hypothesis Test 1: High vs Low Calorie Ratings',
                   labels={'x': 'Difference in Mean Ratings'})
fig1.add_vline(x=observed_diff_cal, line_color='red',
               annotation_text='Observed Difference')
fig1.show()
fig1.write_html("assets/fig13.html", include_plotlyjs="cdn")

fig2 = px.histogram(x=permuted_diffs_time,
                   title='Hypothesis Test 2: Long vs Short Preparation Time Ratings',
                   labels={'x': 'Difference in Mean Ratings'})
fig2.add_vline(x=observed_diff_time, line_color='red',
               annotation_text='Observed Difference')
fig2.show()
fig2.write_html("assets/fig14.html", include_plotlyjs="cdn")

# Distribution comparisons
fig3 = px.histogram(recipes_with_ratings, 
                   x='avg_rating',
                   color=recipes_with_ratings['calories'] > median_calories,
                   barmode='overlay',
                   opacity=0.7,
                   labels={'color': 'High Calorie'},
                   title='Distribution of Ratings by Calorie Content')
fig3.show()
fig3.write_html("assets/fig15.html", include_plotlyjs="cdn")

fig4 = px.histogram(recipes_with_ratings, 
                   x='avg_rating',
                   color=recipes_with_ratings['minutes'] > median_time,
                   barmode='overlay',
                   opacity=0.7,
                   labels={'color': 'Long Preparation Time'},
                   title='Distribution of Ratings by Preparation Time')
fig4.show()
fig4.write_html("assets/fig16.html", include_plotlyjs="cdn")
```

    Hypothesis Test 1: Effect of Calorie Content on Ratings
    Null: High-calorie recipes and low calorie recipes have the same average rating
    Alternative: High calorie recipes have a different average rating than low calorie recipes
    
    Calorie Test Results:
    Observed difference in means (high cal - low cal): -0.0082
    P-value: 0.0740
    
    Hypothesis Test 2: Effect of Preparation Time on Ratings
    Null: Recipes with longer and shorter preparation times have the same average rating
    Alternative: Recipes with longer preparation times have a different average rating than those with shorter preparation times
    
    Preparation Time Test Results:
    Observed difference in means (long prep - short prep): -0.0315
    P-value: 0.0000



<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_24.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_24.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_24.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_24.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Framing a Prediction Problem


```python
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split

# Define functions
def extract_nutrition_columns(df):
    """
    Extracts nutrition information from the 'nutrition' column into separate columns.
    """
    df_copy = df.copy()
    nutrition_values = df_copy['nutrition'].apply(eval)
    df_copy['calories'] = nutrition_values.str[0]
    df_copy['total_fat'] = nutrition_values.str[1]
    df_copy['sugar'] = nutrition_values.str[2]
    df_copy['sodium'] = nutrition_values.str[3]
    df_copy['protein'] = nutrition_values.str[4]
    df_copy['saturated_fat'] = nutrition_values.str[5]
    df_copy['carbohydrates'] = nutrition_values.str[6]
    return df_copy

def prepare_classification_data(df):
    """
    Prepares features and target for predicting if a recipe will be highly rated.
    """
    df = extract_nutrition_columns(df)
    nutrition_cols = ['calories', 'total_fat', 'sugar', 'sodium', 
                      'protein', 'saturated_fat', 'carbohydrates']
    features = df[['minutes', 'n_steps', 'n_ingredients'] + nutrition_cols].copy()
    target = (df['avg_rating'] >= 4.5).astype(int)
    mask = ~target.isna()
    features = features[mask]
    target = target[mask]
    return features, target

# Prepare the data
features, target = prepare_classification_data(recipes_with_ratings)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Print and save information about the prediction problem
print("\nPrediction Problem Information:")
print(f"Total number of samples: {len(features)}")
print(f"Number of features: {features.shape[1]}")
print("\nFeatures used:")
for col in features.columns:
    print(f"- {col}")
print(f"\nProportion of highly-rated recipes: {target.mean():.3f}")
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Save feature information as an HTML file
features_info = pd.DataFrame({
    "Feature": features.columns,
    "Description": ["Time in minutes", "Number of steps", "Number of ingredients"] + 
                   ["Calories", "Total fat", "Sugar", "Sodium", 
                    "Protein", "Saturated fat", "Carbohydrates"]
})
features_info.to_html("assets/fig17.html")

# Create and save visualizations
fig1 = px.histogram(target, title="Distribution of Highly Rated Recipes (Binary Target)",
                    labels={'value': 'Highly Rated (1 = Yes, 0 = No)'})
fig1.show()
fig1.write_html("assets/fig18.html", include_plotlyjs="cdn")

# Visualize feature distributions
num = 18
for col in features.columns:
    fig = px.histogram(features, x=col, title=f"Distribution of {col}",
                       labels={col: col})
    fig.show()
    num = num + 1
    fig.write_html(f"assets/fig{num}.html", include_plotlyjs="cdn")

num = num + 1

# Print and save nutrition column sample
print("\nSample nutrition data:")
print(recipes_with_ratings['nutrition'].head())
recipes_with_ratings['nutrition'].head().to_frame().to_html(f"assets/fig{num}.html")
```

    
    Prediction Problem Information:
    Total number of samples: 83782
    Number of features: 10
    
    Features used:
    - minutes
    - n_steps
    - n_ingredients
    - calories
    - total_fat
    - sugar
    - sodium
    - protein
    - saturated_fat
    - carbohydrates
    
    Proportion of highly-rated recipes: 0.727
    
    Training set size: 67025
    Test set size: 16757



<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>




<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_28.html"
    frameborder="0"
    allowfullscreen
></iframe>



    
    Sample nutrition data:
    0         [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]
    1     [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0]
    2        [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]
    3    [878.3, 63.0, 326.0, 13.0, 20.0, 123.0, 39.0]
    4       [267.0, 30.0, 12.0, 12.0, 29.0, 48.0, 2.0]
    Name: nutrition, dtype: object


## Baseline Model


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np

# Let's use 2 features for baseline:
# - minutes: time to prepare (quantitative)
# - calories: calorie content (quantitative)
baseline_features = ['minutes', 'calories']

# Create X and y for the baseline model
X_baseline = recipes_with_ratings[baseline_features].copy()
y_baseline = (recipes_with_ratings['avg_rating'] >= 4.5).astype(int)

# Remove any rows with missing values
mask = ~(X_baseline.isna().any(axis=1) | y_baseline.isna())
X_baseline = X_baseline[mask]
y_baseline = y_baseline[mask]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_baseline, y_baseline, test_size=0.2, random_state=42
)

# Create baseline pipeline with balanced class weights
baseline_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        random_state=42,
        class_weight='balanced',  # Add class weights
        max_iter=1000  # Increase max iterations
    ))
])

# Fit the pipeline
baseline_pipeline.fit(X_train, y_train)

# Make predictions
train_predictions = baseline_pipeline.predict(X_train)
test_predictions = baseline_pipeline.predict(X_test)

# Print class distributions
print("Class Distributions:")
print("\nActual Training Distribution:")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nPredicted Training Distribution:")
print(pd.Series(train_predictions).value_counts(normalize=True))

# Calculate metrics
print("\nBaseline Model Performance")
print("\nTraining Set Performance:")
print(f"Accuracy: {accuracy_score(y_train, train_predictions):.3f}")
print(f"F1-Score: {f1_score(y_train, train_predictions):.3f}")

print("\nTest Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, test_predictions):.3f}")
print(f"F1-Score: {f1_score(y_test, test_predictions):.3f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, test_predictions, zero_division=0))

# Print feature coefficients
coefficients = pd.DataFrame({
    'Feature': baseline_features,
    'Coefficient': baseline_pipeline.named_steps['classifier'].coef_[0]
})
print("\nFeature Coefficients:")
print(coefficients)

# Look at some predictions
print("\nSample Predictions:")
sample_data = pd.DataFrame({
    'Actual': y_test[:5],
    'Predicted': test_predictions[:5],
    'Minutes': X_test['minutes'][:5],
    'Calories': X_test['calories'][:5]
})
print(sample_data)
```

    Class Distributions:
    
    Actual Training Distribution:
    avg_rating
    1    0.726609
    0    0.273391
    Name: proportion, dtype: float64
    
    Predicted Training Distribution:
    1    0.683223
    0    0.316777
    Name: proportion, dtype: float64
    
    Baseline Model Performance
    
    Training Set Performance:
    Accuracy: 0.590
    F1-Score: 0.710
    
    Test Set Performance:
    Accuracy: 0.592
    F1-Score: 0.712
    
    Classification Report on Test Set:
                  precision    recall  f1-score   support
    
               0       0.28      0.32      0.30      4513
               1       0.73      0.69      0.71     12244
    
        accuracy                           0.59     16757
       macro avg       0.51      0.51      0.51     16757
    weighted avg       0.61      0.59      0.60     16757
    
    
    Feature Coefficients:
        Feature  Coefficient
    0   minutes    -0.002492
    1  calories    -0.014594
    
    Sample Predictions:
           Actual  Predicted  Minutes  Calories
    53372       1          1        5     252.2
    75333       0          1       20     378.3
    46528       1          1      495     225.3
    53825       1          1       22     379.1
    69457       1          0       25     458.4


## Final Model


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

# Create focused feature set
def create_features(df):
    """Create enhanced but focused feature set for final model"""
    df = df.copy()
    
    # Extract nutrition values
    if 'total_fat' not in df.columns:
        nutrition_values = df['nutrition'].apply(eval)
        df['total_fat'] = nutrition_values.str[1]
        df['protein'] = nutrition_values.str[4]
    
    # Engineer key features
    df['calories_per_step'] = df['calories'] / df['n_steps']
    df['steps_per_ingredient'] = df['n_steps'] / df['n_ingredients']
    
    return df

# Use focused set of features
features = [
    'minutes', 'n_steps', 'n_ingredients', 'calories',
    'total_fat', 'protein', 'calories_per_step', 'steps_per_ingredient'
]

# Create enhanced dataset
enhanced_data = create_features(recipes_with_ratings)

# Prepare X and y
X = enhanced_data[features]
y = (enhanced_data['avg_rating'] >= 4.5).astype(int)

# Remove rows with missing values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Smaller parameter grid
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [5]
}

# Grid search with 3-fold CV
grid_search = GridSearchCV(
    final_pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

# Fit the model
print("Training model with 3-fold cross validation...")
grid_search.fit(X_train, y_train)

# Print best parameters
print("\nBest Parameters:", grid_search.best_params_)

# Evaluate on test set
y_pred = grid_search.predict(X_test)

# Print performance metrics
print("\nFinal Model Performance")
print("\nTest Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': grid_search.best_estimator_.named_steps['classifier'].feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance)

print("\nImprovement over Baseline:")
print("Baseline Test F1-Score: 0.712")
print(f"Final Model Test F1-Score: {f1_score(y_test, y_pred):.3f}")
```

    Training model with 3-fold cross validation...
    
    Best Parameters: {'classifier__max_depth': 20, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 100}
    
    Final Model Performance
    
    Test Set Performance:
    Accuracy: 0.670
    F1-Score: 0.790
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.30      0.17      0.22      4513
               1       0.74      0.85      0.79     12244
    
        accuracy                           0.67     16757
       macro avg       0.52      0.51      0.51     16757
    weighted avg       0.62      0.67      0.64     16757
    
    
    Feature Importances:
                    feature  importance
    6     calories_per_step    0.178831
    3              calories    0.177195
    5               protein    0.125463
    0               minutes    0.124861
    7  steps_per_ingredient    0.119478
    4             total_fat    0.119256
    2         n_ingredients    0.080315
    1               n_steps    0.074601
    
    Improvement over Baseline:
    Baseline Test F1-Score: 0.712
    Final Model Test F1-Score: 0.790


## Fairness Analysis


```python
from sklearn.metrics import precision_score
import plotly.express as px

# Define our groups based on calories
median_calories = X_test['calories'].median()

# Create our groups
low_cal_mask = X_test['calories'] <= median_calories
high_cal_mask = X_test['calories'] > median_calories

# Get predictions using our trained grid_search model
y_pred = grid_search.predict(X_test)

# Calculate precision for each group
low_cal_precision = precision_score(y_test[low_cal_mask], y_pred[low_cal_mask])
high_cal_precision = precision_score(y_test[high_cal_mask], y_pred[high_cal_mask])

# Calculate observed difference in precision
observed_diff = high_cal_precision - low_cal_precision

print("Fairness Analysis: High-Calorie vs Low-Calorie Recipes")
print("\nNull Hypothesis: Our model is fair. Its precision for high-calorie and")
print("low-calorie recipes are roughly the same, and any differences are due to random chance.")
print("\nAlternative Hypothesis: Our model is unfair. Its precision for high-calorie")
print("recipes is different from its precision for low-calorie recipes.")

print(f"\nPrecision for low-calorie recipes: {low_cal_precision:.3f}")
print(f"Precision for high-calorie recipes: {high_cal_precision:.3f}")
print(f"Observed difference in precision: {observed_diff:.3f}")

# Perform permutation test
n_permutations = 1000
permuted_diffs = []

for _ in range(n_permutations):
    # Shuffle the calorie labels
    permuted_mask = np.random.permutation(low_cal_mask)
    
    # Calculate precision for shuffled groups
    perm_low_precision = precision_score(y_test[permuted_mask], y_pred[permuted_mask])
    perm_high_precision = precision_score(y_test[~permuted_mask], y_pred[~permuted_mask])
    
    # Store difference in precision
    perm_diff = perm_high_precision - perm_low_precision
    permuted_diffs.append(perm_diff)

# Calculate two-sided p-value
p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

print(f"\nP-value: {p_value:.3f}")

# Visualize the permutation test results
fig = px.histogram(x=permuted_diffs,
                  title='Permutation Test Results: Differences in Precision Between High and Low Calorie Recipes',
                  labels={'x': 'Difference in Precision'})
fig.add_vline(x=observed_diff, line_color='red',
              annotation_text='Observed Difference')
fig.show()
fig.write_html(f"assets/fig31.html", include_plotlyjs="cdn")


# Make conclusion
alpha = 0.05
print("\nConclusion:")
if p_value < alpha:
    print(f"With a p-value of {p_value:.3f}, we reject the null hypothesis.")
    print("There is significant evidence that the model's precision differs between")
    print("high-calorie and low-calorie recipes.")
else:
    print(f"With a p-value of {p_value:.3f}, we fail to reject the null hypothesis.")
    print("There is not significant evidence that the model's precision differs between")
    print("high-calorie and low-calorie recipes.")
```

    Fairness Analysis: High-Calorie vs Low-Calorie Recipes
    
    Null Hypothesis: Our model is fair. Its precision for high-calorie and
    low-calorie recipes are roughly the same, and any differences are due to random chance.
    
    Alternative Hypothesis: Our model is unfair. Its precision for high-calorie
    recipes is different from its precision for low-calorie recipes.
    
    Precision for low-calorie recipes: 0.745
    Precision for high-calorie recipes: 0.728
    Observed difference in precision: -0.017
    
    P-value: 0.017



<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_35.html"
    frameborder="0"
    allowfullscreen
></iframe>



    
    Conclusion:
    With a p-value of 0.017, we reject the null hypothesis.
    There is significant evidence that the model's precision differs between
    high-calorie and low-calorie recipes.



```python

```


```python

```
