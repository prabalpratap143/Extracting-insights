 # Extracting-insights
Extracting insights using visual and statistical exploration.

# Import basic library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load training data
train_df = pd.read_csv(r"C:\Users\praba\Downloads\train.csv")

# Using Basic Exploration
train_df.info()
print(train_df.describe())
print(train_df['Survived'].value_counts())
print(train_df['Sex'].value_counts())
print(train_df['Pclass'].value_counts())
print(train_df.isnull().sum())

# Using Pairplot
pair_df = train_df.copy()
pair_df['Sex'] = pair_df['Sex'].map({'male': 0, 'female': 1})
sns.pairplot(pair_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']], hue='Survived')
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()

# ---Observation: Survivors are more likely female, 1st class, and paid higher fares.

# Using Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pair_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ---Observation: Survival positively correlates with Sex (female), Fare; negatively with Pclass.

# Using Histograms
train_df['Age'].hist(bins=30, color='skyblue')
plt.title("Age Distribution")
plt.show()

# ---Observation: Majority of passengers are between 20–40 years.

train_df['Fare'].hist(bins=30, color='lightgreen')
plt.title("Fare Distribution")
plt.show()

# ---Observation: Fare distribution is skewed with most paying under 100.

# Using Boxplots
sns.boxplot(x='Survived', y='Age', data=train_df)
plt.title("Age vs Survival")
plt.show()

# ---Observation: Survivors are slightly younger on average.

sns.boxplot(x='Survived', y='Fare', data=train_df)
plt.title("Fare vs Survival")
plt.show()

# ---Observation: Survivors paid higher fares.

# Using Scatterplot
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train_df)
plt.title("Fare vs Age by Survival")
plt.show()

# ---Observation: Younger and higher-paying passengers had better survival.

# Using Barplots for Categorical Features
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title("Survival Rate by Gender")
plt.show()

# ---Observation: Females had a much higher survival rate.

sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title("Survival Rate by Class")
plt.show()

# ---Observation: First class had highest survival rate.

# Using Barplots
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title("Survival Rate by Embarkation")
plt.show()

# ---Observation: Passengers from Cherbourg (C) had higher survival.

# printing summary of the finding Summary
print("""
🔍 Summary of Findings:
- Females had the highest survival rate (~74%).
- First-class passengers were more likely to survive than second or third.
- Higher fare and younger age correlate with increased survival.
- Strong predictors: Sex, Pclass, Fare.
""")


