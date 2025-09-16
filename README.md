# MBTI personality types prediction
MBTI stands for Myers-Briggs Type Indicator, which is a popular personality assessment method designed to categorize individuals into 16 distinct personality types based on their preferences in how they perceive the world and make decisions. 

According to Wikipedia, the MBTI method was constructed during World War II by Americans Katharine Cook Briggs and her daughter Isabel Briggs Myers. It was originally inspired by Carl Jung's theory on psychological types, introduced in 1921 in the book named Psychological Types.

> [!NOTE]<br>
> This project is conducted within a Linux environment. All processes and implementations have been carefully executed to ensure accuracy and reliability. For window user, please consider other alternative approaches.

### Overview
This project leverages a comprehensive dataset collected by Umair Zia from Kaggle to analyze and predict MBTI personality types. The dataset, consisting of over 100,000 samples, provides a rich resource for exploring the relationship between demographic factors, areas of interest, and personality scores. Each sample represents an individual observation, with various features contributing to the determination of their MBTI type. Below is an overview of the included features:
- **Demographic Information**: Data such as age, gender, and education level, offering insights into how personality types might correlate with these factors.
- **Interest Areas**: Indicators of preferences and hobbies, which may influence or reflect personality characteristics.
- **Personality Scores**: Quantitative measures across key psychological dimensions, aligning with MBTI type classification.

For more detail, please visit: [Predict People Personality Types](https://www.kaggle.com/datasets/stealthtechnologies/predict-people-personality-types/data)

### Problem Statement
The MBTI framework comprises 16 distinct personality types, making this a multinomial classification problem. To address this challenge, I'm going to evaluate the dataset using four machine learning models: 
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting Trees

These models will be implemented utilizing the scikit-learn library for efficient and consistent developing environmet. In addition, hypeparameter tuning will be conducted using Grid Search with Cross-Validation, ensuring optimal hyperparameter selection for each model. For the final model validation, K-Fold Cross-Validation will be used to assess performance robustness and reduce the risk of overfitting.

List of 16 personality types:
| MBTI Type | Nickname | Description |
|-----------|----------|--------------|
| ISTJ | The Inspector | Systematic and practical |
| ISFJ | The Protector | Nurturing and responsible |
| INFJ | The Advocate | Idealistic and insightful |
| INTJ | The Architect | Strategic and analytical |
| ISTP | The Virtuoso | Spontaneous and skillful |
| ISFP | The Composer | Artistic and gentle |
| INFP | The Mediator | Compassionate and imaginative |
| INTP | The Thinker | Logical and innovative |
| ESTP | The Dynamo | Energetic and action-oriented |
| ESFP | The Entertainer | Playful and sociable |
| ENFP | The Campaigner | Enthusiastic and creative |
| ENTP | The Debater | Quick-witted and challenging |
| ESTJ | The Executive | Organized and decisive |
| ESFJ | The Consul | Supportive and harmonizing |
| ENFJ | The Protagonist | Charismatic and inspiring |
| ENTJ | The Commander | Bold and strategic |

Where each characters stand for: 
1. Energy Orientation:
  - Extraversion (E): Draws energy from interacting with people and external activities.
  - Introversion (I): Prefers solitary activities and draws energy from inner thoughts.

2. Information Processing:
  - Sensing (S): Focuses on concrete facts, details, and the present.
  - Intuition (N): Focuses on patterns, possibilities, and the future.

3. Decision-Making:
  - Thinking (T): Makes decisions based on logic, fairness, and objective criteria.
  - Feeling (F): Makes decisions based on personal values, empathy, and the impact on others.

4. Approach to Structure:
  - Judging (J): Prefers structure, plans, and decisiveness.
  - Perceiving (P): Prefers flexibility, spontaneity, and keeping options open.

### Installation
<details open>
<summary> Installing dataset using `curl` and modify your path to your desired directory: </summary>

```bash
curl -L -o ~/Downloads/archive.zip\
https://www.kaggle.com/api/v1/datasets/download/stealthtechnologies/predict-people-personality-types
```
</details>

<details open>
<summary> To run program locally, please clone this repository and run: </summary>
  
  - Create virtual environment and activate it: 
```bash
python3.11 -m venv .env
source ./env/bin/activate
```
  - Upgrade pip and install required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
  - Run `test.py` to test model:
```
python test.py
```
</details>

To run on docker, run the following commands:
  - Build docker image and initialize it: 
```bash
docker build -t anyname .
docker run -it --rm -p 8185:8185 anyname
```
  - Run `test.py` to test model:
```bash
python test.py
```

### References
[Scikit-learn guide for model selection process](https://scikit-learn.org/stable/model_selection.html)

[Synthetic Personality Dataset for Predicting MBTI Types](https://www.kaggle.com/datasets/stealthtechnologies/predict-people-personality-types)

[DataTalksClub](https://github.com/DataTalksClub/machine-learning-zoomcamp)
