import pandas as pd
import numpy as np

def gradient_descent(x1: list[float], 
                     x2: list[float], 
                     x3: list[float], 
                     x4: list[float], 
                     x5: list[float],
                     x6: list[float],
                     x7: list[float],
                     y: list[int]):
    
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w0 = 0
    n = len(x1)
    epochs = 10000
    learning_rate = 0.0001

    for _ in range(epochs):
        y_predicted = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7 + w0
        cost =  (1/n) * sum([val**2 for val in (y-y_predicted)])
        grad_1 = -(2/n)*sum(x1*(y-y_predicted))
        grad_2 = -(2/n)*sum(x2*(y-y_predicted))
        grad_3 = -(2/n)*sum(x3*(y-y_predicted))
        grad_4 = -(2/n)*sum(x4*(y-y_predicted))
        grad_5 = -(2/n)*sum(x5*(y-y_predicted))
        grad_6 = -(2/n)*sum(x6*(y-y_predicted))
        grad_7 = -(2/n)*sum(x7*(y-y_predicted))
        grad_8 = -(2/n)*sum(y-y_predicted)

        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2
        w3 = w3 - learning_rate * grad_3
        w4 = w4 - learning_rate * grad_4
        w5 = w5 - learning_rate * grad_5
        w6 = w6 - learning_rate * grad_6
        w7 = w7 - learning_rate * grad_7
        w0 = w0 - learning_rate * grad_8
        
        print(cost)


"""
1. GRE Scores ( out of 340 )
2. TOEFL Scores ( out of 120 )
3. University Rating ( out of 5 )
4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
5. Undergraduate GPA ( out of 10 )
6. Research Experience ( either 0 or 1 )
7. Chance of Admit ( ranging from 0 to 1 )
"""

def clean_data(df: pd.DataFrame):
    # Rename Column
    df.rename(columns={
        "LOR ": "LOR", 
        "Chance of Admit ": "Chance of Admit"}, inplace=True)
    
    gre_scores = df["GRE Score"]
    toefl_scores = df["TOEFL Score"]
    university_ratings = df["University Rating"]
    sop = df["SOP"]
    lor = df["LOR"]
    gpa = df['CGPA']
    research = df['Research']
    
    # Normalization
    gre_scores = gre_scores.apply(lambda x: (x - min(gre_scores)) / (max(gre_scores) - min(gre_scores)))
    toefl_scores = toefl_scores.apply(lambda x: (x - min(toefl_scores)) / (max(toefl_scores) - min(toefl_scores)))
    university_ratings = university_ratings.apply(lambda x: (x - min(university_ratings)) / (max(university_ratings) - min(university_ratings)))
    sop = sop.apply(lambda x: (x - min(sop)) / (max(sop) - min(sop)))
    lor = lor.apply(lambda x: (x - min(lor)) / (max(lor) - min(lor)))
    gpa = gpa.apply(lambda x: (x - min(gpa)) / (max(gpa) - min(gpa)))
    
    return gre_scores, toefl_scores, university_ratings, sop, lor, gpa, research
    
if __name__ == "__main__":
    # Read CSV
    df = pd.read_csv("admission.csv")

    # Clean Data
    gre_scores, toefl_scores, university_ratings, sop, lor, gpa, research = clean_data(df)

    # Gradient Descent 
    gradient_descent(gre_scores, toefl_scores, university_ratings, sop, lor, gpa, research, df['Chance of Admit'])