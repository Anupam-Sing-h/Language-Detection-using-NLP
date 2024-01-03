import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Read the dataset
data = pd.read_csv("Language Detection.csv")

# Separate independent (X) and dependent (y) variables
independent = data["Text"]
dependent = data["Language"]

# Convert categorical variables to numerical
label_encoder = LabelEncoder()
dependent = label_encoder.fit_transform(dependent)

# Preprocess the text
data_list = [re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text.lower()) for text in independent]

# Bag of words model
cv = CountVectorizer()
independent = cv.fit_transform(data_list).toarray()

# Split train and testing dataset
i_train, i_test, d_train, d_test = train_test_split(independent, dependent, test_size=0.2)

# Training the model
model = MultinomialNB()
model.fit(i_train, d_train)

# Predict output of the test set
d_pred = model.predict(i_test)


# Evaluate the accuracy of our model (you can uncomment these lines if needed)
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# ac = accuracy_score(d_test, d_pred)
# cm = confusion_matrix(d_test, d_pred)
# print("Accuracy is :", ac)
# plt.figure(figsize=(15, 10))
# sns.heatmap(cm, annot=True)
# plt.show()

# Predict the data
def predict(text):
    independent = cv.transform([text]).toarray()
    lang = model.predict(independent)
    lang = label_encoder.inverse_transform(lang)
    print(f"The language is in {lang[0]}")


# Test the prediction
predict("നിറഞ്ഞു നിന്നു നിലവിലെ നക്ഷത്രങ്ങൾഎന്റെ മനസ്സ")
