import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


df = pd.read_csv('sms_spam.csv')
df['type'].value_counts()
df['label_num'] = df['type'].apply(lambda x: 1 if x == 'spam' else 0)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.3, random_state=42)
print("train set:", X_train.shape)  # rows in train set
print("test set:", X_test.shape)  # rows in test set
lst = X_train.tolist()

# Applying Tf-IDF Vectorization
vectorizer = TfidfVectorizer(input = lst, lowercase = True, stop_words = "english")
train_transformed = vectorizer.fit_transform(X_train)
test_transformed = vectorizer.transform(X_test)

# Fit the transformed train data to the model.
model = MultinomialNB()
model.fit(train_transformed, y_train)
prediction = model.predict(test_transformed)
actual = y_test

print("Prediction:", list(prediction))
print("Actual:    ",list(actual))
matrix = confusion_matrix(prediction, actual)
matrix
precision = matrix[1][1]/(matrix[1][1]+matrix[0][1])
recall = matrix[1][1]/(matrix[1][1]+matrix[1][0])
f1score = matrix[1][1]/(matrix[1][1]+(matrix[1][0]+(matrix[0][1]/2)))

print("precision score:", precision)
print("recall score:", recall)
print("f1_score:", f1score)
messages = ["Congragulations! You have won a $10,000. Go to https://bit.ly/23343 to claim now.",
           "Get $10 Amazon Gift Voucher on Completing the Demo:- va.pcb3.in/ click this link to claim now",
           "You have won a $500. Please register your account today itself to claim now https://imp.com",
           "Please dont respond to missed calls from unknown international numbers Call/ SMS on winning prize. lottery as this may be fraudulent call."
          ]

message_transformed = vectorizer.transform(messages)

new_prediction = model.predict(message_transformed)

for i in range(len(new_prediction)):
    if new_prediction[i] == 0:
        print("Ham.")
    else:
        print("Spam.")
