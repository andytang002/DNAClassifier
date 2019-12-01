import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# Function to convert sequence strings into k-mer words
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1


human_data = pd.read_table('human_data.txt')
chimp_data = pd.read_table('chimp_data.txt')
dog_data = pd.read_table('dog_data.txt')

human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
human_data = human_data.drop('sequence', axis=1)
chimp_data['words'] = chimp_data.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp_data = chimp_data.drop('sequence', axis=1)
dog_data['words'] = dog_data.apply(lambda x: getKmers(x['sequence']), axis=1)
dog_data = dog_data.drop('sequence', axis=1)



# Use scikit-learn natural language processing tools to do k-mer counting, 
human_texts = list(human_data['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
y_data_human = human_data.iloc[:, 0].values

# Do the same for chimpanzee and dog
chimp_texts = list(chimp_data['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
y_data_chimp = chimp_data.iloc[:, 0].values                       

dog_texts = list(dog_data['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
y_data_dog = dog_data.iloc[:, 0].values


# Apply the BAG of WORDS using CountVectorizer using NLP

# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
cv = CountVectorizer(ngram_range=(4,4))
X_human = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)

print("human")
print(X_human.shape)
print("chimp")
print(X_chimp.shape)
print("dog")
print(X_dog.shape)


# Split human dataset into the training and test sets
X_train_human, X_test_human, y_train_human, y_test_human = train_test_split(X_human, 
                                                    y_data_human, 
                                                    test_size = 0.20, 
                                                    random_state=42)
# Do the same for chimp 
X_train_chimp, X_test_chimp, y_train_chimp, y_test_chimp = train_test_split(X_chimp,
                                                                            y_data_chimp,
                                                                            test_size=0.20,
                                                                            random_state=42)

# Do the same for dog 
X_train_dog, X_test_dog, y_train_dog, y_test_dog = train_test_split(X_dog, y_data_dog,
                                                            test_size=0.20, random_state=42)


# print(X_train_human.shape)
# print(X_test_human.shape)

### Multinomial Naive Bayes Classifier ###
classifier_human = MultinomialNB(alpha=0.1)
classifier_human.fit(X_train_human, y_train_human)

classifier_chimp = MultinomialNB(alpha=0.15)
classifier_chimp.fit(X_train_chimp, y_train_chimp)

classifier_dog = MultinomialNB(alpha=0.75)
classifier_dog.fit(X_train_dog, y_train_dog)


y_pred_human = classifier_human.predict(X_test_human)

y_pred_chimp = classifier_chimp.predict(X_test_chimp)

y_pred_dog = classifier_dog.predict(X_test_dog)


print("human gene classcification accuracy")
accuracy, precision, recall, f1 = get_metrics(y_test_human, y_pred_human)
print("accuracy = %.3f \n" % (accuracy))

print("chimp gene classcification accuracy")
accuracy, precision, recall, f1 = get_metrics(y_test_chimp, y_pred_chimp)
print("accuracy = %.3f \n" % (accuracy))

print("dog gene classcification accuracy")
accuracy, precision, recall, f1 = get_metrics(y_test_dog, y_pred_dog)
print("accuracy = %.3f \n" % (accuracy))

print("gene similarity test")
print("human model vs chimp test data")
y_pred_chimp_on_human = classifier_human.predict(X_chimp)
accuracy, precision, recall, f1 = get_metrics(y_data_chimp, y_pred_chimp_on_human)
print("accuracy = %.3f \n" % (accuracy))

print("human model vs dog test data")
y_pred_dog_on_human = classifier_human.predict(X_dog)
accuracy, precision, recall, f1 = get_metrics(y_data_dog, y_pred_dog_on_human)
print("accuracy = %.3f \n" % (accuracy))

print("chimp model vs human test data")
y_pred_human_on_chimp = classifier_chimp.predict(X_human)
accuracy, precision, recall, f1 = get_metrics(y_data_human, y_pred_human_on_chimp)
print("accuracy = %.3f \n" % (accuracy))

print("chimp model vs dog test data")
y_pred_dog_on_chimp = classifier_chimp.predict(X_dog)
accuracy, precision, recall, f1 = get_metrics(y_data_dog, y_pred_dog_on_chimp)
print("accuracy = %.3f \n" % (accuracy))

