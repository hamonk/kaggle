""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import confusion_matrix


def transform_data_frame(df):
    # female = 0, Male = 1
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df['Miss']   = df['Name'].str.contains('Miss\.').astype(int)
    df['Mr']     = df['Name'].str.contains('Mr\.').astype(int)
    df['Master'] = df['Name'].str.contains('Master\.').astype(int)
    df['Mrs']    = df['Name'].str.contains('Mrs\.').astype(int)

    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

    Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

    df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = df['Age'].dropna().median()
    if len(df.Age[ df.Age.isnull() ]) > 0:
        df.loc[ (df.Age.isnull()), 'Age'] = median_age

    df['Adult'] = (df.Age > 17).astype(int)

    # All the missing Fares -> assume median of their respective class
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

    return df
    
to_drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId']

"""
 TRAIN DATA
"""
train_df = pd.read_csv('input/train.csv', header=0)        # Load the train file into a dataframe
train_df = transform_data_frame(train_df)

ids_train = train_df['PassengerId'].values
survived_train = train_df['Survived'].values


"""
OVERWRITE
"""
test_df = train_df[-100:]
train_df = train_df[:-100]
reference_values = test_df.Survived

test_df = test_df.drop('Survived', axis=1)
print len(test_df)
print len(train_df)

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(to_drop, axis=1) 

"""
 TEST DATA
"""
#test_df = pd.read_csv('input/test.csv', header=0)        # Load the test file into a dataframe
#test_df = transform_data_frame(test_df)


# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(to_drop, axis=1) 

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.sort(axis=1).values
test_data = test_df.sort(axis=1).values

print 'Training...'
forest = RandomForestClassifier(n_estimators=100,verbose=1)
forest = forest.fit( train_data[:,:-1], train_data[:,-1] )
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
names = []

for f in range(len(importances)):
    print("%d. feature %d: %s (%f)" % (f + 1, indices[f], train_df.columns[1:][indices[f]], importances[indices[f]]))
    names.append(train_df.columns[1:][indices[f]])
    

# Plot the feature importances of the forest
print indices
print importances
plt.figure().set_figwidth(10)
plt.title("Feature importances")
plt.bar(range(len(importances)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(importances)), names)
plt.xlim([-1, 10])
plt.show()


print 'Predicting...'
output = forest.predict(test_data).astype(int)

predictions_file = open("output/output.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Died','Survived'], rotation=45)
    plt.yticks(tick_marks, ['Died','Survided'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)


# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
    
    
    
