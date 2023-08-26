# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/Thirukaalathessvarar-S/basic-nn-model/assets/121166390/ffd4b0c7-e964-4f70-adb4-51731caefa60)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Developed by: Thirukaalathessvarar S
Register No : 212222230161
```

### Importing module
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default
```

### Authenticate & Create Dataframe using Data in Sheets
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MyMLData').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

dataset1.head()
```

### Assign X and Y values
```
X = dataset1[['Input']].values
y = dataset1[['Output']].values

X
y
```

### Normalize the values & Split the data
```
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1
```
## Create a Neural Network & Train it:
### Create the model
```
ai=Sequential([
    Dense(7,activation='relu'),
    Dense(14,activation='relu'),
    Dense(1)
])
```
### Compile the model
```
ai.compile(optimizer='rmsprop',loss='mse')
```

### Fit the model
```

ai.fit(X_train1,y_train,epochs=2000)
ai.fit(X_train1,y_train,epochs=2000)
Include your code here
```

### Plot the Loss
```
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()
```

### Evaluate the model
```
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)
```

### Predict for some value
```
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```

## Dataset Information

![image](https://github.com/Thirukaalathessvarar-S/basic-nn-model/assets/121166390/04fade96-1160-498f-99d1-5a7bae02a389)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Thirukaalathessvarar-S/basic-nn-model/assets/121166390/20b6ac55-7a0a-41ce-8ee9-5968cad5b4a6)


### Test Data Root Mean Squared Error

![image](https://github.com/Thirukaalathessvarar-S/basic-nn-model/assets/121166390/8161bbef-e302-4ff1-be9e-376649259322)

### New Sample Data Prediction

![image](https://github.com/Thirukaalathessvarar-S/basic-nn-model/assets/121166390/2bd820b0-1ff8-4ef6-8ff2-5e2dde95382e)

## RESULT
Thus the neural network regression model for the given dataset is developed and executed successfully.
