from sklearn.linear_model import LinearRegression

# features (hours studied)
x=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

#  label (exam scores)
y=[30,40,45,55,65,70,80,85,90,95]

# model training
model=LinearRegression()
model.fit(x,y)

# prediction 6th hour
prediction=model.predict([[6]])
print(round(prediction[0], 2))

import matplotlib.pyplot as plt
import numpy as np 

x=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([30,40,45,55,65,70,80,85,90,95])

# calculate line y=mx+b m=slope ,b=where the line start
m=model.coef_[0]
b=model.intercept_
y_pred=m*x+b

plt.scatter(x,y,color="red",label="actual data")
plt.plot(x,y_pred,color="green",label="regression line")

plt.xlabel('hours  studied')
plt.ylabel('exams score')
plt.title('linear regression')
plt.legend()
plt.show()

score=model.predict([[6]])
print("predicted score",score[0])