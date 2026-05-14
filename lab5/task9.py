import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('src/nsw74psid1.csv')

X = data[['age', 'educ', 'black', 'hisp', 'marr', 'nodeg', 're74', 're75']]
y = data['re78']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

tree_model2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_model2.fit(X_train, y_train)
y_pred_tree2 = tree_model2.predict(X_test)
mse_tree2 = mean_squared_error(y_test, y_pred_tree2)
r2_tree2 = r2_score(y_test, y_pred_tree2)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

svm_model = SVR(kernel='rbf', C=1, epsilon=0.1)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

svm_model2 = SVR(kernel='rbf', C=1, epsilon=0.1)
svm_model2.fit(X_train_scaled, y_train_scaled)
y_pred_svm2 = svm_model2.predict(X_test_scaled)
mse_svm2 = mean_squared_error(y_test_scaled, y_pred_svm2)
r2_svm2 = r2_score(y_test_scaled, y_pred_svm2)

with open("result9.csv", "w", encoding='utf-8') as f:
    f.write("Модель,MSE,R²\n")
    f.write(f"Регрессионное дерево,{mse_tree:.2f},{r2_tree:.4f}\n")
    f.write(f"Регрессионное дерево с глубиной 3,{mse_tree2:.2f},{r2_tree2:.4f}\n")
    f.write(f"Линейная регрессия,{mse_lr:.2f},{r2_lr:.4f}\n")
    f.write(f"SVM-регрессия,{mse_svm:.2f},{r2_svm:.4f}\n")
    f.write(f"SVM-регрессия на нормированных данных,{mse_svm2:.2f},{r2_svm2:.4f}\n")
