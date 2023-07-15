import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import numpy as np
import csv

def chart_fail(matr_fit_x, matr_fit_y):
    X_train, X_test, y_train, y_test = train_test_split(matr_fit_x, matr_fit_y,
                                                    train_size=0.7,
                                                    random_state=100)
    rmse_fail_all = []
    for i in range(1, 51):
        pls = PLSRegression(n_components=i)
        model = pls.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_fail = np.sqrt(np.mean(np.square(y_test - y_pred)))
        rmse_fail_all.append(rmse_fail)

    print(rmse_fail_all)
    plt.figure()
    plt.title("Зависимость RMSE от кол-ва компонент")
    plt.xlabel('кол-во компонент')
    plt.ylabel('RMSE')
    plt.plot([i for i in range(1,51)], rmse_fail_all)
    plt.savefig("chart.png")

    create_PLS(X_train,y_train,X_test,y_test)

def chart_r2(matr_fit_x, matr_fit_y):
    X_train, X_test, y_train, y_test = train_test_split(matr_fit_x, matr_fit_y,
                                                    train_size=0.7,
                                                    random_state=100)
    r2_all_train = []
    r2_all_test = []
    for i in range(1, 51):
        pls = PLSRegression(n_components=i)
        model = pls.fit(X_train, y_train)
        r_sq = model.score(X_train, y_train)
        r2_all_train.append(r_sq)
        r_sq = model.score(X_test, y_test)
        r2_all_test.append(r_sq)

    plt.figure()
    plt.title("Зависимость R^2 от кол-ва компонент (обучающая выборка)")
    plt.xlabel('кол-во компонент')
    plt.ylabel('R^2')
    plt.plot([i for i in range(1,51)], r2_all_train)
    plt.savefig("chart_r2_train.png")

    plt.figure()
    plt.title("Зависимость R^2 от кол-ва компонент (тестовая выборка)")
    plt.xlabel('кол-во компонент')
    plt.ylabel('R^2')
    plt.plot([i for i in range(1, 51)], r2_all_test)
    plt.savefig("chart_r2_test.png")

def create_PLS(X_train,y_train,X_test,y_test):
    pls = PLSRegression(n_components=7)
    model = pls.fit(X_train, y_train)
    r_sq = model.score(X_train, y_train)
    print("R^2 для обучающей выборки")
    print(r_sq)
    r_sq = model.score(X_test, y_test)
    print("R^2 для тестовой выборки")
    print(r_sq)
    y_pred = model.predict(X_test)

    with open("y_pred.csv", "w") as f:
        writer = csv.writer(f)
        for row in y_pred:
            writer.writerow(row)
