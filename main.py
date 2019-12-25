import config
import argparse
from sklearn.linear_model import LogisticRegression
from LogitRegression import LogitRegression_Li
from data_utils import load_data
from sklearn import metrics


seed = 20191225


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--build_in",
                    action='store_true',
                    help="whether use build in LR class")
    parser.add_argument("--n_class",
                    type=int,
                    default=3,
                    help="the class number of irls dataset, 2 or 3")

    args = parser.parse_args()

    if args.n_class == 2:
        two_class = True
    else:
        two_class = False

    # load data
    x_train, x_test, y_train, y_test = load_data(seed=seed, two_class=two_class)

    # build model
    if args.build_in:
        model = LogisticRegression(penalty='l2',solver='newton-cg',multi_class='multinomial', max_iter=100)
    else:
        model = LogitRegression_Li(n_class=args.n_class)

    # training
    model.fit(x_train, y_train)

    # eval
    print("Logistic Regression模型训练集的准确率：%.3f" %model.score(x_train, y_train))
    print("Logistic Regression模型测试集的准确率：%.3f" %model.score(x_test, y_test))

    pred = model.predict(x_test)
    target_names = ['setosa', 'versicolor', 'virginica']
    print(metrics.classification_report(y_test, pred, target_names = target_names))


if __name__ == "__main__":
    main()