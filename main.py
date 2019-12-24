import config
import argparse
from sklearn.linear_model import LogisticRegression
from LogitRegression import LogitRegression_Li
from data_utils import load_data


seed = 20191225


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--build_in",
                    action='store_true',
                    help="whether use build in LR class")

    args = parser.parse_args()

    # load data
    x_train, x_test, y_train, y_test = load_data(seed=seed)

    # build model
    if args.build_in:
        model = LogisticRegression(penalty='l2',solver='sag',multi_class='multinomial', max_iter=5000)
    else:
        model = LogitRegression_Li()

    # training
    model.fit(x_train, y_train)

    # eval
    print("Logistic Regression模型训练集的准确率：%.3f" %model.score(x_train, y_train))
    print("Logistic Regression模型测试集的准确率：%.3f" %model.score(x_test, y_test))


if __name__ == "__main__":
    main()