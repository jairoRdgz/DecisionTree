import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics


class DecisionTree:
    def __init__(self):
        self.dtc = tree.DecisionTreeClassifier()
        self.dt = pd.read_csv('Data/Dataset.csv', ';')
        self.train_model()

    def train_model(self):
        from sklearn.model_selection import train_test_split
        objetivo = self.dt['y'].values
        valores = self.dt['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
        x_train, x_test, y_train, y_test = train_test_split(valores, objetivo, test_size=0.2)

        self.dtc = self.dtc.fit(x_train, y_train)

        prediction_objective = self.dtc.predict(x_test)
        print(f"The acuracy of this prediction it's: {metrics.accuracy_score(y_test,prediction_objective)}%")
        return metrics.accuracy_score(y_test, prediction_objective)

    def draw_tree(self):
        from io import StringIO
        from IPython.display import Image, display
        import pydotplus

        out = StringIO()
        tree.export_graphviz(self.tdc, out_file = out)

        graph = pydotplus.graph_from_dot_data(out.getvalue())
        return graph.write_png('tree.png')

    def bank_prediction(self, bank_client):
        features = np.array([[bank_client['age'],
                              bank_client['job'],
                              bank_client['marital'],
                              bank_client['education'],
                              bank_client['default'],
                              bank_client['balance'],
                              bank_client['housing'],
                              bank_client['loan']]])
        predict_bank_decision = self.dtc.predict(features)
        result = np.int64(predict_bank_decision[0]).item()
        return result
