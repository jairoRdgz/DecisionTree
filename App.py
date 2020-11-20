from flask import Flask, jsonify, request
from Model.DecisionTree import DecisionTree

tree = DecisionTree()

app = Flask(__name__)


@app.route('/')
def helloworld():
    return jsonify({'message': 'Hello World!'})


@app.route('/train')
def train_model():
    return jsonify({
        'message': 'Modelo entrenado',
        'Accuracy': tree.train_model()
    })


@app.route('/prediction', methods=['POST'])
def bank_prediction():
    return jsonify({
        'message': 'prediccion hecha',
        'prediction': tree.bank_prediction(request.json)
    })


if __name__ == '__main__':
    app.run()
