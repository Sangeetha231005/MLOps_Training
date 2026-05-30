from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_categories', methods=['GET'])
def get_categories():
    response = jsonify({
        'categories': util.get_categories()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_disaster', methods=['GET', 'POST'])
def predict_disaster():
    if request.method == 'POST':
        # Handles typical application/x-www-form-urlencoded POST data
        text = request.form.get('text', '')
    else:
        # Handles GET query string inputs
        text = request.args.get('text', '')

    result = util.predict_disaster(text)

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server for Disaster Classification...")
    util.load_saved_artifacts()
    app.run(port=5000, debug=True)
