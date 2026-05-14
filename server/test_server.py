from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    if request.method == 'POST':
        print(f"[TEST] POST received: {request.form}")
        return jsonify({"success": True, "message": "POST works!"})
    return jsonify({"success": True, "message": "GET works!"})

if __name__ == '__main__':
    print("Starting minimal test server...")
    serve(app, host='0.0.0.0', port=5001, threads=4)
