#!/usr/bin/env python3
"""
Flask Communication Test Server.

A simple Flask server for testing image upload and prediction endpoints.
Provides a minimal HTML interface for file uploads and validates server
communication before deploying to production.

Endpoints:
    GET /: Returns HTML test interface
    POST /predict: Accepts image uploads and returns mock predictions
"""

from flask import Flask, request, jsonify
import os

app = Flask(__name__)


@app.route('/')
def index():
    """
    Serve the test interface HTML page.
    
    Returns:
        str: HTML content for the test interface.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Communication Test</title>
    </head>
    <body>
        <h1> Communication Test</h1>
        <input type="file" id="fileInput" accept="image/*">
        <button onclick="testUpload()">Test Upload</button>
        <div id="result"></div>
        
        <script>
            async function testUpload() {
                const fileInput = document.getElementById('fileInput');
                const result = document.getElementById('result');
                
                if (fileInput.files.length === 0) {
                    result.innerHTML = ' Select a file first';
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                
                try {
                    result.innerHTML = ' Sending request...';
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    result.innerHTML = ' Request successful: ' + JSON.stringify(data);
                    
                } catch (error) {
                    result.innerHTML = ' Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
Technical documentation in English.
print(f" Method: {request.method}")
print(f" Files: {list(request.files.keys())}")
    
if 'image' in request.files:
file = request.files['image']
print(f" Name file: {file.filename}")
Technical documentation in English.
file.seek(0) # Reset file pointer
        
return jsonify({
'status': 'success',
Technical documentation in English.
'filename': file.filename
})
else:
Technical documentation in English.

if __name__ == "__main__":
print(" Starting server of test...")
print(" Abre: http://localhost:5001")
app.run(host='localhost', port=5001, debug=True)