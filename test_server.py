# !/usr/bin/env python3
"""
Technical documentation in English.
"""

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prueba de Comunicación</title>
    </head>
    <body>
        <h1>🧪 Prueba de Comunicación</h1>
        <input type="file" id="fileInput" accept="image/*">
        <button onclick="testUpload()">Probar Subida</button>
        <div id="result"></div>
        
        <script>
            async function testUpload() {
                const fileInput = document.getElementById('fileInput');
                const result = document.getElementById('result');
                
                if (fileInput.files.length === 0) {
                    result.innerHTML = '❌ Selecciona un archivo primero';
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                
                try {
                    result.innerHTML = '⏳ Enviando petición...';
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    result.innerHTML = '✅ Petición exitosa: ' + JSON.stringify(data);
                    
                } catch (error) {
                    result.innerHTML = '❌ Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
Technical documentation in English.
print(f"🔍 Method: {request.method}")
print(f"📁 Files: {list(request.files.keys())}")
    
if 'image' in request.files:
file = request.files['image']
print(f"📄 Name file: {file.filename}")
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
print("🧪 Iniciando server de prueba...")
print("📱 Abre: http://localhost:5001")
app.run(host='localhost', port=5001, debug=True)