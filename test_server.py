#!/usr/bin/env python3
"""
Servidor de prueba simple para diagnosticar comunicación
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
    print("🚀 ¡PETICIÓN RECIBIDA!")
    print(f"🔍 Método: {request.method}")
    print(f"📁 Archivos: {list(request.files.keys())}")
    
    if 'image' in request.files:
        file = request.files['image']
        print(f"📄 Nombre archivo: {file.filename}")
        print(f"📏 Tamaño: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer
        
        return jsonify({
            'status': 'success',
            'message': '¡Comunicación funciona!',
            'filename': file.filename
        })
    else:
        return jsonify({'error': 'No image found'})

if __name__ == "__main__":
    print("🧪 Iniciando servidor de prueba...")
    print("📱 Abre: http://localhost:5001")
    app.run(host='localhost', port=5001, debug=True)