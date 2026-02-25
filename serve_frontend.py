import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Cambiar al directorio del frontend
os.chdir(r"C:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\dog-detector-frontend\public")

PORT = 3000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

try:
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"🌐 Servidor frontend iniciado en:")
        print(f"   URL: http://localhost:{PORT}")
        print(f"   Standalone: http://localhost:{PORT}/standalone.html")
        print("   Presiona Ctrl+C para detener el servidor")
        
        # No abrir navegador automáticamente para evitar conflictos
        # webbrowser.open(f'http://localhost:{PORT}/standalone.html')
        
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Servidor detenido")
except OSError as e:
    if "Address already in use" in str(e):
        print(f"❌ Error: El puerto {PORT} ya está en uso")
        print("   Intenta cambiar el puerto o cerrar otras aplicaciones")
    else:
        print(f"❌ Error: {e}")