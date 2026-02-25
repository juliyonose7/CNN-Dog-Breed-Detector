import http.server
import socketserver
import os
import threading
import time

# Cambiar to the directory of the frontend
frontend_dir = r"C:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\dog-detector-frontend\public"
os.chdir(frontend_dir)

PORT = 3000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Menos verbose logging
        if args[1] != '404':
            super().log_message(format, *args)

def start_server():
    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            print(f"🌐 Servidor frontend iniciado exitosamente!")
            print(f"   📁 Directorio: {frontend_dir}")
            print(f"   🌍 URL: http://localhost:{PORT}")
            print(f"   🎨 App: http://localhost:{PORT}/standalone.html")
            print(f"   📋 Index: http://localhost:{PORT}/index.html")
            print(f"   🛑 Para detener: Ctrl+C")
            print("=" * 50)
            
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Error: El puerto {PORT} ya está en uso")
            print("   💡 Solución: Cambia el puerto o cierra otras aplicaciones")
        else:
            print(f"❌ Error del servidor: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Servidor frontend detenido por el usuario")

if __name__ == "__main__":
    start_server()