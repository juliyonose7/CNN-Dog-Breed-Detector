# !/usr/bin/env python3
"""
server simple for frontend html css js of the clasificador de breeds
script that sirve files estaticos of the frontend with server http basico
incluye soporte cors y auto-apertura of the navegador for facilidad de uso

funcionalidades:
- server http simple for servir files html css js
- headers cors configurados for comunicacion with API
- verificacion de files requeridos antes de start
- auto-apertura of the navegador en la url correcta
- manejo de errors de port ocupado
- logs informativos for debugging
"""

# imports of the system operativo y server web
import os                                      # operaciones of the system operativo
import sys                                     # informacion y control of the interprete
import webbrowser                              # control of the navegador web
import threading                               # manejo de hilos for tareas concurrentes
import time                                    # operaciones de tiempo y delays
from pathlib import Path                       # manejo moderno de rutas de files
from http.server import HTTPServer, SimpleHTTPRequestHandler  # server http basico

# class personalizada de request handler that agrega soporte cors
# extiende simplehttrequesthandler for permitir comunicacion with API
class CORSRequestHandler(SimpleHTTPRequestHandler):
    """handler with soporte for cors habilitado"""
    
    # sobrescribe end_headers for agregar headers cors necesarios
    def end_headers(self):
        # permite requests desde cualquier origen star es permisivo
        self.send_header('Access-Control-Allow-Origin', '*')
        
        # metodos http permitidos for la API
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        
        # headers permitidos en requests cors
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # llama to the method padre for completar headers
        super().end_headers()
    
    # maneja requests options necesarios for cors preflight
    def do_OPTIONS(self):
        self.send_response(200)  # respuesta exitosa
        self.end_headers()       # termina headers cors

# function principal that inicia el server frontend en port especificado
# maneja toda la logica de configuration y startup of the server http
def start_frontend_server(port=3000):
    """start server for frontend with port personalizable"""
    
    # cambia el directory de trabajo to the directory of the script
    # esto garantiza that los files se sirvan desde la ubicacion correcta
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    print(f"🌐 Iniciando servidor frontend en puerto {port}...")
    print(f"📁 Directorio: {frontend_dir}")
    
    try:
        # crea server http with handler personalizado that incluye cors
        server = HTTPServer(('localhost', port), CORSRequestHandler)
        
        # mensajes informativos for el user
        print(f"✅ Servidor frontend iniciado en: http://localhost:{port}")
        print(f"📄 Página principal: http://localhost:{port}/simple_frontend_119.html")
        print("\n🔧 Asegúrate de que la API esté ejecutándose en puerto 8000")
        print("   Ejecuta: python testing_api_119_classes.py")
        print("\n⏹️  Presiona Ctrl+C para detener el servidor")
        
        # function for abrir navegador automaticamente despues de delay
        def open_browser():
            time.sleep(2)  # espera 2 segundos for that el server this listo
            webbrowser.open(f"http://localhost:{port}/simple_frontend_119.html")
        
        # ejecuta apertura de navegador en hilo separado for no bloquear
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True  # hilo daemon termina with programa principal
        browser_thread.start()
        
        # inicia server en loop infinito hasta interrupcion manual
        server.serve_forever()
        
    except KeyboardInterrupt:
        # manejo graceful de interrupcion ctrl+c
        print("\n🛑 Deteniendo servidor frontend...")
        server.shutdown()     # detiene server gracefully
        server.server_close() # cierra socket of the server
        print("✅ Servidor frontend detenido")
        
    except OSError as e:
        # manejo de errors de system operativo como port ocupado
        if "Address already in use" in str(e):
            print(f"❌ Error: Puerto {port} ya está en uso")
            print(f"💡 Intenta con otro puerto o detén el proceso que usa el puerto {port}")
        else:
            print(f"❌ Error al iniciar servidor: {e}")
        sys.exit(1)  # Implementation note.

# verifica that all los files necesarios of the frontend esten presentes
# previene errors to the intentar servir files inexistentes
def check_files():
    """verificar that los files necesarios existan antes de start"""
    
    # list de files criticos requeridos for el funcionamiento
    required_files = [
        "simple_frontend_119.html",  # pagina principal of the frontend
        "styles.css",                # estilos visuales
        "app.js"                     # logica javascript
    ]
    
    missing_files = []  # list for acumular files faltantes
    
    # verifica existencia de cada file requerido
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)  # agrega a list de faltantes
    
    # if hay files faltantes, informa to the user
    if missing_files:
        print("❌ Archivos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        return False  # falla la verificacion
    
    print("✅ Todos los archivos necesarios están presentes")
    return True  # pasa la verificacion

def show_help():
    """Mostrar ayuda"""
    print("""
🐕 Dog Breed Classifier - Frontend Server

Uso:
    python start_frontend.py [puerto]

Argumentos:
    puerto    Puerto para el servidor frontend (default: 3000)

Ejemplos:
    python start_frontend.py          # Port 3000
    python start_frontend.py 8080     # Port 8080

Archivos necesarios:
    - simple_frontend_119.html (página principal)
    - styles.css (estilos CSS)
    - app.js (lógica JavaScript)

Notas:
    - La API debe estar ejecutándose en puerto 8000
    - El navegador se abrirá automáticamente
    - Usa Ctrl+C para detener el servidor
""")

def main():
    """Función principal"""
    
    # Verificar argumentos
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_help()
            return
        
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                print("❌ Error: El puerto debe estar entre 1024 y 65535")
                return
        except ValueError:
            print("❌ Error: El puerto debe ser un número válido")
            return
    else:
        port = 3000
    
    # Verificar files
    if not check_files():
        print("\n💡 Asegúrate de ejecutar este script en el directorio con los archivos del frontend")
        return
    
    # start server
    start_frontend_server(port)

if __name__ == "__main__":
    main()