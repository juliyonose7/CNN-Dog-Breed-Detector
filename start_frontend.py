# !/usr/bin/env python3
"""
server simple for frontend html css js of the clasificador of breeds
script that serves files estaticos of the frontend with server http basico
incluye soporte cors and auto-apertura of the navegador for facilidad of uso

funcionalidades:
- server http simple for servir files html css js
- headers cors configurados for comunicacion with API
- verificacion of files requeridos antes of start
- auto-apertura of the navegador en the url correcta
- handling of errors of port ocupado
- logs informativos for debugging
"""

# imports of the system operating and server web
import os                                      # operations of the system operating
import sys                                     # informacion and control of the interprete
import webbrowser                              # control of the navegador web
import threading                               # handling of hilos for tareas concurrentes
import time                                    # operations of time and delays
from pathlib import Path                       # handling moderno of paths of files
from http.server import HTTPServer, SimpleHTTPRequestHandler  # server http basico

# class personalizada of request handler that adds soporte cors
# extiende simplehttrequesthandler for permitir comunicacion with API
class CORSRequestHandler(SimpleHTTPRequestHandler):
    """handler with soporte for cors habilitado"""
    
    # sobrescribe end_headers for add headers cors necesarios
    def end_headers(self):
        # allows requests from cualquier origen star es permisivo
        self.send_header('Access-Control-Allow-Origin', '*')
        
        # methods http permitidos for the API
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        
        # headers permitidos en requests cors
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # llama to the method padre for completar headers
        super().end_headers()
    
    # maneja requests options necesarios for cors preflight
    def do_OPTIONS(self):
        self.send_response(200)  # respuesta exitosa
        self.end_headers()       # termina headers cors

# function main that inicia the server frontend en port especificado
# maneja toda the logica of configuration and startup of the server http
def start_frontend_server(port=3000):
    """start server for frontend with port personalizable"""
    
    # changes the directory of trabajo to the directory of the script
    # esto garantiza that the files se serve from the ubicacion correcta
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    print(f"🌐 Iniciando servidor frontend en puerto {port}...")
    print(f"📁 Directorio: {frontend_dir}")
    
    try:
        # creates server http with handler personalizado that incluye cors
        server = HTTPServer(('localhost', port), CORSRequestHandler)
        
        # mensajes informativos for the user
        print(f"✅ Servidor frontend iniciado en: http://localhost:{port}")
        print(f"📄 Página principal: http://localhost:{port}/simple_frontend_119.html")
        print("\n🔧 Asegúrate de que la API esté ejecutándose en puerto 8000")
        print("   Ejecuta: python testing_api_119_classes.py")
        print("\n⏹️  Presiona Ctrl+C para detener el servidor")
        
        # function for abrir navegador automaticamente despues of delay
        def open_browser():
            time.sleep(2)  # wait 2 segundos for that the server this listo
            webbrowser.open(f"http://localhost:{port}/simple_frontend_119.html")
        
        # ejecuta apertura of navegador en hilo separado for no bloquear
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True  # hilo daemon termina with programa main
        browser_thread.start()
        
        # inicia server en loop infinito until interrupcion manual
        server.serve_forever()
        
    except KeyboardInterrupt:
        # handling graceful of interrupcion ctrl+c
        print("\n🛑 Deteniendo servidor frontend...")
        server.shutdown()     # detiene server gracefully
        server.server_close() # cierra socket of the server
        print("✅ Servidor frontend detenido")
        
    except OSError as e:
        # handling of errors of system operating como port ocupado
        if "Address already in use" in str(e):
            print(f"❌ Error: Puerto {port} ya está en uso")
            print(f"💡 Intenta con otro puerto o detén el proceso que usa el puerto {port}")
        else:
            print(f"❌ Error al iniciar servidor: {e}")
        sys.exit(1)  # Implementation note.

# verifies that all the files necesarios of the frontend esten presentes
# previene errors to the intentar servir files inexistentes
def check_files():
    """verify that the files necesarios exist antes of start"""
    
    # list of files critical requeridos for the funcionamiento
    required_files = [
        "simple_frontend_119.html",  # pagina main of the frontend
        "styles.css",                # estilos visuales
        "app.js"                     # logica javascript
    ]
    
    missing_files = []  # list for acumular files faltantes
    
    # verifies existencia of cada file requerido
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)  # adds a list of faltantes
    
    # if hay files faltantes, informa to the user
    if missing_files:
        print("❌ Archivos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        return False  # falla the verificacion
    
    print("✅ Todos los archivos necesarios están presentes")
    return True  # pasa the verificacion

def show_help():
    """Show ayuda"""
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
    
    # Verify argumentos
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
    
    # Verify files
    if not check_files():
        print("\n💡 Asegúrate de ejecutar este script en el directorio con los archivos del frontend")
        return
    
    # start server
    start_frontend_server(port)

if __name__ == "__main__":
    main()