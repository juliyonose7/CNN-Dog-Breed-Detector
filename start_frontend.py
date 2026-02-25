#!/usr/bin/env python3
"""
servidor simple para frontend html css js del clasificador de razas
script que sirve archivos estaticos del frontend con servidor http basico
incluye soporte cors y auto-apertura del navegador para facilidad de uso

funcionalidades:
- servidor http simple para servir archivos html css js
- headers cors configurados para comunicacion con api
- verificacion de archivos requeridos antes de iniciar
- auto-apertura del navegador en la url correcta
- manejo de errores de puerto ocupado
- logs informativos para debugging
"""

# imports del sistema operativo y servidor web
import os                                      # operaciones del sistema operativo
import sys                                     # informacion y control del interprete
import webbrowser                              # control del navegador web
import threading                               # manejo de hilos para tareas concurrentes
import time                                    # operaciones de tiempo y delays
from pathlib import Path                       # manejo moderno de rutas de archivos
from http.server import HTTPServer, SimpleHTTPRequestHandler  # servidor http basico

# clase personalizada de request handler que agrega soporte cors
# extiende simplehttrequesthandler para permitir comunicacion con api
class CORSRequestHandler(SimpleHTTPRequestHandler):
    """handler con soporte para cors habilitado"""
    
    # sobrescribe end_headers para agregar headers cors necesarios
    def end_headers(self):
        # permite requests desde cualquier origen star es permisivo
        self.send_header('Access-Control-Allow-Origin', '*')
        
        # metodos http permitidos para la api
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        
        # headers permitidos en requests cors
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # llama al metodo padre para completar headers
        super().end_headers()
    
    # maneja requests options necesarios para cors preflight
    def do_OPTIONS(self):
        self.send_response(200)  # respuesta exitosa
        self.end_headers()       # termina headers cors

# funcion principal que inicia el servidor frontend en puerto especificado
# maneja toda la logica de configuracion y inicio del servidor http
def start_frontend_server(port=3000):
    """iniciar servidor para frontend con puerto personalizable"""
    
    # cambia el directorio de trabajo al directorio del script
    # esto garantiza que los archivos se sirvan desde la ubicacion correcta
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    print(f"🌐 Iniciando servidor frontend en puerto {port}...")
    print(f"📁 Directorio: {frontend_dir}")
    
    try:
        # crea servidor http con handler personalizado que incluye cors
        server = HTTPServer(('localhost', port), CORSRequestHandler)
        
        # mensajes informativos para el usuario
        print(f"✅ Servidor frontend iniciado en: http://localhost:{port}")
        print(f"📄 Página principal: http://localhost:{port}/simple_frontend_119.html")
        print("\n🔧 Asegúrate de que la API esté ejecutándose en puerto 8000")
        print("   Ejecuta: python testing_api_119_classes.py")
        print("\n⏹️  Presiona Ctrl+C para detener el servidor")
        
        # funcion para abrir navegador automaticamente despues de delay
        def open_browser():
            time.sleep(2)  # espera 2 segundos para que el servidor este listo
            webbrowser.open(f"http://localhost:{port}/simple_frontend_119.html")
        
        # ejecuta apertura de navegador en hilo separado para no bloquear
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True  # hilo daemon termina con programa principal
        browser_thread.start()
        
        # inicia servidor en loop infinito hasta interrupcion manual
        server.serve_forever()
        
    except KeyboardInterrupt:
        # manejo graceful de interrupcion ctrl+c
        print("\n🛑 Deteniendo servidor frontend...")
        server.shutdown()     # detiene servidor gracefully
        server.server_close() # cierra socket del servidor
        print("✅ Servidor frontend detenido")
        
    except OSError as e:
        # manejo de errores de sistema operativo como puerto ocupado
        if "Address already in use" in str(e):
            print(f"❌ Error: Puerto {port} ya está en uso")
            print(f"💡 Intenta con otro puerto o detén el proceso que usa el puerto {port}")
        else:
            print(f"❌ Error al iniciar servidor: {e}")
        sys.exit(1)  # termina programa con codigo de error

# verifica que todos los archivos necesarios del frontend esten presentes
# previene errores al intentar servir archivos inexistentes
def check_files():
    """verificar que los archivos necesarios existan antes de iniciar"""
    
    # lista de archivos criticos requeridos para el funcionamiento
    required_files = [
        "simple_frontend_119.html",  # pagina principal del frontend
        "styles.css",                # estilos visuales
        "app.js"                     # logica javascript
    ]
    
    missing_files = []  # lista para acumular archivos faltantes
    
    # verifica existencia de cada archivo requerido
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)  # agrega a lista de faltantes
    
    # si hay archivos faltantes, informa al usuario
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
    python start_frontend.py          # Puerto 3000
    python start_frontend.py 8080     # Puerto 8080

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
    
    # Verificar archivos
    if not check_files():
        print("\n💡 Asegúrate de ejecutar este script en el directorio con los archivos del frontend")
        return
    
    # Iniciar servidor
    start_frontend_server(port)

if __name__ == "__main__":
    main()