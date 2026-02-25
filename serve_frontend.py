"""Frontend HTTP Server for Dog Breed Classifier.

Simple HTTP server that serves static frontend files with CORS
headers enabled for API communication.

Features:
    - Basic HTTP server for HTML/CSS/JS files
    - CORS headers configured for API communication
    - Error handling for port conflicts
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Change to the frontend directory
os.chdir(r"C:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\dog-detector-frontend\public")

PORT = 3000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support.
    
    Extends SimpleHTTPRequestHandler to add CORS headers
    for cross-origin requests from frontend to API.
    """
    
    def end_headers(self):
        """Add CORS headers before finishing response."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

try:
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"🌐 Frontend server started at:")
        print(f"   URL: http://localhost:{PORT}")
        print(f"   Standalone: http://localhost:{PORT}/standalone.html")
        print("   Press Ctrl+C to stop the server")
        
        # Auto-open browser (commented out)
        # webbrowser.open(f'http://localhost:{PORT}/standalone.html')
        
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Server stopped")
except OSError as e:
    if "Address already in use" in str(e):
        print(f"❌ Error: Port {PORT} is already in use")
        print("   Try changing the port or closing other applications")
    else:
        print(f"❌ Error: {e}")