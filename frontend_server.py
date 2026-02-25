"""
Frontend Static File Server Module
==================================

This module implements a simple HTTP server for serving the dog detector
frontend application. It handles static files with CORS support enabled
for cross-origin API requests.

Features:
    - Static file serving from the frontend public directory
    - CORS headers for frontend-backend communication
    - Minimal logging (suppresses 404 noise)
    - Easy startup on configurable port

Usage:
    Run this script to start the frontend server:
    $ python frontend_server.py
    
    Then access the application at:
    - http://localhost:3000/standalone.html - Main application
    - http://localhost:3000/index.html - Index page

Author: AI System
Date: 2024
"""

import http.server
import socketserver
import os
import threading
import time

# Change working directory to the frontend public folder
frontend_dir = r"C:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\dog-detector-frontend\public"
os.chdir(frontend_dir)

# Server configuration
PORT = 3000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP request handler with CORS support.
    
    Extends SimpleHTTPRequestHandler to add Cross-Origin Resource Sharing
    headers, enabling the frontend to communicate with APIs on different
    origins/ports.
    
    CORS Headers Added:
        - Access-Control-Allow-Origin: * (allows all origins)
        - Access-Control-Allow-Methods: GET, POST, OPTIONS
        - Access-Control-Allow-Headers: Content-Type
    """
    
    def end_headers(self):
        """
        Override to inject CORS headers before ending response headers.
        """
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def log_message(self, format, *args):
        """
        Override to reduce logging verbosity.
        
        Only logs requests that are not 404 errors to reduce console noise
        from missing favicon or other common missing resources.
        
        Args:
            format: Log message format string.
            *args: Format arguments (status code is typically args[1]).
        """
        # Suppress 404 errors from logging
        if args[1] != '404':
            super().log_message(format, *args)

def start_server():
    """
    Start the frontend HTTP server.
    
    Initializes and runs a TCP server on the configured PORT, serving
    static files from the frontend directory with CORS support.
    
    Handles common exceptions:
        - OSError: When the port is already in use
        - KeyboardInterrupt: For graceful shutdown via Ctrl+C
    """
    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            print(f" Frontend server started successfully!")
            print(f"    Directory: {frontend_dir}")
            print(f"    URL: http://localhost:{PORT}")
            print(f"    App: http://localhost:{PORT}/standalone.html")
            print(f"    Index: http://localhost:{PORT}/index.html")
            print(f"    To stop: Ctrl+C")
            print("=" * 50)
            
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f" Error: Port {PORT} is already in use")
            print("    Solution: Change the port or close other applications")
        else:
            print(f" Server error: {e}")
    except KeyboardInterrupt:
        print("\n Frontend server stopped by user")

if __name__ == "__main__":
    start_server()