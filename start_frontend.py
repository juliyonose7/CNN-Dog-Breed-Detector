# !/usr/bin/env python3
"""
Frontend Server for Dog Breed Classifier HTML/CSS/JS Application.

Simple HTTP server that serves static frontend files with CORS support
for communication with the backend API.

Features:
    - HTTP server for serving HTML/CSS/JS files
    - CORS headers configured for API communication
    - Required files verification before startup
    - Auto-opens browser to the correct URL
    - Error handling for port conflicts
    - Informative logs for debugging
"""

# System and web server imports
import os                                      # Operating system operations
import sys                                     # Interpreter control
import webbrowser                              # Web browser control
import threading                               # Thread handling
import time                                    # Time and delay operations
from pathlib import Path                       # Modern path handling
from http.server import HTTPServer, SimpleHTTPRequestHandler  # Basic HTTP server


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support enabled.
    
    Extends SimpleHTTPRequestHandler to add CORS headers
    necessary for frontend-API communication.
    """
    
    def end_headers(self):
        """Add CORS headers before completing response."""
        # Allows requests from any origin (permissive for development)
        self.send_header('Access-Control-Allow-Origin', '*')
        
        # Allowed HTTP methods for the API
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        
        # Allowed headers in CORS requests
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # Call parent method to complete headers
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests needed for CORS preflight."""
        self.send_response(200)  # Success
        self.end_headers()       # Complete with CORS headers

def start_frontend_server(port=3000):
    """Start the frontend HTTP server on specified port.
    
    Handles all configuration and startup logic for the HTTP server.
    
    Args:
        port: Port number to listen on (default: 3000).
    """
    # Change working directory to script location
    # This ensures files are served from the correct location
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    print(f" Starting frontend server on port {port}...")
    print(f" Directory: {frontend_dir}")
    
    try:
        # Create HTTP server with custom CORS-enabled handler
        server = HTTPServer(('localhost', port), CORSRequestHandler)
        
        # Informative messages for user
        print(f" Frontend server started at: http://localhost:{port}")
        print(f" Main page: http://localhost:{port}/simple_frontend_119.html")
        print("\n Make sure the API is running on port 8000")
        print("   Run: python testing_api_119_classes.py")
        print("\n  Press Ctrl+C to stop the server")
        
        # Function to auto-open browser after delay
        def open_browser():
            time.sleep(2)  # Wait 2 seconds for server to be ready
            webbrowser.open(f"http://localhost:{port}/simple_frontend_119.html")
        
        # Run browser opening in separate thread to not block
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True  # Daemon thread terminates with main program
        browser_thread.start()
        
        # Start server infinite loop until manual interruption
        server.serve_forever()
        
    except KeyboardInterrupt:
        # Graceful handling of Ctrl+C interruption
        print("\n Stopping frontend server...")
        server.shutdown()     # Stop server gracefully
        server.server_close() # Close server socket
        print(" Frontend server stopped")
        
    except OSError as e:
        # Handle OS errors like port already in use
        if "Address already in use" in str(e):
            print(f" Error: Port {port} is already in use")
            print(f" Try another port or stop the process using port {port}")
        else:
            print(f" Error starting server: {e}")
        sys.exit(1)  # Exit with error code

def check_files():
    """Verify that all required frontend files exist.
    
    Prevents errors when trying to serve non-existent files.
    
    Returns:
        bool: True if all files exist, False otherwise.
    """
    # List of critical files required for operation
    required_files = [
        "simple_frontend_119.html",  # Main frontend page
        "styles.css",                # Visual styles
        "app.js"                     # JavaScript logic
    ]
    
    missing_files = []  # List to accumulate missing files
    
    # Verify existence of each required file
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)  # Add to missing list
    
    # If there are missing files, inform user
    if missing_files:
        print(" Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False  # Verification failed
    
    print(" All required files are present")
    return True  # Verification passed


def show_help():
    """Display help information for the script."""
    print("""
 Dog Breed Classifier - Frontend Server

Usage:
    python start_frontend.py [port]

Arguments:
    port    Port for the frontend server (default: 3000)

Examples:
    python start_frontend.py          # Port 3000
    python start_frontend.py 8080     # Port 8080

Required files:
    - simple_frontend_119.html (main page)
    - styles.css (CSS styles)
    - app.js (JavaScript logic)

Notes:
    - The API must be running on port 8000
    - Browser will open automatically
    - Use Ctrl+C to stop the server
""")

def main():
    """Main entry point for frontend server."""
    
    # Verify arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_help()
            return
        
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                print(" Error: Port must be between 1024 and 65535")
                return
        except ValueError:
            print(" Error: Port must be a valid number")
            return
    else:
        port = 3000
    
    # Verify files
    if not check_files():
        print("\n Make sure to run this script in the directory with frontend files")
        return
    
    # Start server
    start_frontend_server(port)

if __name__ == "__main__":
    main()