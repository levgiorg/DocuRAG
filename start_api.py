#!/usr/bin/env python3
"""API launcher script for DocuRAG system.

Ensures proper environment setup and launches FastAPI server with correct configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def find_python_executable():
    """Find the correct Python executable."""
    # Try current Python executable first
    current_python = sys.executable
    
    # Test if current Python has required packages
    try:
        import uvicorn
        import fastapi
        return current_python
    except ImportError:
        pass
    
    # Try common Python executables
    python_candidates = [
        "python3",
        "python",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3"
    ]
    
    for python_exec in python_candidates:
        try:
            # Test if executable exists and has required packages
            result = subprocess.run([
                python_exec, "-c", 
                "import uvicorn, fastapi; print('OK')"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "OK" in result.stdout:
                return python_exec
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    # If no suitable Python found, return current
    return current_python

def install_requirements():
    """Install required packages if not present."""
    python_exec = find_python_executable()
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        print("Installing requirements...")
        try:
            subprocess.run([
                python_exec, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("Requirements installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install requirements: {e}")
    else:
        print("No requirements.txt found, skipping package installation.")

def main():
    """Main launcher function."""
    print("🚀 Starting DocuRAG API Server...")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Find Python executable
    python_exec = find_python_executable()
    print(f"Using Python: {python_exec}")
    
    # Check if uvicorn is available
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not found, attempting to install requirements...")
        install_requirements()
    
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", str(script_dir))
    
    try:
        # Launch API server
        print("Starting FastAPI server on http://0.0.0.0:8000")
        print("API documentation available at: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run using uvicorn
        subprocess.run([
            python_exec, "-m", "uvicorn", 
            "api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        
        # Fallback: try running directly
        print("Attempting fallback launch method...")
        try:
            subprocess.run([python_exec, "api.py"], check=True)
        except subprocess.CalledProcessError as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            print("\nTroubleshooting:")
            print("1. Ensure all requirements are installed: pip install -r requirements.txt")
            print("2. Check Python version compatibility (requires 3.10+)")
            print("3. Verify FastAPI and uvicorn are properly installed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()