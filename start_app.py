#!/usr/bin/env python3
"""Simple launcher for DocuRAG Streamlit app using correct Python environment."""

import os
import sys
import subprocess

def main():
    """Launch the Streamlit app with environment setup."""
    print("🚀 Starting DocuRAG...")
    
    # Set environment variables
    os.environ["STREAMLIT_TELEMETRY_DISABLED"] = "1"
    
    # Use the current Python interpreter
    python_exe = sys.executable
    
    try:
        # Launch streamlit
        cmd = [python_exe, "-m", "streamlit", "run", "app.py", "--server.headless", "true"]
        print(f"Running: {' '.join(cmd)}")
        
        # Run the command
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 DocuRAG stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting DocuRAG: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())