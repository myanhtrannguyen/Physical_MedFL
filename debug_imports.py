"""
Debug script to check if app_client.py can be imported correctly.
"""
import sys
print(f"Python version: {sys.version}")
print("Attempting to import app_client...")

try:
    import app_client
    print("Successfully imported app_client")
    print(f"app_client.app: {app_client.app}")
except Exception as e:
    print(f"Error importing app_client: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to import app_server...")
try:
    import app_server
    print("Successfully imported app_server")
    print(f"app_server.app: {app_server.app}")
except Exception as e:
    print(f"Error importing app_server: {e}")
    import traceback
    traceback.print_exc()
