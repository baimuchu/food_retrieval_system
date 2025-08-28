import os
from pathlib import Path

# Test the path calculation from config.py
def get_api_key_from_file():
    key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'LiteLLM_Proxy_Key.txt')
    print(f"Config.py path calculation: {key_file}")
    print(f"File exists: {os.path.exists(key_file)}")
    return key_file

# Test alternative path calculation
def get_api_key_from_file_alt():
    current_file = Path(__file__).resolve()
    demo_dir = current_file.parent
    src_dir = demo_dir.parent
    project_root = src_dir.parent
    key_file = project_root / "LiteLLM_Proxy_Key.txt"
    print(f"Alternative path calculation: {key_file}")
    print(f"File exists: {key_file.exists()}")
    return str(key_file)

# Test from current working directory
def get_api_key_from_cwd():
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")
    
    # Look for prosusai_assignment_data directory
    for parent in [cwd] + list(cwd.parents):
        if (parent / "prosusai_assignment_data").exists():
            project_root = parent
            break
    else:
        project_root = cwd
    
    key_file = project_root / "LiteLLM_Proxy_Key.txt"
    print(f"From CWD path calculation: {key_file}")
    print(f"File exists: {key_file.exists()}")
    return str(key_file)

if __name__ == "__main__":
    print("=== Testing API Key File Paths ===")
    print()
    
    print("1. Config.py method:")
    get_api_key_from_file()
    print()
    
    print("2. Alternative method:")
    get_api_key_from_file_alt()
    print()
    
    print("3. From CWD method:")
    get_api_key_from_cwd()
    print()
    
    # Check if we can read the file
    print("4. Testing file reading:")
    try:
        with open("LiteLLM_Proxy_Key.txt", 'r') as f:
            key = f.read().strip()
            print(f"Direct read success: {key[:10]}...{key[-4:]}")
    except Exception as e:
        print(f"Direct read failed: {e}")
    
    try:
        with open("../LiteLLM_Proxy_Key.txt", 'r') as f:
            key = f.read().strip()
            print(f"Relative read success: {key[:10]}...{key[-4:]}")
    except Exception as e:
        print(f"Relative read failed: {e}")
    
    try:
        with open("../../LiteLLM_Proxy_Key.txt", 'r') as f:
            key = f.read().strip()
            print(f"Parent read success: {key[:10]}...{key[-4:]}")
    except Exception as e:
        print(f"Parent read failed: {e}")
