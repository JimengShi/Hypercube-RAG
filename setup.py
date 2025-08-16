#!/usr/bin/env python3
"""
Hypercube-RAG Project Setup Script

This script sets up the environment and downloads the dataset from Hugging Face.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path


def run_command(cmd, description="", timeout=300):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        print(f"✅ {description} completed")
        return True
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def check_python_version():
    """Check if Python version is 3.10+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor} detected")
        return True
    else:
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        return False


def setup_environment():
    """Set up Python virtual environment"""
    print("\n=== Environment Setup ===")
    
    if not check_python_version():
        print("Please install Python 3.10 or later")
        return False
    
    # Create virtual environment
    env_name = "hypercube_env"
    if not run_command(f"python -m venv {env_name}", "Creating virtual environment"):
        return False
    
    # Get activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = f"{env_name}\\Scripts\\activate"
        python_cmd = f"{env_name}\\Scripts\\python"
        pip_cmd = f"{env_name}\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_cmd = f"source {env_name}/bin/activate"
        python_cmd = f"{env_name}/bin/python"
        pip_cmd = f"{env_name}/bin/pip"
    
    print(f"📝 Virtual environment created: {env_name}")
    print(f"📝 To activate manually: {activate_cmd}")
    
    return python_cmd, pip_cmd


def install_dependencies(pip_cmd):
    """Install required dependencies"""
    print("\n=== Installing Dependencies ===")
    
    # Essential packages for data download
    essential_packages = [
        "huggingface_hub>=0.32.0",
        "datasets>=3.0.0",
        "tqdm"
    ]
    
    for package in essential_packages:
        if not run_command(f"{pip_cmd} install {package}", f"Installing {package}"):
            return False
    
    # Install requirements.txt if exists
    if Path("requirements.txt").exists():
        if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements.txt", timeout=600):
            print("⚠️  Warning: Some packages from requirements.txt failed to install")
            print("    You can install them manually later if needed")
    
    return True


def download_dataset(python_cmd):
    """Download dataset from Hugging Face"""
    print("\n=== Downloading Dataset ===")
    
    # Create download script
    download_script = '''
import os
from datasets import load_dataset
from pathlib import Path
import json

print("📥 Downloading Hypercube-RAG dataset from Hugging Face...")

try:
    # Load dataset from Hugging Face
    dataset = load_dataset("Rtian/hypercube-rag", trust_remote_code=True)
    
    print("✅ Dataset downloaded successfully")
    
    # Create data directory structure
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (data_dir / "query").mkdir(exist_ok=True)
    (data_dir / "corpus").mkdir(exist_ok=True) 
    (data_dir / "hypercube").mkdir(exist_ok=True)
    
    print("📁 Created data directory structure")
    
    # Process and save files by iterating through the dataset
    if hasattr(dataset, 'keys'):
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"📊 Processing {split_name} split with {len(split_data)} files")
            
            for item in split_data:
                # Determine file path based on the path in the dataset
                if 'path' in item:
                    file_path = Path("data") / item['path']
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write content to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        if 'content' in item:
                            f.write(item['content'])
                    
                    print(f"✅ Saved {file_path}")
    
    # Alternative: try to access files directly if the above doesn't work
    else:
        print("📄 Attempting alternative download method...")
        # This would need to be customized based on the actual dataset structure
        pass
    
    print("🎉 Dataset setup completed!")
    print(f"📍 Data location: {data_dir.absolute()}")
    
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("💡 You can try downloading manually:")
    print("   from datasets import load_dataset")
    print('   dataset = load_dataset("Rtian/hypercube-rag")')
    exit(1)
'''
    
    # Write and execute download script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(download_script)
        temp_script = f.name
    
    try:
        success = run_command(f"{python_cmd} {temp_script}", "Downloading dataset from Hugging Face", timeout=600)
        os.unlink(temp_script)
        return success
    except Exception:
        os.unlink(temp_script)
        return False


def create_activation_helper():
    """Create a helper script for environment activation"""
    print("\n=== Creating Helper Scripts ===")
    
    env_name = "hypercube_env"
    
    if os.name == 'nt':  # Windows
        script_content = f'''@echo off
echo Activating Hypercube-RAG environment...
call {env_name}\\Scripts\\activate.bat
echo ✅ Environment activated! You can now run Python scripts.
echo 💡 To deactivate, run: deactivate
cmd /k
'''
        script_name = "activate_env.bat"
    else:  # Unix/Linux/MacOS
        script_content = f'''#!/bin/bash
echo "Activating Hypercube-RAG environment..."
source {env_name}/bin/activate
echo "✅ Environment activated! You can now run Python scripts."
echo "💡 To deactivate, run: deactivate"
exec "$SHELL"
'''
        script_name = "activate_env.sh"
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    if os.name != 'nt':
        os.chmod(script_name, 0o755)
    
    print(f"✅ Created {script_name}")
    return script_name


def main():
    """Main setup function"""
    print("🚀 Hypercube-RAG Project Setup")
    print("=" * 40)
    print("This script will:")
    print("1. Create a Python virtual environment")
    print("2. Install required dependencies")
    print("3. Download the dataset from Hugging Face")
    print("4. Set up the project structure")
    print()
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    # Setup environment
    result = setup_environment()
    if not result:
        print("❌ Environment setup failed")
        return
    
    python_cmd, pip_cmd = result
    
    # Install dependencies
    if not install_dependencies(pip_cmd):
        print("❌ Dependency installation failed")
        return
    
    # Download dataset
    if not download_dataset(python_cmd):
        print("❌ Dataset download failed")
        return
    
    # Create helper scripts
    script_name = create_activation_helper()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print(f"1. Activate environment: ./{script_name}")
    print("2. Verify setup: python -c \"import datasets; print('✅ Ready!')\"")
    print("3. Check data: ls data/")
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()