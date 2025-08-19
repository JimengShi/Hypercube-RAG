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
    """Set up conda environment"""
    print("\n=== Environment Setup ===")
    
    # Check if conda is available
    if not run_command("conda --version", "Checking conda installation"):
        print("❌ Conda not found. Please install Miniforge or Anaconda first.")
        print("💡 For ARM64 systems, install Miniforge:")
        print("   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh")
        print("   bash Miniforge3-Linux-aarch64.sh")
        return False
    
    env_name = "hypercube_env"
    
    # Check if environment already exists
    result = subprocess.run(
        "conda env list | grep hypercube_env", 
        shell=True, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0 and env_name in result.stdout:
        print(f"✅ Conda environment '{env_name}' already exists, skipping creation")
    else:
        # Create conda environment
        print(f"🔄 Creating conda environment: {env_name}")
        
        # Create environment with Python 3.10+
        if not run_command(f"conda create -n {env_name} python=3.10 -y", "Creating conda environment"):
            return False
        
        print(f"📝 Conda environment created: {env_name}")
    
    # Get conda activation command
    activate_cmd = f"conda activate {env_name}"
    python_cmd = f"conda run -n {env_name} python"
    pip_cmd = f"conda run -n {env_name} pip"
    
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
        if not run_command(f'{pip_cmd} install "{package}"', f"Installing {package}"):
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
    
    # Create download script using huggingface_hub to download files directly
    download_script = '''
import os
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

print("📥 Downloading Hypercube-RAG dataset from Hugging Face...")

try:
    # Download the entire repository
    repo_path = snapshot_download(
        repo_id="Rtian/hypercube-rag",
        repo_type="dataset",
        local_dir="data_temp",
        local_dir_use_symlinks=False
    )
    
    print("✅ Dataset downloaded successfully")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    if data_dir.exists():
        print("⚠️  Data directory already exists, backing up...")
        shutil.move("data", f"data_backup_{Path('data').stat().st_mtime}")
    
    # Move downloaded data to the correct location
    shutil.move("data_temp", "data")
    
    print("📁 Data structure created successfully")
    
    # Clean up unnecessary files
    print("🧹 Cleaning up unnecessary files...")
    unnecessary_files = [
        data_dir / ".gitattributes",
        data_dir / "README.md",
        data_dir / ".cache"
    ]
    
    for file_path in unnecessary_files:
        if file_path.exists():
            if file_path.is_dir():
                shutil.rmtree(file_path)
                print(f"  Removed directory: {file_path.name}")
            else:
                file_path.unlink()
                print(f"  Removed file: {file_path.name}")
    
    # List what was downloaded
    for subdir in ["corpus", "query", "hypercube"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("**/*.jsonl"))
            print(f"✅ {subdir}: {len(files)} files downloaded")
    
    print("🎉 Dataset setup completed!")
    print(f"📍 Data location: {data_dir.absolute()}")
    
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("💡 You can try downloading manually:")
    print("   from huggingface_hub import snapshot_download")
    print('   snapshot_download(repo_id="Rtian/hypercube-rag", repo_type="dataset", local_dir="data")')
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
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print(f"1. Activate environment: conda activate hypercube_env")
    print("2. Verify setup: python -c \"import datasets; print('✅ Ready!')\"")
    print("3. Check data: ls data/")
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()