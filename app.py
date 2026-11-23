import os
import subprocess
import modal


sd_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0", "libgoogle-perftools-dev")
    # Clone the webui repository directly into the container build
    .run_commands(
        "git clone --depth 1 --branch dev https://github.com/AUTOMATIC1111/stable-diffusion-webui.git"
    )
    .run_commands(
        "pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121",
        "pip install -r /stable-diffusion-webui/requirements_versions.txt",
        "pip install xformers==0.0.23.post1",
        "cd /stable-diffusion-webui && python launch.py --exit --skip-torch-cuda-test --xformers",
    )
    # Download your requested checkpoint model into the correct directory
    .run_commands(
        'wget -O "/stable-diffusion-webui/models/Stable-diffusion/One-obsession-v16.safetensors" '
        '"https://civitai.com/api/downxload/models/2044887?fileId=1941948&token=d49ffffe1dd758fa6f1d4244fc6db97d"'
    )
)

# Initialize the Modal App
app = modal.App("stable-diffusion-webui", image=sd_image)

# Expose a public URL endpoint (Gradio relies on standard web routing)
@app.function(
    gpu="T4",              # Request a GPU (A10G or T4 are cost-effective choices)
    cpu=4,
    memory=16384,
    timeout=3600,         
)
@modal.web_server(port=7860, startup_timeout=300)
def run_webui():
    os.chdir("/stable-diffusion-webui")
    cmd = [
        "python", "launch.py",
        "--listen",
        "--port", "7860",
        "--xformers",
        "--no-gradio-queue",
        "--no-half",
        "--enable-insecure-extension-access",
    ]
    
    subprocess.Popen(cmd)
