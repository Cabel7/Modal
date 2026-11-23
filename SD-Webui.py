import os
import subprocess
import modal


sd_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0", "libgoogle-perftools-dev")
    # Clone the webui repository directly into the container build
    .run_commands(
        "git clone --branch dev https://github.com/AUTOMATIC1111/stable-diffusion-webui.git",
        "cd /stable-diffusion-webui && git checkout 1937682a20f7f0442311a1ede68f9f0cb480163b",
    )
    .run_commands(
        "pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121",
        "pip install -r /stable-diffusion-webui/requirements_versions.txt",
        "pip install xformers==0.0.23.post1",
        "cd /stable-diffusion-webui && python launch.py --exit --skip-torch-cuda-test --xformers",
    )
    # Download your requested checkpoint model into the correct directory
    .run_commands(
        'wget -O "/stable-diffusion-webui/models/Stable-diffusion/model.safetensors" '
        '"https://civitai.com/api/download/models/2044887?fileId=1941948&token=d49ffffe1dd758fa6f1d4244fc6db97d"'
    )
    .run_commands(
        "mkdir -p /stable-diffusion-webui/models/Lora",
        'wget -O "/stable-diffusion-webui/models/Lora/lauma-v2.safetensors" "https://huggingface.co/Jagerblue/Lora-Created/resolve/main/lauma-v2.safetensors"',
        'wget -O "/stable-diffusion-webui/models/Lora/nefer-v1.safetensors" "https://huggingface.co/Jagerblue/Lora-Created/resolve/main/Nefer-v1.safetensors"',
        'wget -O "/stable-diffusion-webui/models/Lora/illuxy-styleTA.safetensors" "https://huggingface.co/Jagerblue/FreeLora/resolve/main/illuxyStyleTA_trained.safetensors"',
    )
)

# Initialize the Modal App
app = modal.App("stable-diffusion-webui", image=sd_image)

@app.function(
    gpu="L4",              # Request a GPU (A10G or T4 are cost-effective choices)
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
        "--medvram",
        "--no-gradio-queue",
        "--enable-insecure-extension-access",
    ]
    
    subprocess.Popen(cmd)
