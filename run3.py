import modal, os, sys, shlex

stub = modal.Stub("stable-diffusion-webui")
volume = modal.NetworkFileSystem.from_name("stable-diffusion-webui", create_if_missing=True)

@stub.function(
    image=modal.Image.from_registry("nvidia/cuda:12.2.0-base-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt update -y && \
        apt install -y software-properties-common && \
        apt update -y && \
        add-apt-repository -y ppa:git-core/ppa && \
        apt update -y && \
        apt install -y git git-lfs && \
        git --version  && \
        apt install -y aria2 libgl1 libglib2.0-0 wget && \
        pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
        pip install -q xformers==0.0.20 triton==2.0.0 packaging==23.1"
    ),
    network_file_systems={"/content/stable-diffusion-webui": volume},
    gpu="T4",
    timeout=60000,
)
GIT_SHA = (
    "abd922bd0c43a504e47eca2ed354c3634bd00834"  # specify the commit to fetch
),
image = (
    image.apt_install("git")


async def run():
    # os.system(f"git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /content/stable-diffusion-webui")
    # os.system(f"git clone https://github.com/Cabel7/Webui /content/stable-diffusion-webui")
    
    os.system(f"git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui /content/stable-diffusion-webui")
    os.system(f"git clone https://github.com/camenduru/sd-civitai-browser /content/stable-diffusion-webui/extensions/sd-civitai-browser")
    os.system(f"git clone https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git /content/stable-diffusion-webui/extensions/tag-autocomplete") 
    os.system(f"git clone https://github.com/camenduru/stable-diffusion-webui-huggingface /content/stable-diffusion-webui/extensions/stable-diffusion-webui-huggingface") 
    # os.system(f"git clone https://github.com/Cabel7/Webui/tree/bump-Pillow-blendmodes-dependency/modules /content/stable-diffusion-webui/modules") 
    # os.system(f"git clone https://github.com/XavierXiao/Dreambooth-Stable-Diffusion /content/stable-diffusion-webui/extensions/Dreambooth-Stable-Diffusion") 
    # os.system(f"rm -rf /content/stable-diffusion-webui/modules")
    #os.system(f"wget https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/master/modules /content/stable-diffusion-webui/modules")
    os.system(f"git clone https://github.com/Cabel7/modules /content/stable-diffusion-webui/modules") 


    os.chdir(f"/content/stable-diffusion-webui")
    # os.system(f"rm -rf /content/stable-diffusion-webui/repositories")    
    os.system(f"git reset --hard")
    os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Jagerblue/Anime_Dex/blob/main/.ipynb_checkpoints/Untitled-checkpoint.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o Aniverse_thxEd14Pruned.safetensors")
    # os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Jagerblue/Sdxl/resolve/main/Anime_Pastel_Face.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o Anime_Pastel_Face.safetensors")
    # os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Jagerblue/Sdxl/resolve/main/Goofy_Anime_Pastel.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o Goofy_Anime_Pastel.safetensors")
    # os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Jagerblue/Sdxl/resolve/main/Hass_APF.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o Hass_APF.safetensors")
    # os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Jagerblue/Sdxl/resolve/main/Hass_Goofy.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o Hass_Goofy.safetensors")
    
    # os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/juggernaut-xl/resolve/main/juggernautXL_version2.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o juggernautXL_version2.safetensors")
    # os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sd_xl_refiner_1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o sd_xl_refiner_1.0_0.9vae.safetensors")
    # os.system(f"sed -i -e 's/\["sd_model_checkpoint"\]/\["sd_model_checkpoint","sd_vae","CLIP_stop_at_last_layers"\]/g' /content/stable-diffusion-webui/modules/shared_options.py") 
    os.environ['HF_HOME'] = '/content/stable-diffusion-webui/cache/huggingface'
    # os.system(f"python launch.py --cors-allow-origins=* --xformers --theme dark --gradio-debug --share --enable-insecure-extension-access")
    sys.path.append('/content/stable-diffusion-webui')
    sys.argv = shlex.split("--cors-allow-origins=* --xformers --theme dark --gradio-debug --share --no-half-vae --enable-insecure-extension-access")
    from modules import launch_utils
    launch_utils.startup_timer.record("initial startup")
    launch_utils.prepare_environment()
    launch_utils.start()

@stub.local_entrypoint()
def main():
    run.remote()
