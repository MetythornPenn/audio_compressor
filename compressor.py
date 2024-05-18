

import torch, os, torchaudio, argparse, warnings, yaml
from network import make_model

warnings.filterwarnings("ignore")

audio_path = "./audio/0.04_usd.mp3"
save_path = "./outputs"
config_path = "./checkpoint/config.yaml"
checkpoint_path = "./checkpoint/model.pth"


device = "cpu"


def read_yaml(pth):
    with open(pth, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_file_size_in_kb(file_path):
    try:
        file_size_bytes = os.path.getsize(file_path)
        file_size_kb = file_size_bytes / 1024
        return file_size_kb
    except OSError as e:
        print(f"Error: {e}")
        return None



x, sr = torchaudio.load(audio_path)
x = x.to(device)

model = make_model(read_yaml(config_path)['model'])
model.load_state_dict(
    torch.load(checkpoint_path, map_location="cpu")["model_state_dict"],
)
model = model.to(device)

codes, size = model.encode(x, num_streams=6)
recon_x = model.decode(codes, size)

fname = audio_path.split("/")[-1]

if not os.path.exists(save_path):
    os.makedirs(save_path)

output_path = f"{save_path}/{fname}"
torchaudio.save(output_path, recon_x, sr)



print(f"Compression outputs saved into {save_path}")
print(f"Size of original audio: {get_file_size_in_kb(audio_path)} ")
print(f"Size of compress audio: {get_file_size_in_kb(output_path)} ")
