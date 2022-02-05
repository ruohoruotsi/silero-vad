

import torch
from pprint import pprint

torch.set_num_threads(1)
SAMPLING_RATE = 16000

model, utils = torch.hub.load(repo_or_dir='/Users/iorife/github/silero-vad/',
                              model='silero_number_detector',
                              source='local',
                              force_reload=True,
                              onnx=False)

(get_number_ts,
 save_audio,
 read_audio,
 collect_chunks,
 drop_chunks) = utils

path = "/data/Weeds_S4_Till_We_Meet_5077021396.center.16k.wav"
wav = read_audio(path, sampling_rate=SAMPLING_RATE)
# get number timestamps from full audio file
number_timestamps = get_number_ts(wav, model)
pprint(number_timestamps)