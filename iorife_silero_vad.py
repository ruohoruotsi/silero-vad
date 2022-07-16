

import torch
import time
import os
import runez
import sys
import json

torch.set_num_threads(1)
SAMPLING_RATE = 16000
temp_output_dir = "/tmp/test_segments_silero"

start_time = time.time()
model, utils = torch.hub.load(repo_or_dir='/Users/iorife/github/silero-vad/',
                              model='silero_vad',
                              source='local',
                              force_reload=True,
                              onnx=False)

# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                               model='silero_vad',
#                               force_reload=True,
#                               onnx=False)

print("VAD model loading --- %s seconds ---" % (time.time() - start_time))
(get_speech_timestamps, _, read_audio, _, _) = utils

def vad_audio(path):

    start_time = time.time()
    wav = read_audio(path, sampling_rate=SAMPLING_RATE)
    # get number timestamps from full audio file
    vad_timestamps = get_speech_timestamps(wav, model)
    print("VAD predictions --- %s seconds ---" % (time.time() - start_time))
    print("num VAD segments {}".format(len(vad_timestamps)))
    # pprint(vad_timestamps)
    return vad_timestamps


def segment_original_audiofile(timestamps, single_channel_wav_path):

    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)

    # Generate WAVS for each utterance
    start_time = time.time()
    index = 0
    for segment in timestamps:
        start = segment['start']/ SAMPLING_RATE # need seconds
        end = segment['end'] / SAMPLING_RATE  # need seconds
        duration = end - start
        segment_file, ext = os.path.splitext(os.path.basename(single_channel_wav_path))

        # check for empty extension
        if not ext:
            ext = ".wav"
        segment_file = str(index) + "_" + segment_file + "_" + str(start) + "_" + str(end) + ext
        segment_fullpath = os.path.join(temp_output_dir, segment_file)
        run_result = runez.run(
            runez.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "panic",
            "-ss",
            start,
            "-t",
            duration,
            "-i",
            "%s" % single_channel_wav_path,
            segment_fullpath,
            fatal=False,
        )
        if run_result.failed:
            print("Error ffmpeg segment file failed")
        else:
            print("the segment_fullpath: " + segment_fullpath)

        index += 1
    print("VAD snipping (ffmpeg) took --- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: iorife_silero_vad [path-to-audio-file]\n")
        exit()

    weeds_path = sys.argv[1]
    timestamps = vad_audio(weeds_path)
    # segment_original_audiofile(timestamps, weeds_path)

    results_file_root = "/tmp/netflix_catalog.medium.en_sad_silero/"
    basename, ext = os.path.splitext(os.path.basename(weeds_path))
    with open(results_file_root + basename + ".silero.txt", 'w') as file:
        file.write(json.dumps(timestamps))