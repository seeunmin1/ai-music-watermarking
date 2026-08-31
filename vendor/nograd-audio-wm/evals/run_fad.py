import os
import sys
import glob
from frechet_audio_distance import FrechetAudioDistance


# for dataset in ["audio_prompts", "librispeech"]:

for dataset in ["music"]:
    unwatermarked = f"/ihchomes/milis/wmar/wmar_audio/outputs/wm_generations_new/{dataset}/unwatermarked_2/audio_standard"

    # watermarked_dirs = [
    #     f"/ihchomes/milis/wmar/wmar_audio/outputs/wm_generations_new/{dataset}/unwatermarked_1/audio_standard"
    # ]

    watermarked_dirs = [
        "/ihchomes/milis/wmar/wmar_audio/outputs/wm_generations_new/music/final_clusters0123_h0_delta1/audio_standard",
        "/ihchomes/milis/wmar/wmar_audio/outputs/wm_generations_new/music/final_clusters0123_h0_delta1/audio_selected",
        "/ihchomes/milis/wmar/wmar_audio/outputs/wm_generations_new/music/meta_noaug_clusters1234_h0_delta1/audio_standard",
        "/ihchomes/milis/wmar/wmar_audio/outputs/wm_generations_new/music/unwatermarked_2/audio_standard"
    ]

    # pann has some unpickling error, encodec says "all the input array dimensions except for the concatenation axis must match exactly"
    model_configs = [
        ("vggish", 16000),
        ("clap", 48000)
    ]

    for config in model_configs:
        model_name, sr = config

        print(model_name)

        # Add this to __load_audio_files
        # if fname.split(".")[-1] not in ["wav", "flac"]: continue
        fad = FrechetAudioDistance(model_name=model_name, sample_rate=sr)

        for watermarked in watermarked_dirs:
            if not os.path.isdir(watermarked):
                continue

            if not (os.path.isdir(unwatermarked) and os.path.isdir(watermarked)):
                print(f"Skipping: {unwatermarked} or {watermarked} not found.")
                continue

            fad_score = fad.score(unwatermarked, watermarked)
            print(watermarked, "FAD: %.8f" % fad_score, flush=True)
            print(watermarked, "FAD: %.8f" % fad_score, file=sys.stderr, flush=True)

        print()
