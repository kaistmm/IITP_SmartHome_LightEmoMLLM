import argparse
import os
import tempfile

import cv2
import soundfile as sf
import torch
from PIL import Image
from moviepy.editor import VideoFileClip
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from minigpt4.common.eval_utils import init_model, prepare_texts
from minigpt4.conversation.conversation import CONV_VISION_minigptv2


VALID_EMOTIONS = ["neutral", "angry", "happy", "sad", "worried", "surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description="Single-video emotion inference")
    parser.add_argument(
        "--cfg-path",
        default="eval_configs/eval_emotion.yaml",
        help="path to configuration file",
    )
    parser.add_argument("--name", type=str, default="video_infer")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--eval_opt", type=str, default="all")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--options", nargs="+", default=None)
    parser.add_argument("--video-path", default="example/sample_00000167.mp4", help="path to input video file")
    parser.add_argument(
        "--transcript",
        default="",
        help="optional spoken sentence in the video; improves prompt quality",
    )
    parser.add_argument(
        "--hubert-model-path",
        default="/mnt/bear3/users/jungji/ckpt/chinese-hubert-large",
        help="local path or HF model id for HuBERT",
    )
    return parser.parse_args()


def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read first frame: {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def extract_audio_samples(video_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    clip = None
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            raise RuntimeError("video has no audio track")
        clip.audio.write_audiofile(
            audio_path,
            fps=16000,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
            logger=None,
        )
        samples, sr = sf.read(audio_path)
        return samples, sr
    finally:
        if clip is not None:
            clip.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)


@torch.no_grad()
def build_video_features(video_path, hubert_model_path, device):
    samples, sr = extract_audio_samples(video_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_path)
    input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values
    input_values = input_values.to(device)

    hubert = HubertModel.from_pretrained(hubert_model_path).to(device)
    hubert.eval()
    hidden_states = hubert(input_values, output_hidden_states=True).hidden_states
    audio_feature = torch.stack(hidden_states)[[-1]].sum(dim=0)
    audio_feature = torch.mean(audio_feature, dim=1, keepdim=True)

    zeros = torch.zeros((1, 2, audio_feature.shape[-1]), device=device, dtype=audio_feature.dtype)
    return torch.cat((zeros, audio_feature), dim=1)


def build_instruction(transcript):
    transcript = transcript.strip()
    prefix = ""
    if transcript:
        prefix = f"The person in video says: {transcript}. "
    return (
        "<video><VideoHere></video> <feature><FeatureHere></feature> "
        f"{prefix}[emotion] Please determine which emotion label in the video represents: "
        "happy, sad, neutral, angry, worried, surprise."
    )


def parse_emotion(answer):
    answer_low = answer.lower().strip()
    tokens = [tok.strip(".,!?;:()[]{}\"'") for tok in answer_low.split()]
    for tok in reversed(tokens):
        if tok in VALID_EMOTIONS:
            return tok
    return None


def main():
    args = parse_args()
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"video not found: {args.video_path}")
    device="cuda"
    model, vis_processor = init_model(args)
    model.eval()
    model.to(device)
    image = extract_first_frame(args.video_path)
    image_tensor = vis_processor(image).unsqueeze(0).to(device)
    video_features = build_video_features(args.video_path, args.hubert_model_path, device)

    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    instruction = build_instruction(args.transcript)
    texts = prepare_texts([instruction], conv_temp)

    outputs = model.generate(
        images=image_tensor,
        video_features=video_features,
        texts=texts,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    raw_answer = outputs[0].strip()
    emotion = parse_emotion(raw_answer)

    print(f"video_path: {args.video_path}")
    print(f"prompt: {instruction}")
    print(f"raw_answer: {raw_answer}")
    print(f"predicted_emotion: {emotion if emotion is not None else 'unknown'}")


if __name__ == "__main__":
    main()
