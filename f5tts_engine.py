import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN") or ("hf_" + "gfJmLawUbyDekFSjQfKAeabVRTOImgHgoD")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import re
import subprocess
import shutil
import urllib.request
from typing import Optional, Dict, Any, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
F5TTS_DIR = os.path.join(BASE_DIR, "F5-TTS")
SRC_DIR = os.path.join(F5TTS_DIR, "src")
if os.path.exists(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

CKPTS_DIR = os.path.join(F5TTS_DIR, "ckpts")
os.makedirs(CKPTS_DIR, exist_ok=True)

# 1. Automatic Device & Precision Autocast Detection
import torch
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16 if (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32

import text_corrector

# 2. Vietnamese Text Cleaning & Auto-Correction Function
def clean_vietnamese_text(text: str) -> str:
    """Preprocess Vietnamese text to fix word repetitions, typos, and format punctuation."""
    if not text:
        return ""
    cleaned, _ = text_corrector.auto_correct_text(text)
    return cleaned

# 3. Audio Preprocessing for Reference Sample
def preprocess_reference_audio(input_path: str) -> str:
    """Preprocess reference audio into clean 24,000Hz Mono WAV for F5-TTS mel-spectrogram pipeline."""
    try:
        abs_path = os.path.abspath(input_path)
        temp_dir = os.path.join(BASE_DIR, "temp_f5_audio")
        os.makedirs(temp_dir, exist_ok=True)
        clean_name = re.sub(r"[^\w\.-]", "_", os.path.basename(abs_path))
        out_wav = os.path.join(temp_dir, f"ref_24k_{clean_name}.wav")

        import soundfile as sf
        import librosa

        audio, sr = librosa.load(abs_path, sr=24000, mono=True)
        max_val = max(abs(audio.min()), abs(audio.max()))
        if max_val > 0:
            audio = audio / max_val * 0.95

        sf.write(out_wav, audio, 24000)
        print(f"[Ref Audio Preprocess] Successfully converted to 24kHz Mono WAV: {out_wav}")
        return out_wav
    except Exception as e:
        print(f"[Ref Audio Preprocess Warning]: {e}")
        return os.path.abspath(input_path)

# 4. Automatic Speech-to-Text Transcription via Faster-Whisper
_whisper_model = None

def transcribe_audio(audio_path: str, language: str = "vi") -> Dict[str, Any]:
    """Auto-transcribe reference audio file to text using faster-whisper AI model."""
    global _whisper_model
    abs_path = os.path.abspath(audio_path)
    if not os.path.exists(abs_path):
        return {"status": "error", "message": f"Không tìm thấy file: {audio_path}"}

    try:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        segments, info = _whisper_model.transcribe(abs_path, language=language, beam_size=5)
        text_result = " ".join([segment.text.strip() for segment in segments])
        cleaned_text = clean_vietnamese_text(text_result)

        return {
            "status": "success",
            "transcription": cleaned_text,
            "language": info.language,
            "duration": info.duration
        }
    except Exception as e:
        print(f"[Whisper ASR Error]: {e}")
        return {"status": "error", "message": str(e)}

# 5. Optimized Inference Settings
def get_inference_config() -> Dict[str, Any]:
    return {
        "nfe_step": 16 if device != "cpu" else 10,
        "sway_sampling_coef": -1.0,
        "cfg_strength": 2.0,
        "speed": 1.0,
        "target_rms": 0.1
    }

def check_status() -> Dict[str, Any]:
    """Check F5-TTS library installation status and GPU CUDA availability."""
    f5_installed = False
    try:
        import f5_tts
        f5_installed = True
    except ImportError:
        f5_installed = os.path.exists(F5TTS_DIR)

    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "CPU Mode"

    return {
        "installed": f5_installed,
        "has_gpu": has_gpu,
        "device": gpu_name,
        "dtype": str(dtype),
        "config": get_inference_config()
    }

def run_voice_cloning(
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    output_dir: str = "results_f5tts",
    speed_val: float = 1.0,
    nfe_step_val: Optional[int] = None,
    cfg_strength_val: float = 2.0,
    sway_coef_val: float = -1.0
) -> Dict[str, Any]:
    """
    Run F5-TTS Zero-Shot Voice Cloning with optimized Vietnamese pipeline & 24kHz preprocessing.
    """
    abs_ref_audio = os.path.abspath(ref_audio_path)
    if not os.path.exists(abs_ref_audio):
        return {"status": "error", "message": f"Không tìm thấy file mẫu âm thanh: {ref_audio_path}"}

    processed_ref_audio = preprocess_reference_audio(abs_ref_audio)

    cleaned_gen_text = clean_vietnamese_text(gen_text)
    if not cleaned_gen_text:
        return {"status": "error", "message": "Vui lòng nhập văn bản cần nhái giọng!"}

    cleaned_ref_text = clean_vietnamese_text(ref_text)
    if not cleaned_ref_text:
        print("[F5-TTS Engine] Reference text empty. Auto-transcribing reference audio...")
        trans_res = transcribe_audio(processed_ref_audio)
        if trans_res.get("status") == "success":
            cleaned_ref_text = trans_res.get("transcription", "")
            print(f"[F5-TTS Engine] Auto-transcribed reference text: {cleaned_ref_text}")

    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    out_file = os.path.join(abs_output_dir, "cloned_voice_output.wav")
    if os.path.exists(out_file):
        try:
            os.remove(out_file)
        except Exception:
            pass

    config = get_inference_config()
    nfe = nfe_step_val if nfe_step_val is not None else config["nfe_step"]
    sway = sway_coef_val if sway_coef_val is not None else config["sway_sampling_coef"]
    cfg = cfg_strength_val if cfg_strength_val is not None else config["cfg_strength"]
    spd = speed_val if speed_val is not None else config["speed"]

    # Build F5-TTS CLI command with 24kHz preprocessed audio, remove_silence, and target_rms
    cmd = [
        sys.executable, "-m", "f5_tts.infer.infer_cli",
        "--ref_audio", processed_ref_audio,
        "--ref_text", cleaned_ref_text,
        "--gen_text", cleaned_gen_text,
        "--output_dir", abs_output_dir,
        "--output_file", "cloned_voice_output.wav",
        "--nfe_step", str(nfe),
        "--sway_sampling_coef", str(sway),
        "--cfg_strength", str(cfg),
        "--speed", str(spd),
        "--target_rms", "0.1",
        "--remove_silence",
        "--device", device
    ]

    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    print(f"[F5-TTS Engine] Executing Voice Cloning ({device}, dtype={dtype}): ref_audio={processed_ref_audio}, gen_text_len={len(cleaned_gen_text)}")
    try:
        process = subprocess.run(
            cmd,
            cwd=F5TTS_DIR if os.path.exists(F5TTS_DIR) else BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600
        )

        if process.returncode != 0:
            print(f"[F5-TTS Execution Error]: {process.stderr}")
            if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
                return {"status": "error", "message": f"Lỗi chạy F5-TTS: {process.stderr[:300]}"}

        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            return {
                "status": "success",
                "message": "Nhái giọng bằng F5-TTS AI thành công!",
                "output_audio": out_file,
                "ref_text_used": cleaned_ref_text,
                "filename": "cloned_voice_output.wav"
            }
        else:
            return {"status": "error", "message": "Không tạo được file âm thanh đầu ra."}
    except Exception as e:
        print(f"[F5-TTS Engine Exception]: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("Checking F5-TTS Engine status:")
    print(check_status())
