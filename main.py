import argparse
import collections
import dataclasses
import queue
import signal
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
import sys


@dataclasses.dataclass
class Settings:
    """Runtime configuration."""

    model_path: str = "models/whisper-small"  # dossier ou fichier modèle faster-whisper
    device: str = "cpu"  # "cpu" ou "cuda"
    compute_type: str = "int8"  # int8 pour CPU, float16 pour GPU
    language: str = "fr"
    sample_rate: int = 16000
    frame_ms: int = 30  # frames de 10/20/30 ms compatibles webrtcvad
    silence_ms: int = 900  # silence cumulé pour clore un segment
    min_voice_ms: int = 300  # durée minimale d'un segment valide
    vad_aggressiveness: int = 2  # 0..3 (3 = plus strict)
    save_wav: bool = False  # True pour debug
    captures_dir: Path = dataclasses.field(default_factory=lambda: Path("captures"))
    input_device: Optional[Union[int, str]] = None  # index ou nom du device audio d'entrée
    codex_pid: Optional[int] = None  # PID d'un process Codex cible (send.py)
    send_script: Optional[Path] = None  # chemin vers send.py


class Frame:
    """Audio frame alignée sur webrtcvad."""

    def __init__(self, data: bytes, timestamp: float, duration_ms: int) -> None:
        self.data = data
        self.timestamp = timestamp
        self.duration_ms = duration_ms


class VADSegmenter:
    """Segmenter basé sur webrtcvad ; émet un segment dès qu'un silence suffisant est détecté."""

    def __init__(
        self,
        vad: webrtcvad.Vad,
        frame_ms: int,
        sample_rate: int,
        silence_ms: int,
        min_voice_ms: int,
    ) -> None:
        self.vad = vad
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate
        self.num_padding_frames = max(1, silence_ms // frame_ms)
        self.min_frames = max(1, min_voice_ms // frame_ms)
        self.ring_buffer: collections.deque = collections.deque(maxlen=self.num_padding_frames)
        self.triggered = False
        self.voiced_frames: List[Frame] = []

    def process(self, frame: Frame) -> Optional[List[Frame]]:
        """Retourne un segment (liste de frames) quand fin d'énoncé détectée."""
        is_speech = self.vad.is_speech(frame.data, sample_rate=self.sample_rate)

        if not self.triggered:
            self.ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
            if num_voiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = True
                self.voiced_frames.extend(f for f, _ in self.ring_buffer)
                self.ring_buffer.clear()
        else:
            self.voiced_frames.append(frame)
            self.ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                segment = self.voiced_frames
                self._reset()
                if len(segment) >= self.min_frames:
                    return segment
        return None

    def _reset(self) -> None:
        self.triggered = False
        self.ring_buffer.clear()
        self.voiced_frames = []


def pcm_frames_from_queue(
    audio_queue: queue.Queue,
    frame_size: int,
    frame_ms: int,
) -> Iterable[Frame]:
    """Transforme les blocs bruts du callback audio en objets Frame."""
    while True:
        data = audio_queue.get()
        timestamp = time.time()
        if len(data) != frame_size:
            continue  # ignore frames corrompues
        yield Frame(data=data, timestamp=timestamp, duration_ms=frame_ms)


def transcribe_segment(model: WhisperModel, pcm_bytes: bytes, settings: Settings) -> str:
    """Passe le segment au modèle Whisper et retourne le texte."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio=audio,
        language=settings.language,
        vad_filter=True,
        beam_size=5,
    )
    texts = [seg.text.strip() for seg in segments]
    return " ".join(t for t in texts if t).strip()


def save_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def send_to_codex(text: str, settings: Settings) -> None:
    """Envoi vers Codex via send.py + PID. Si pas de PID, affiche sur stdout."""
    if settings.codex_pid:
        send_script = settings.send_script or (Path(__file__).parent / "send.py")
        cmd_list = [sys.executable, str(send_script), text, "--pid", str(settings.codex_pid)]
        try:
            result = subprocess.run(cmd_list, capture_output=True, text=True)
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                print(f"[codex send error] code={result.returncode} {stderr}")
        except Exception as exc:
            print(f"[codex send exception] {exc}")
        return

    print(f"[-> Codex] {text}")


class CodexSender:
    """Wrapper minimal qui envoie via send.py (PID)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send(self, text: str) -> None:
        send_to_codex(text, self.settings)

    def close(self) -> None:
        pass

def run(settings: Settings) -> None:
    blocksize = int(settings.sample_rate * settings.frame_ms / 1000)
    frame_size = blocksize * 2  # int16 mono -> 2 octets par échantillon

    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def handle_interrupt(signum, frame) -> None:  # type: ignore
        stop_event.set()
    signal.signal(signal.SIGINT, handle_interrupt)

    vad = webrtcvad.Vad(settings.vad_aggressiveness)
    segmenter = VADSegmenter(
        vad=vad,
        frame_ms=settings.frame_ms,
        sample_rate=settings.sample_rate,
        silence_ms=settings.silence_ms,
        min_voice_ms=settings.min_voice_ms,
    )
    model = WhisperModel(
        settings.model_path,
        device=settings.device,
        compute_type=settings.compute_type,
    )
    codex_sender = CodexSender(settings)

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio warning] {status}")
        audio_queue.put(bytes(indata))

    device = settings.input_device
    if device is None:
        device = select_input_device()

    with sd.RawInputStream(
        samplerate=settings.sample_rate,
        blocksize=blocksize,
        dtype="int16",
        channels=1,
        device=device,
        callback=audio_callback,
    ):
        print(f"Écoute en cours... device={device} | Ctrl+C pour arrêter.")
        for frame in pcm_frames_from_queue(audio_queue, frame_size, settings.frame_ms):
            if stop_event.is_set():
                break

            segment = segmenter.process(frame)
            if not segment:
                continue

            pcm_bytes = b"".join(f.data for f in segment)
            start_ts = datetime.fromtimestamp(segment[0].timestamp)
            duration_ms = len(segment) * settings.frame_ms

            text = transcribe_segment(model, pcm_bytes, settings) or "[vide]"
            print(f"[{start_ts:%H:%M:%S}] {duration_ms} ms | {text}")
            codex_sender.send(text)

            if settings.save_wav:
                fname = start_ts.strftime("%Y%m%d-%H%M%S") + ".wav"
                save_wav(settings.captures_dir / fname, pcm_bytes, settings.sample_rate)

    codex_sender.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcription locale avec VAD et envoi automatique vers Codex.")
    parser.add_argument("--model-path", default=Settings.model_path, help="Chemin du modèle faster-whisper local.")
    parser.add_argument("--device", default=Settings.device, choices=["cpu", "cuda"], help="Périphérique pour Whisper.")
    parser.add_argument("--compute-type", default=Settings.compute_type, help="int8 pour CPU, float16 pour GPU.")
    parser.add_argument("--language", default=Settings.language, help="Langue forcée pour Whisper (ex: fr).")
    parser.add_argument("--frame-ms", type=int, default=Settings.frame_ms, choices=[10, 20, 30], help="Durée frame ms (VAD).")
    parser.add_argument("--silence-ms", type=int, default=Settings.silence_ms, help="Silence pour clore un segment.")
    parser.add_argument("--min-voice-ms", type=int, default=Settings.min_voice_ms, help="Durée minimale d'un segment.")
    parser.add_argument("--vad-aggressiveness", type=int, default=Settings.vad_aggressiveness, choices=[0, 1, 2, 3], help="0 (tolérant) à 3 (strict).")
    parser.add_argument("--save-wav", action="store_true", help="Sauvegarder chaque segment dans captures/.")
    parser.add_argument("--captures-dir", default=str(Settings().captures_dir), help="Dossier de sauvegarde des WAV.")
    parser.add_argument("--codex-pid", type=int, default=None, help="PID d'un process Codex existant (utilise send.py --pid).")
    parser.add_argument(
        "--send-script",
        default=None,
        help="Chemin vers send.py (si différent du chemin par défaut).",
    )
    parser.add_argument("--input-device", default=None, help="Index ou nom du device d'entrée micro (sounddevice).")
    return parser.parse_args()


def select_input_device() -> Union[int, str]:
    """Choisit un device d'entrée : privilégie le défaut s'il est valide, sinon le premier avec entrée."""
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            return default_in
    except Exception:
        pass

    devices = sd.query_devices()
    for idx, info in enumerate(devices):
        if info.get("max_input_channels", 0) > 0:
            return idx
    raise RuntimeError("Aucun device audio d'entrée disponible.")


if __name__ == "__main__":
    args = parse_args()
    settings = Settings(
        model_path=args.model_path,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        frame_ms=args.frame_ms,
        silence_ms=args.silence_ms,
        min_voice_ms=args.min_voice_ms,
        vad_aggressiveness=args.vad_aggressiveness,
        save_wav=args.save_wav,
        captures_dir=Path(args.captures_dir),
        input_device=int(args.input_device) if (args.input_device and args.input_device.isdigit()) else args.input_device,
        codex_pid=args.codex_pid,
        send_script=Path(args.send_script) if args.send_script else None,
    )
    run(settings)
