"""
Audio aligner: extracts audio from any video (no size limit),
transcribes with faster-whisper/WhisperX, produces word-level
timestamps and narrative beats for b-roll cut points.

Uses ffmpeg (bundled via imageio-ffmpeg) for audio extraction.
Supports Tamil, English, and 90+ other languages (auto-detected).
"""
from pathlib import Path
from dataclasses import dataclass, field
import subprocess
import tempfile
import math
import re
import os
import torch


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    score: float = 1.0


@dataclass
class NarrativeBeat:
    index: int
    start_time: float
    end_time: float
    duration: float
    text: str
    words: list[WordTimestamp] = field(default_factory=list)
    search_keywords: list[str] = field(default_factory=list)

    @property
    def formatted_range(self) -> str:
        s_m, s_s = divmod(self.start_time, 60)
        e_m, e_s = divmod(self.end_time, 60)
        return f"{int(s_m):02d}:{s_s:05.2f} - {int(e_m):02d}:{e_s:05.2f}"


def _get_ffmpeg_path() -> str:
    """Get ffmpeg binary path — bundled via imageio-ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    # Try system ffmpeg
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path

    raise RuntimeError(
        "ffmpeg not found. Install it: pip install imageio-ffmpeg\n"
        "Or install ffmpeg system-wide: https://ffmpeg.org/download.html"
    )


def extract_audio(media_path: Path, output_path: Path = None,
                  sample_rate: int = 16000, mono: bool = True) -> Path:
    """
    Extract audio from any video/audio file using ffmpeg.
    No file size or duration limit.

    Args:
        media_path: Input video or audio file (mp4, mkv, avi, mov, webm, mp3, etc.)
        output_path: Where to save extracted WAV. Auto-generated if None.
        sample_rate: Output sample rate (16000 for Whisper).
        mono: Convert to mono (required for Whisper).

    Returns:
        Path to extracted WAV file.
    """
    media_path = Path(media_path).resolve()
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    # If already a WAV with correct params, skip extraction
    if media_path.suffix.lower() == ".wav" and media_path.stat().st_size < 500_000_000:
        # Small WAV files can go direct to whisper
        return media_path

    ffmpeg_bin = _get_ffmpeg_path()

    if output_path is None:
        # Put extracted audio in temp_cache alongside the project
        from config.settings import settings
        cache_dir = settings.TEMP_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / f"{media_path.stem}_audio.wav"

    output_path = Path(output_path)

    # Build ffmpeg command
    cmd = [
        ffmpeg_bin,
        "-i", str(media_path),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # 16-bit PCM WAV
        "-ar", str(sample_rate),  # Sample rate
    ]
    if mono:
        cmd.extend(["-ac", "1"])  # Mono
    cmd.extend([
        "-y",                     # Overwrite
        str(output_path),
    ])

    print(f"[AudioAligner] Extracting audio: {media_path.name} → {output_path.name}")
    print(f"[AudioAligner] File size: {media_path.stat().st_size / (1024*1024):.1f} MB")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for very large files
        )
        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {error_msg}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced empty output")

        print(f"[AudioAligner] Audio extracted: {output_path.stat().st_size / (1024*1024):.1f} MB")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timed out (>10 min). File may be corrupted.")


def get_media_duration(media_path: Path) -> float:
    """Get duration of any media file in seconds using ffmpeg."""
    try:
        ffmpeg_bin = _get_ffmpeg_path()
        cmd = [
            ffmpeg_bin, "-i", str(media_path),
            "-hide_banner", "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Parse duration from stderr
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
        if match:
            h, m, s, cs = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            return h * 3600 + m * 60 + s + cs / 100.0
    except Exception as e:
        print(f"[AudioAligner] Duration probe failed: {e}")
    return 0.0


class AudioBeatAligner:
    """
    Ingests video or audio files of any size/duration.
    Extracts audio via ffmpeg, then transcribes with WhisperX / faster-whisper.
    Produces word-level timestamps and narrative beats.

    Supported languages: auto-detected (Tamil, English, Hindi, etc.)
    """

    def __init__(self, model_size: str = "base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self._whisperx_available = False
        self._whisper_model = None

    def _init_models(self):
        if self._whisper_model is not None:
            return

        # Attempt whisperx first
        try:
            import whisperx
            compute_type = "float16" if self.device == "cuda" else "int8"
            print(f"[AudioAligner] Loading WhisperX ({self.model_size}) on {self.device} ({compute_type})...")
            self._whisper_model = whisperx.load_model(self.model_size, self.device, compute_type=compute_type)
            self._whisperx_available = True
        except Exception as e:
            print(f"[AudioAligner] WhisperX unavailable ({e}). Using faster-whisper...")
            try:
                from faster_whisper import WhisperModel
                compute_type = "float16" if self.device == "cuda" else "int8"
                self._whisper_model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
                self._whisperx_available = False
            except Exception as fe:
                print(f"[AudioAligner] faster-whisper fallback warning: {fe}")
                self._whisper_model = None

    def transcribe_and_align(self, media_path: Path | str) -> list[WordTimestamp]:
        """
        Transcribes any video/audio file. Steps:
        1. Extract audio via ffmpeg (handles any size/codec)
        2. Transcribe with whisper (auto language detection)
        3. Return word-level timestamps

        Supports: mp4, mkv, avi, mov, webm, mp3, wav, aac, flac, ogg, etc.
        Languages: auto-detected (Tamil, English, Hindi, Spanish, etc.)
        """
        path = Path(media_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        # Step 1: Extract audio from video (skip if already WAV)
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts"}
        audio_exts_need_convert = {".mp3", ".aac", ".flac", ".ogg", ".m4a", ".wma", ".opus"}

        if path.suffix.lower() in video_exts:
            print(f"[AudioAligner] Video detected ({path.suffix}). Extracting audio with ffmpeg...")
            audio_path = extract_audio(path)
        elif path.suffix.lower() in audio_exts_need_convert:
            print(f"[AudioAligner] Converting {path.suffix} to WAV...")
            audio_path = extract_audio(path)
        else:
            # Assume WAV or compatible
            audio_path = path

        # Step 2: Transcribe
        self._init_models()
        words: list[WordTimestamp] = []

        if self._whisperx_available:
            words = self._transcribe_whisperx(audio_path)

        if not words and self._whisper_model is not None and not self._whisperx_available:
            words = self._transcribe_faster_whisper(audio_path)

        if not words:
            print("[AudioAligner] Using synthetic fallback (no whisper model available).")
            words = self._synthetic_fallback()

        return words

    def _transcribe_whisperx(self, audio_path: Path) -> list[WordTimestamp]:
        """Transcribe with WhisperX (best quality, word-level alignment)."""
        import whisperx
        words = []
        try:
            audio = whisperx.load_audio(str(audio_path))
            result = self._whisper_model.transcribe(audio, batch_size=16)

            language = result.get("language", "en")
            print(f"[AudioAligner] Detected language: {language}")

            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=self.device
            )
            aligned = whisperx.align(
                result["segments"], align_model, metadata, audio,
                self.device, return_char_alignments=False
            )

            for seg in aligned.get("segments", []):
                for w in seg.get("words", []):
                    if "start" in w and "end" in w:
                        words.append(WordTimestamp(
                            word=w["word"].strip(),
                            start=float(w["start"]),
                            end=float(w["end"]),
                            score=float(w.get("score", 1.0)),
                        ))
        except Exception as e:
            print(f"[AudioAligner] WhisperX failed: {e}")
        return words

    def _transcribe_faster_whisper(self, audio_path: Path) -> list[WordTimestamp]:
        """Transcribe with faster-whisper (word timestamps)."""
        words = []
        try:
            segments, info = self._whisper_model.transcribe(
                str(audio_path), word_timestamps=True
            )
            print(f"[AudioAligner] Detected language: {info.language} (probability {info.language_probability:.2f})")

            for seg in segments:
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        words.append(WordTimestamp(
                            word=w.word.strip(),
                            start=float(w.start),
                            end=float(w.end),
                            score=float(getattr(w, "probability", 1.0)),
                        ))
                else:
                    # Segment-level fallback: distribute evenly
                    text_words = seg.text.strip().split()
                    if not text_words:
                        continue
                    dur_per_word = (seg.end - seg.start) / max(len(text_words), 1)
                    for i, tw in enumerate(text_words):
                        w_start = seg.start + i * dur_per_word
                        words.append(WordTimestamp(
                            word=tw, start=w_start,
                            end=w_start + dur_per_word, score=0.9,
                        ))
        except Exception as e:
            print(f"[AudioAligner] faster-whisper failed: {e}")
        return words

    def _synthetic_fallback(self) -> list[WordTimestamp]:
        """Demo fallback when no whisper model is available."""
        return [
            WordTimestamp(word="In", start=0.0, end=0.3),
            WordTimestamp(word="today's", start=0.3, end=0.7),
            WordTimestamp(word="fast-paced", start=0.7, end=1.4),
            WordTimestamp(word="world,", start=1.4, end=2.0),
            WordTimestamp(word="technology", start=2.2, end=2.9),
            WordTimestamp(word="empowers", start=2.9, end=3.5),
            WordTimestamp(word="creators", start=3.5, end=4.1),
            WordTimestamp(word="to", start=4.1, end=4.3),
            WordTimestamp(word="build", start=4.3, end=4.7),
            WordTimestamp(word="stunning", start=4.7, end=5.2),
            WordTimestamp(word="cinematic", start=5.2, end=5.8),
            WordTimestamp(word="stories.", start=5.8, end=6.5),
            WordTimestamp(word="From", start=7.0, end=7.3),
            WordTimestamp(word="dynamic", start=7.3, end=7.8),
            WordTimestamp(word="drone", start=7.8, end=8.2),
            WordTimestamp(word="aerials", start=8.2, end=8.8),
            WordTimestamp(word="over", start=8.8, end=9.1),
            WordTimestamp(word="misty", start=9.1, end=9.5),
            WordTimestamp(word="mountain", start=9.5, end=10.1),
            WordTimestamp(word="peaks,", start=10.1, end=10.8),
            WordTimestamp(word="to", start=11.0, end=11.2),
            WordTimestamp(word="sleek", start=11.2, end=11.6),
            WordTimestamp(word="neon-lit", start=11.6, end=12.2),
            WordTimestamp(word="urban", start=12.2, end=12.7),
            WordTimestamp(word="skylines.", start=12.7, end=13.5),
        ]

    def segment_into_narrative_beats(
        self,
        words: list[WordTimestamp],
        min_beat_duration: float = 2.5,
        max_beat_duration: float = 6.0,
        pause_threshold: float = 0.6,
    ) -> list[NarrativeBeat]:
        """
        Segments word-level timestamps into narrative beats for B-roll cut points.
        """
        if not words:
            return []

        beats: list[NarrativeBeat] = []
        current_words: list[WordTimestamp] = []
        beat_index = 1

        for i, w in enumerate(words):
            current_words.append(w)
            current_duration = current_words[-1].end - current_words[0].start

            is_last_word = (i == len(words) - 1)
            has_sentence_end = bool(re.search(r"[.?!;]$", w.word))

            pause_after = 0.0
            if not is_last_word:
                pause_after = max(0.0, words[i + 1].start - w.end)

            should_split = False
            if is_last_word:
                should_split = True
            elif current_duration >= max_beat_duration:
                should_split = True
            elif current_duration >= min_beat_duration and (has_sentence_end or pause_after >= pause_threshold):
                should_split = True

            if should_split and current_words:
                text = " ".join(cw.word for cw in current_words)
                start_t = current_words[0].start
                end_t = current_words[-1].end
                duration = round(end_t - start_t, 2)

                clean_tokens = [re.sub(r"[^\w\s-]", "", cw.word).strip().lower() for cw in current_words]
                stop_words = {
                    "the", "a", "an", "in", "to", "and", "is", "of", "for", "with",
                    "on", "at", "by", "from", "it", "this", "that", "are", "be",
                }
                keywords = [tok for tok in clean_tokens if len(tok) > 2 and tok not in stop_words]

                beats.append(NarrativeBeat(
                    index=beat_index,
                    start_time=round(start_t, 2),
                    end_time=round(end_t, 2),
                    duration=duration,
                    text=text,
                    words=list(current_words),
                    search_keywords=keywords,
                ))
                beat_index += 1
                current_words = []

        return beats
