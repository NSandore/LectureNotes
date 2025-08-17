# app.py
# macOS Python GUI app: record/import audio, transcribe with Whisper, summarize with Ollama.
# Outputs per-lecture: transcript.txt, transcript.srt, notes.md, meta.json

import os, sys, threading, time, json, textwrap, traceback
from pathlib import Path
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from tkinter import scrolledtext

# Audio recording
import sounddevice as sd
import soundfile as sf

# Transcription + media utils
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests

APP_DIR = Path.home() / "LectureApp"
OUT_DIR = APP_DIR / "Outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CFG_PATH = APP_DIR / "config.json"

DEFAULT_CFG = {
    "whisper_model": "large-v3",         # try 'medium' or 'small' for speed
    "ollama_host": "http://localhost:11434",
    "ollama_model": "llama3.1:8b",
    "summarize": True,
    "srt_bucket_seconds": 90,            # SRT caption block length
    "sample_rate": 16000,                # recording sample rate
    "channels": 1
}

def load_cfg():
    if CFG_PATH.exists():
        try:
            return {**DEFAULT_CFG, **json.loads(CFG_PATH.read_text())}
        except Exception:
            pass
    return DEFAULT_CFG.copy()

def save_cfg(cfg):
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(cfg, indent=2))

CFG = load_cfg()

# ----------------- Audio Recorder -----------------
class Recorder:
    def __init__(self, samplerate=16000, channels=1, dtype='int16'):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self._stream = None
        self._file = None
        self._recording = False
        self.out_path = None

    def start(self, out_wav_path: Path):
        if self._recording:
            return
        self.out_path = out_wav_path
        # SoundFile writes WAV; sounddevice callback feeds frames
        self._file = sf.SoundFile(str(out_wav_path), mode='w',
                                  samplerate=self.samplerate,
                                  channels=self.channels, subtype='PCM_16')
        def callback(indata, frames, time_info, status):
            if status:
                # Input over/underrun warnings may appear; keep writing frames.
                pass
            self._file.write(indata)

        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels,
                                      dtype=self.dtype, callback=callback)
        self._stream.start()
        self._recording = True

    def stop(self):
        if not self._recording:
            return None
        try:
            self._stream.stop(); self._stream.close()
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass
        self._recording = False
        return self.out_path

    @property
    def is_recording(self):
        return self._recording

# ----------------- Transcription / Summary -----------------
def to_wav_16k(src_path: Path, dst_path: Path, samplerate=16000):
    # Requires ffmpeg (brew install ffmpeg)
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(samplerate)
    audio.export(dst_path, format="wav")
    return dst_path

def transcribe_wav(wav_path: Path, model_name: str, bucket_sec: int, log):
    log(f"Loading Whisper model: {model_name}")
    model = WhisperModel(model_name, device="auto", compute_type="auto")

    log("Transcribing (pass 1: text)…")
    segments, info = model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
    text_lines = [seg.text.strip() for seg in segments]
    transcript_text = "\n".join(text_lines)

    log("Transcribing (pass 2: timestamps for SRT)…")
    segments, _ = model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
    srt_blocks, bucket_start, bucket_text = [], None, []

    def fmt_ts(s):
        h=int(s//3600); m=int((s%3600)//60); sec=s%60
        return f"{h:02d}:{m:02d}:{int(sec):02d},{int((sec-int(sec))*1000):03d}"

    for seg in segments:
        if bucket_start is None:
            bucket_start = seg.start
        bucket_text.append(seg.text.strip())
        if (seg.end - bucket_start) >= bucket_sec:
            srt_blocks.append((bucket_start, seg.end, " ".join(bucket_text)))
            bucket_start, bucket_text = None, []

    if bucket_text:
        end = bucket_start + bucket_sec if bucket_start is not None else 0
        srt_blocks.append((bucket_start or 0, end, " ".join(bucket_text)))

    srt_lines = []
    for i, (st, en, txt) in enumerate(srt_blocks, 1):
        srt_lines.append(str(i))
        srt_lines.append(f"{fmt_ts(st)} --> {fmt_ts(en)}")
        srt_lines.append(txt.strip())
        srt_lines.append("")
    srt_text = "\n".join(srt_lines)

    lang = getattr(info, "language", "unknown")
    log(f"Language: {lang}")
    return transcript_text, srt_text, lang

def summarize_with_ollama(transcript: str, custom_instructions: str, host: str, model: str, log, temperature=0.2):
    system = "You are a precise academic note-taker. Use clean Markdown. Be concise and faithful to the transcript."
    user = textwrap.dedent(f"""
    Follow these instructions when summarizing the transcript:

    {custom_instructions.strip()}

    If the instructions are missing any sections, include this baseline:
    # TL;DR
    - 8–12 bullets

    # Key Concepts
    - 10–20 terms with one-line definitions

    # Topic Notes
    - Bullets grouped by topic
      - sub-bullets for examples, formulas, caveats

    # Exam Practice
    - 8–10 likely exam questions with brief model answers

    Transcript:
    {transcript}
    """).strip()

    url = f"{host.rstrip('/')}/api/chat"
    log(f"Summarizing via Ollama ({model})…")
    r = requests.post(url, json={
        "model": model,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        "stream": False,
        "options": {"temperature": temperature}
    }, timeout=3600)
    r.raise_for_status()
    return r.json()["message"]["content"]

# ----------------- GUI -----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Lecture Notes (Python)")
        self.geometry("900x700")
        self.minsize(820, 620)

        self.recorder = Recorder(samplerate=CFG["sample_rate"], channels=CFG["channels"])
        self.current_audio: Path | None = None

        # ---- Controls ----
        tk.Label(self, text="Custom Instructions").pack(anchor="w", padx=12, pady=(12, 2))

        self.instructions = scrolledtext.ScrolledText(self, height=10, font=("Menlo", 12))
        self.instructions.insert(
            "1.0",
            "# TL;DR\n- 8–12 bullets\n\n# Key Concepts\n- 10–20 terms (one line defs)\n\n# Topic Notes\n- Bullets by topic with sub-bullets\n\n# Exam Practice\n- 8–10 Q&A"
        )
        self.instructions.pack(fill="both", expand=False, padx=12, pady=(0, 12))

        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=12)

        self.record_btn = tk.Button(btn_frame, text="Record Lecture", command=self.toggle_record, width=18)
        self.record_btn.pack(side="left")

        self.process_btn = tk.Button(btn_frame, text="Process", command=self.process_current, width=12)
        self.process_btn.pack(side="left", padx=8)

        self.import_btn = tk.Button(btn_frame, text="Import File…", command=self.import_file, width=12)
        self.import_btn.pack(side="left", padx=8)

        self.open_out_btn = tk.Button(btn_frame, text="Open Outputs", command=self.open_outputs, width=14)
        self.open_out_btn.pack(side="left", padx=8)

        # Settings
        set_frame = tk.LabelFrame(self, text="Settings")
        set_frame.pack(fill="x", padx=12, pady=12)

        tk.Label(set_frame, text="Whisper model").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.whisper_var = tk.StringVar(value=CFG["whisper_model"])
        tk.Entry(set_frame, textvariable=self.whisper_var, width=20).grid(row=0, column=1, sticky="w", padx=8)

        tk.Label(set_frame, text="Ollama host").grid(row=0, column=2, sticky="w", padx=8)
        self.ollama_host_var = tk.StringVar(value=CFG["ollama_host"])
        tk.Entry(set_frame, textvariable=self.ollama_host_var, width=26).grid(row=0, column=3, sticky="w", padx=8)

        tk.Label(set_frame, text="Ollama model").grid(row=1, column=0, sticky="w", padx=8)
        self.ollama_model_var = tk.StringVar(value=CFG["ollama_model"])
        tk.Entry(set_frame, textvariable=self.ollama_model_var, width=20).grid(row=1, column=1, sticky="w", padx=8)

        self.summarize_var = tk.BooleanVar(value=CFG["summarize"])
        tk.Checkbutton(set_frame, text="Summarize with Ollama", variable=self.summarize_var).grid(row=1, column=2, sticky="w", padx=8)

        tk.Label(set_frame, text="SRT bucket (s)").grid(row=1, column=3, sticky="w", padx=8)
        self.bucket_var = tk.IntVar(value=CFG["srt_bucket_seconds"])
        tk.Entry(set_frame, textvariable=self.bucket_var, width=6).grid(row=1, column=3, sticky="e", padx=8)

        for i in range(4):
            set_frame.grid_columnconfigure(i, weight=1)

        # Status
        self.status = scrolledtext.ScrolledText(self, height=12, font=("Menlo", 11))
        self.status.pack(fill="both", expand=True, padx=12, pady=(0,12))
        self.log("Ready.")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, msg):
        self.status.insert("end", f"{time.strftime('[%H:%M:%S]')} {msg}\n")
        self.status.see("end")
        self.update_idletasks()

    def toggle_record(self):
        if not self.recorder.is_recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_wav = OUT_DIR / f"Lecture_{ts}.wav"
            try:
                self.recorder.start(out_wav)
                self.current_audio = out_wav
                self.record_btn.configure(text="Stop Recording")
                self.log(f"Recording → {out_wav.name}")
            except Exception as e:
                self.log(f"ERROR starting recording: {e}")
                mb.showerror("Microphone", f"Failed to start recording:\n{e}")
        else:
            path = self.recorder.stop()
            self.record_btn.configure(text="Record Lecture")
            self.log(f"Stopped. File saved: {path.name if path else '(unknown)'}")
            # Uncomment to auto-process on stop:
            # self.process_current()

    def import_file(self):
        f = fd.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", ".m4a .mp3 .wav .aac .flac .ogg .wma"), ("All", "*.*")]
        )
        if not f: return
        src = Path(f)
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest = OUT_DIR / f"Imported_{ts}{src.suffix.lower()}"
        try:
            dest.write_bytes(src.read_bytes())
        except Exception as e:
            self.log(f"ERROR importing: {e}")
            return
        self.current_audio = dest
        self.log(f"Imported: {dest.name}")

    def process_current(self):
        if not self.current_audio or not self.current_audio.exists():
            mb.showwarning("No audio", "Record or import a file first.")
            return
        custom_instructions = self.instructions.get("1.0", "end").strip()
        whisper_model = self.whisper_var.get().strip()
        bucket = int(self.bucket_var.get())
        summarize = bool(self.summarize_var.get())
        ollama_host = self.ollama_host_var.get().strip()
        ollama_model = self.ollama_model_var.get().strip()

        # Persist config
        CFG.update({
            "whisper_model": whisper_model,
            "ollama_host": ollama_host,
            "ollama_model": ollama_model,
            "summarize": summarize,
            "srt_bucket_seconds": bucket
        })
        save_cfg(CFG)

        threading.Thread(
            target=self._process_worker,
            args=(self.current_audio, custom_instructions, whisper_model, bucket, summarize, ollama_host, ollama_model),
            daemon=True
        ).start()

    def _process_worker(self, audio_path, custom_instructions, whisper_model, bucket, summarize, ollama_host, ollama_model):
        try:
            self.log(f"Processing: {audio_path.name}")
            job_dir = OUT_DIR / audio_path.stem
            job_dir.mkdir(exist_ok=True)

            # Ensure WAV 16k mono input
            if audio_path.suffix.lower() != ".wav":
                wav_path = job_dir / (audio_path.stem + ".wav")
                self.log("Converting to 16k mono WAV…")
                to_wav_16k(audio_path, wav_path, samplerate=CFG["sample_rate"])
            else:
                wav_path = job_dir / audio_path.name
                if audio_path != wav_path:
                    audio_path.replace(wav_path)

            # Transcribe
            transcript, srt_text, lang = transcribe_wav(wav_path, whisper_model, bucket, self.log)
            (job_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
            (job_dir / "transcript.srt").write_text(srt_text, encoding="utf-8")
            self.log("Transcript + SRT written.")

            # Summarize
            if summarize:
                try:
                    notes = summarize_with_ollama(transcript, custom_instructions, ollama_host, ollama_model, self.log)
                    (job_dir / "notes.md").write_text(notes, encoding="utf-8")
                    self.log("Notes written: notes.md")
                except Exception as e:
                    (job_dir / "notes.md").write_text(f"Summarization skipped or failed: {e}", encoding="utf-8")
                    self.log(f"Summarization failed: {e}")

            meta = {
                "source": str(audio_path),
                "language": lang,
                "whisper_model": whisper_model,
                "ollama_model": ollama_model if summarize else None,
                "finished": int(time.time())
            }
            (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self.log(f"Done. Outputs in:\n{job_dir}")

        except Exception:
            self.log("ERROR:\n" + traceback.format_exc())

    def open_outputs(self):
        os.system(f'open "{OUT_DIR}"')

    def on_close(self):
        try:
            if self.recorder.is_recording:
                self.recorder.stop()
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    App().mainloop()