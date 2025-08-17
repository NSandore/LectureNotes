# app.py — Minimal UI + Settings dialog + "Other" default + Date-named outputs
# Features:
#  - Big ⏺︎ Record / ⏹︎ Stop
#  - Import Voice Memo
#  - Class dropdown (defaults to "Other" if present)
#  - Template dropdown + Template Manager (in Settings)
#  - Optional "Use Custom Instructions" textbox for this run only
#  - Notes Root picker (in Settings)
#  - Progress bar + single-line status
#  - Token-safe chunked summarization (Ollama) + Whisper transcription + SRT
#
# Outputs:
#   <notes_root>/<class>/<lecture_stem>/
#       <YYYY-MM-DD> - transcript.txt
#       <YYYY-MM-DD> - transcript.srt
#       <YYYY-MM-DD> - <Template or Custom> Notes.md
#       meta.json
#
# Deps (install):
#   brew install ffmpeg portaudio libsndfile
#   pip install sounddevice soundfile numpy faster-whisper requests

import os, time, json, textwrap, traceback, subprocess, shlex, threading, re, math
from pathlib import Path
import tkinter as tk
import tkinter.messagebox as mb
import tkinter.filedialog as fd
from tkinter import scrolledtext
from tkinter import ttk

import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import requests

# ---------- Paths / config ----------
APP_DIR = Path.home() / "LectureApp"
DEFAULT_NOTES_ROOT = APP_DIR / "Outputs"
DEFAULT_NOTES_ROOT.mkdir(parents=True, exist_ok=True)
CFG_PATH = APP_DIR / "config.json"

# Built-in templates (can be edited but not deleted)
BUILTIN_TEMPLATES = {
    "Lecture Notes": (
        "# TL;DR\n"
        "- 8–12 bullets (concise, factual)\n\n"
        "# Key Concepts\n"
        "- 10–20 terms with one-line definitions\n\n"
        "# Topic Notes\n"
        "- Bullets grouped by topic\n"
        "  - sub-bullets for formulas/examples/caveats\n\n"
        "# Exam Practice\n"
        "- 8–10 likely exam questions with brief model answers"
    ),
    "Test Prep": (
        "# Exam Crib Sheet\n"
        "- 10–15 must-know points\n\n"
        "# Definitions\n"
        "- key term: one-line definition\n\n"
        "# Formulas\n"
        "- formula → variables → when to use\n\n"
        "# Pitfalls\n"
        "- common misconception → correction\n\n"
        "# Practice Q&A\n"
        "- Q: …\n"
        "- A: … (1–3 lines)"
    ),
    "Flashcards": (
        "# Flashcards (Term → Back)\n"
        "- Term → concise 1–2 line answer\n"
        "- Concept → short explanation\n\n"
        "# Mnemonics\n"
        "- memorable phrase → what it stands for\n\n"
        "# Examples\n"
        "- brief example → takeaway"
    ),
    "Research Brief": (
        "# One-Page Summary\n"
        "- problem statement\n"
        "- approach/methods\n"
        "- key results (with numbers)\n"
        "- limitations\n"
        "- future directions\n\n"
        "# Citations/Names\n"
        "- important names/works mentioned\n\n"
        "# Questions Raised\n"
        "- open questions implied by the lecture"
    ),
}

DEFAULT_CFG = {
    "whisper_model": "medium",          # use "large-v3" for max accuracy if desired
    "ollama_host": "http://localhost:11434",
    "ollama_model": "llama3.1:8b",
    "sample_rate": 16000,
    "channels": 1,
    # Token-safe chunking (llama3.1:8b ~8k ctx; we keep well under)
    "chunk_max_tokens": 3500,
    "reducer_max_tokens": 3500,
    "srt_bucket_seconds": 90,
    "templates": BUILTIN_TEMPLATES,
    "selected_template": "Lecture Notes",
    # Folder selection
    "notes_root": str(DEFAULT_NOTES_ROOT),
    "selected_class_folder": "Other"  # default preference; will use if folder exists
}

def load_cfg():
    if CFG_PATH.exists():
        try:
            data = json.loads(CFG_PATH.read_text())
            # Merge built-ins if missing
            templates = data.get("templates", {})
            for k, v in BUILTIN_TEMPLATES.items():
                templates.setdefault(k, v)
            data["templates"] = templates
            # Ensure selected template exists
            if data.get("selected_template") not in templates:
                data["selected_template"] = "Lecture Notes"
            # Ensure notes_root exists
            nr = Path(data.get("notes_root", DEFAULT_NOTES_ROOT))
            nr.mkdir(parents=True, exist_ok=True)
            data["notes_root"] = str(nr)
            return {**DEFAULT_CFG, **data}
        except Exception:
            pass
    return DEFAULT_CFG.copy()

def save_cfg(cfg):
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(cfg, indent=2))

CFG = load_cfg()

# ---------- Simple audio recorder ----------
class Recorder:
    def __init__(self, samplerate=16000, channels=1, dtype='int16'):
        self.samplerate = samplerate; self.channels = channels; self.dtype = dtype
        self._stream = None; self._file = None; self.out_path = None; self._recording = False

    def start(self, out_wav_path: Path):
        self.out_path = out_wav_path
        self._file = sf.SoundFile(str(out_wav_path), mode='w', samplerate=self.samplerate,
                                  channels=self.channels, subtype='PCM_16')
        def cb(indata, frames, ti, status):
            self._file.write(indata)
        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels,
                                      dtype=self.dtype, callback=cb)
        self._stream.start(); self._recording = True

    def stop(self):
        if not self._recording: return None
        try:
            self._stream.stop(); self._stream.close()
        except Exception: pass
        try:
            self._file.close()
        except Exception: pass
        self._recording = False
        return self.out_path

    @property
    def is_recording(self): return self._recording

# ---------- Helpers ----------
def ffmpeg_to_wav_16k(src: Path, dst: Path, samplerate=16000):
    cmd = f'ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(str(src))} -ac 1 -ar {int(samplerate)} -sample_fmt s16 {shlex.quote(str(dst))}'
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e}")
    return dst

def transcribe_and_srt(wav: Path, model_name: str, bucket_sec: int):
    model = WhisperModel(model_name, device="auto", compute_type="auto")

    # Transcript
    segs, info = model.transcribe(str(wav), beam_size=5, vad_filter=True)
    transcript = "\n".join(s.text.strip() for s in segs)
    lang = getattr(info, "language", "unknown")

    # SRT (bucketed)
    segs, _ = model.transcribe(str(wav), beam_size=5, vad_filter=True)
    def ts(s):
        h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
        return f"{h:02d}:{m:02d}:{int(sec):02d},{int((sec - int(sec)) * 1000):03d}"

    blocks, start, buf = [], None, []
    last_end = 0.0
    for s in segs:
        if start is None:
            start = s.start
        buf.append(s.text.strip())
        last_end = s.end
        if (s.end - start) >= bucket_sec:
            blocks.append((start, s.end, " ".join(buf)))
            start, buf = None, []

    if buf:
        end_time = last_end if last_end else ((start or 0) + bucket_sec)
        blocks.append((start or 0, end_time, " ".join(buf)))

    srt_lines = []
    for i, (st, en, txt) in enumerate(blocks, 1):
        srt_lines += [str(i), f"{ts(st)} --> {ts(en)}", txt.strip(), ""]

    return transcript, "\n".join(srt_lines), lang

# ---------- Chunking (token safety) ----------
def approx_token_count(s: str) -> int:
    return math.ceil(len(s) / 4)  # ~4 chars per token

def split_into_chunks(text: str, max_tokens: int) -> list[str]:
    max_chars = max_tokens * 4
    paras = re.split(r'\n\s*\n', text.strip())
    chunks, cur = [], []
    def cur_len(): return sum(len(p) + 2 for p in cur)
    for p in paras:
        if cur and (cur_len() + len(p)) > max_chars:
            chunks.append("\n\n".join(cur)); cur = [p]
        else:
            cur.append(p)
    if cur: chunks.append("\n\n".join(cur))
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c); continue
        for i in range(0, len(c), max_chars):
            final.append(c[i:i+max_chars])
    return final

def ollama_chat(host: str, model: str, system: str, user: str, temperature=0.2, timeout=3600):
    url = f"{host.rstrip('/')}/api/chat"
    r = requests.post(url, json={
        "model": model,
        "messages": [
            {"role":"system", "content": system},
            {"role":"user",   "content": user}
        ],
        "stream": False,
        "options": {"temperature": temperature}
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]

def summarize_chunked(transcript: str, template_text: str, progress=None) -> str:
    """
    Map-Reduce summarization to guarantee we never exceed model context.
    """
    host = CFG["ollama_host"]; model = CFG["ollama_model"]
    max_toks = CFG["chunk_max_tokens"]; reduce_toks = CFG["reducer_max_tokens"]

    try:
        requests.get(host, timeout=2)
    except Exception:
        return "Summarization skipped: Ollama local server is not reachable."

    chunks = split_into_chunks(transcript, max_toks)

    system = "You are a precise academic note-taker. Use clean Markdown. Be concise and faithful to the transcript."
    map_prompt_tpl = textwrap.dedent("""
        Follow these instructions when summarizing ONLY this chunk:

        {template}

        Only include content present in this chunk.

        Chunk:
        {chunk}
    """).strip()

    if progress:
        progress("determinate_start", maximum=max(1, len(chunks) + 1))  # +1 for reduce

    map_summaries = []
    for i, c in enumerate(chunks, 1):
        try:
            user = map_prompt_tpl.format(template=template_text, chunk=c)
            summary = ollama_chat(host, model, system, user)
        except Exception as e:
            summary = f"(Chunk {i} summary failed: {e})"
        map_summaries.append(summary)
        if progress:
            progress("determinate_step")

    reduce_input = "\n\n---\n\n".join(map_summaries)
    if approx_token_count(reduce_input) > reduce_toks:
        per = int((reduce_toks * 4) / max(1, len(map_summaries)))
        reduce_input = "\n\n---\n\n".join(s[:per] for s in map_summaries)

    reduce_prompt = textwrap.dedent("""
        Combine the following chunk-level notes into ONE coherent study guide,
        strictly following these instructions:

        {template}

        Merge duplicates, keep language tight, and avoid speculation.

        Chunk notes:
        {notes}
    """).strip()

    try:
        final_notes = ollama_chat(host, model, system, reduce_prompt.format(
            template=template_text, notes=reduce_input))
    except Exception as e:
        final_notes = f"Summarization reduce step failed: {e}"

    if progress:
        progress("determinate_step")
    return final_notes

# ---------- UI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Lecture Notes")
        self.geometry("820x520")
        self.minsize(800, 500)

        self.rec = Recorder(samplerate=CFG["sample_rate"], channels=CFG["channels"])
        self.current_audio: Path | None = None

        # Main header row
        top = tk.Frame(self); top.pack(pady=10, fill="x")
        self.rec_btn = tk.Button(top, text="⏺︎  Record", font=("Helvetica", 16, "bold"),
                                 width=12, command=self.toggle_record)
        self.rec_btn.pack(side="left", padx=8)
        self.import_btn = tk.Button(top, text="Import Voice Memo…", command=self.import_file)
        self.import_btn.pack(side="left", padx=6)
        self.open_btn = tk.Button(top, text="Open Notes Folder", command=self.open_notes_root)
        self.open_btn.pack(side="left", padx=6)
        self.settings_btn = tk.Button(top, text="Settings…", command=self.open_settings)
        self.settings_btn.pack(side="right", padx=8)

        # Class + Template row
        row = tk.Frame(self); row.pack(padx=12, pady=4, fill="x")
        tk.Label(row, text="Class:").pack(side="left")
        self.class_var = tk.StringVar(value=CFG.get("selected_class_folder","Other"))
        self.class_combo = ttk.Combobox(row, textvariable=self.class_var, width=24, state="readonly")
        self._populate_class_combo()
        self.class_combo.pack(side="left", padx=(6,14))

        tk.Label(row, text="Template:").pack(side="left")
        self.template_var = tk.StringVar(value=CFG.get("selected_template","Lecture Notes"))
        self.template_combo = ttk.Combobox(row, textvariable=self.template_var, state="readonly", width=26)
        self._refresh_template_combo()
        self.template_combo.pack(side="left", padx=6)

        # Optional custom instructions
        ci = tk.Frame(self); ci.pack(fill="x", padx=12, pady=(8,2))
        self.use_custom_var = tk.BooleanVar(value=False)
        self.use_custom_chk = tk.Checkbutton(ci, text="Use Custom Instructions (for this run)", variable=self.use_custom_var, command=self._toggle_custom_box)
        self.use_custom_chk.pack(anchor="w")
        self.custom_box = scrolledtext.ScrolledText(self, height=7, font=("Menlo", 12))
        self.custom_box.insert("1.0", BUILTIN_TEMPLATES["Lecture Notes"])
        self.custom_box.pack_forget()

        # Progress + status
        bar = tk.Frame(self); bar.pack(fill="x", padx=12, pady=(8,2))
        self.prog = ttk.Progressbar(bar, mode="indeterminate")
        self.prog.pack(side="left", fill="x", expand=True)
        self.prog_running = False
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=12, pady=(2,12))

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI helpers ----------
    def _refresh_template_combo(self):
        names = sorted(CFG["templates"].keys())
        self.template_combo["values"] = names
        if self.template_var.get() not in names:
            self.template_var.set("Lecture Notes")

    def _populate_class_combo(self):
        root = Path(CFG["notes_root"])
        subs = [""]  # empty = root (optional)
        if root.exists():
            for d in sorted([p.name for p in root.iterdir() if p.is_dir()]):
                subs.append(d)
        self.class_combo["values"] = subs

        # Prefer "Other" if present; else use stored pref; fall back to root
        if "Other" in subs:
            self.class_var.set("Other")
            CFG["selected_class_folder"] = "Other"
            save_cfg(CFG)
        else:
            pref = CFG.get("selected_class_folder", "")
            if pref in subs:
                self.class_var.set(pref)
            elif self.class_var.get() not in subs:
                self.class_var.set("")

    def _toggle_custom_box(self):
        if self.use_custom_var.get():
            self.custom_box.pack(fill="x", padx=12, pady=(0,8))
        else:
            self.custom_box.pack_forget()

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    # Progress control
    def _progress(self, action, maximum=None):
        if action == "indeterminate_start":
            self.prog.configure(mode="indeterminate")
            self.prog.start(12); self.prog_running = True
        elif action == "indeterminate_stop":
            if self.prog_running:
                self.prog.stop(); self.prog_running = False
        elif action == "determinate_start":
            if self.prog_running:
                self.prog.stop(); self.prog_running = False
            self.prog.configure(mode="determinate", maximum=maximum or 100, value=0)
        elif action == "determinate_step":
            try: self.prog.step(1)
            except Exception: pass
        elif action == "determinate_stop":
            self.prog.configure(value=0)

    # ---------- Main actions ----------
    def open_notes_root(self):
        os.system(f'open "{CFG["notes_root"]}"')

    def toggle_record(self):
        if not self.rec.is_recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            base = Path(CFG["notes_root"]); base.mkdir(parents=True, exist_ok=True)
            out_wav = base / f"Lecture_{ts}.wav"
            try:
                self.rec.start(out_wav)
                self.current_audio = out_wav
                self.rec_btn.configure(text="⏹︎  Stop")
                self._set_status(f"Recording → {out_wav.name}")
            except Exception as e:
                self._set_status("Failed to start recording.")
                mb.showerror("Microphone", f"Failed to start recording:\n{e}")
        else:
            path = self.rec.stop()
            self.rec_btn.configure(text="⏺︎  Record")
            self._set_status(f"Saved: {path.name if path else '(unknown)'}")
            if path: self._kickoff_processing(path)

    def import_file(self):
        f = fd.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", ".m4a .mp3 .wav .aac .flac .ogg .wma"), ("All", "*.*")]
        )
        if not f: return
        src = Path(f)
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(CFG["notes_root"]); base.mkdir(parents=True, exist_ok=True)
        dest = base / f"Imported_{ts}{src.suffix.lower()}"
        try:
            dest.write_bytes(src.read_bytes())
            self.current_audio = dest
            self._set_status(f"Imported: {dest.name}")
            self._kickoff_processing(dest)
        except Exception as e:
            self._set_status("Import failed."); mb.showerror("Import", str(e))

    def _kickoff_processing(self, audio_path: Path):
        CFG["selected_template"] = self.template_var.get()
        CFG["selected_class_folder"] = self.class_var.get()
        save_cfg(CFG)
        threading.Thread(target=self._process_worker, args=(audio_path,), daemon=True).start()

    def _process_worker(self, audio_path: Path):
        try:
            self._progress("indeterminate_start")
            self._set_status(f"Processing {audio_path.name}…")

            root = Path(CFG["notes_root"])
            class_sub = CFG.get("selected_class_folder","").strip()
            target_root = root / class_sub if class_sub else root
            target_root.mkdir(parents=True, exist_ok=True)

            job_dir = target_root / audio_path.stem
            job_dir.mkdir(parents=True, exist_ok=True)

            # Convert if needed
            if audio_path.suffix.lower() != ".wav":
                wav = job_dir / (audio_path.stem + ".wav")
                self._set_status("Converting to 16k mono WAV…")
                ffmpeg_to_wav_16k(audio_path, wav, samplerate=CFG["sample_rate"])
            else:
                wav = job_dir / audio_path.name
                if audio_path != wav:
                    audio_path.replace(wav)

            # Transcribe
            self._set_status("Transcribing…")
            transcript, srt_text, lang = transcribe_and_srt(wav, CFG["whisper_model"], CFG["srt_bucket_seconds"])

            # Date-based filenames
            date_str = time.strftime("%Y-%m-%d")
            # If you don’t want dated transcript/SRT names, replace with fixed names.
            (job_dir / f"{date_str} - transcript.txt").write_text(transcript, encoding="utf-8")
            (job_dir / f"{date_str} - transcript.srt").write_text(srt_text, encoding="utf-8")

            # Instructions: custom vs template
            if self.use_custom_var.get():
                template_text = self.custom_box.get("1.0","end").strip()
                label = "Custom"
            else:
                tname = CFG.get("selected_template","Lecture Notes")
                template_text = CFG["templates"].get(tname, BUILTIN_TEMPLATES["Lecture Notes"])
                label = tname

            # Summarize with progress
            self._progress("indeterminate_stop")
            self._set_status(f"Summarizing ({label})…")
            def progress_cb(action, maximum=None): self._progress(action, maximum=maximum)
            notes = summarize_chunked(transcript, template_text, progress=progress_cb)
            self._progress("determinate_stop")

            # Safe filename label
            safe_label = "".join(c for c in label if c.isalnum() or c in (" ", "-", "_")).strip()
            notes_name = f"{date_str} - {safe_label} Notes.md"
            (job_dir / notes_name).write_text(notes, encoding="utf-8")

            meta = {
                "source": str(audio_path),
                "language": lang,
                "whisper_model": CFG["whisper_model"],
                "ollama_model": CFG["ollama_model"],
                "instructions_mode": "custom" if self.use_custom_var.get() else "template",
                "template": CFG.get("selected_template") if not self.use_custom_var.get() else None,
                "notes_root": str(root),
                "class_folder": class_sub,
                "finished": int(time.time())
            }
            (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            self._set_status(f"Done → {job_dir.name} / {notes_name}")
        except Exception:
            self._progress("indeterminate_stop"); self._progress("determinate_stop")
            self._set_status("Error. See alert.")
            mb.showerror("Error", traceback.format_exc())

    # ---------- Settings dialog ----------
    def open_settings(self):
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("720x520")
        win.transient(self); win.grab_set()

        # Notes Root
        nr_frame = tk.LabelFrame(win, text="Notes Location")
        nr_frame.pack(fill="x", padx=12, pady=10)
        tk.Label(nr_frame, text="Notes root:").pack(side="left", padx=(10,6), pady=8)
        path_lbl = tk.Label(nr_frame, text=self._short_path(CFG["notes_root"]), anchor="w")
        path_lbl.pack(side="left", fill="x", expand=True, padx=(0,10))
        def choose_root():
            new_root = fd.askdirectory(title="Choose notes root", initialdir=CFG["notes_root"])
            if not new_root: return
            CFG["notes_root"] = new_root; save_cfg(CFG)
            path_lbl.config(text=self._short_path(new_root))
            self._populate_class_combo()
        tk.Button(nr_frame, text="Choose…", command=choose_root).pack(side="left", padx=(0,10))
        tk.Button(nr_frame, text="Refresh Classes", command=self._populate_class_combo).pack(side="left")

        # Templates
        tpl_frame = tk.LabelFrame(win, text="Templates")
        tpl_frame.pack(fill="both", expand=True, padx=12, pady=(4,12))
        tk.Label(tpl_frame, text="Manage templates used for summaries.").pack(anchor="w", padx=10, pady=(8,4))
        tk.Button(tpl_frame, text="Open Template Manager", command=lambda: self._open_manager_in(win)).pack(anchor="w", padx=10, pady=6)

        tk.Button(win, text="Close", command=win.destroy).pack(pady=8)

    def _open_manager_in(self, parent):
        mgr = tk.Toplevel(parent)
        mgr.title("Manage Templates"); mgr.geometry("900x600"); mgr.transient(parent); mgr.grab_set()

        left = tk.Frame(mgr); left.pack(side="left", fill="y", padx=8, pady=8)
        right = tk.Frame(mgr); right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        tk.Label(left, text="Templates").pack(anchor="w")
        listbox = tk.Listbox(left, width=28, height=24); listbox.pack(fill="y", expand=False, pady=(4,8))
        names = sorted(CFG["templates"].keys())
        for n in names: listbox.insert("end", n)

        btns = tk.Frame(left); btns.pack(fill="x")
        def add_template():
            name = simple_prompt(mgr, "New template name:")
            if not name: return
            if name in CFG["templates"]:
                mb.showerror("Exists", "A template with that name already exists."); return
            CFG["templates"][name] = BUILTIN_TEMPLATES["Lecture Notes"]
            save_cfg(CFG); listbox.insert("end", name); self._refresh_template_combo()
        def rename_template():
            sel = listbox.curselection()
            if not sel: return
            old = listbox.get(sel[0])
            new = simple_prompt(mgr, f"Rename '{old}' to:")
            if not new or new == old: return
            if new in CFG["templates"]:
                mb.showerror("Exists", "A template with that name already exists."); return
            CFG["templates"][new] = CFG["templates"].pop(old)
            if CFG.get("selected_template")==old:
                CFG["selected_template"]=new; self.template_var.set(new)
            save_cfg(CFG); listbox.delete(sel[0]); listbox.insert(sel[0], new); self._refresh_template_combo()
        def delete_template():
            sel = listbox.curselection()
            if not sel: return
            name = listbox.get(sel[0])
            if name in BUILTIN_TEMPLATES:
                mb.showwarning("Protected", "Built-in templates cannot be deleted (you can edit them)."); return
            if not mb.askyesno("Delete", f"Delete template '{name}'?"): return
            CFG["templates"].pop(name, None)
            if CFG.get("selected_template")==name:
                CFG["selected_template"]="Lecture Notes"; self.template_var.set("Lecture Notes")
            save_cfg(CFG); listbox.delete(sel[0]); self._refresh_template_combo()

        tk.Button(btns, text="Add", width=8, command=add_template).pack(side="left", padx=2)
        tk.Button(btns, text="Rename", width=8, command=rename_template).pack(side="left", padx=2)
        tk.Button(btns, text="Delete", width=8, command=delete_template).pack(side="left", padx=2)

        tk.Label(right, text="Template Text (Markdown)").pack(anchor="w")
        text = scrolledtext.ScrolledText(right, font=("Menlo", 12))
        text.pack(fill="both", expand=True, pady=(4,8))

        def load_selected(*_):
            sel = listbox.curselection()
            if not sel:
                text.delete("1.0","end"); return
            name = listbox.get(sel[0])
            text.delete("1.0","end")
            text.insert("1.0", CFG["templates"].get(name,""))
        listbox.bind("<<ListboxSelect>>", load_selected)
        if names:
            listbox.selection_set(0); load_selected()

        actions = tk.Frame(right); actions.pack(fill="x")
        def save_changes():
            sel = listbox.curselection()
            if not sel: return
            name = listbox.get(sel[0])
            CFG["templates"][name] = text.get("1.0","end").strip()
            save_cfg(CFG); self._refresh_template_combo()
            mb.showinfo("Saved", f"Template '{name}' saved.")
        def use_selected():
            sel = listbox.curselection()
            if not sel: return
            name = listbox.get(sel[0])
            self.template_var.set(name); CFG["selected_template"]=name; save_cfg(CFG)
            mb.showinfo("Selected", f"Using template '{name}'.")
        tk.Button(actions, text="Save Changes", width=14, command=save_changes).pack(side="right")
        tk.Button(actions, text="Use This Template", width=18, command=use_selected).pack(side="right", padx=8)

    def _short_path(self, p: str, maxlen=60):
        return p if len(p) <= maxlen else (p[:25] + "…" + p[-25:])

    def on_close(self):
        try:
            if self.rec.is_recording: self.rec.stop()
        except Exception: pass
        self.destroy()

# small helper prompt
def simple_prompt(parent, title):
    top = tk.Toplevel(parent); top.title(title); top.grab_set()
    tk.Label(top, text=title).pack(padx=12, pady=(12,6))
    var = tk.StringVar()
    entry = tk.Entry(top, textvariable=var, width=32); entry.pack(padx=12, pady=6); entry.focus_set()
    resp = {"val": None}
    def ok():
        resp["val"] = var.get().strip(); top.destroy()
    def cancel():
        resp["val"] = None; top.destroy()
    row = tk.Frame(top); row.pack(pady=8)
    tk.Button(row, text="OK", width=8, command=ok).pack(side="left", padx=4)
    tk.Button(row, text="Cancel", width=8, command=cancel).pack(side="left", padx=4)
    parent.wait_window(top)
    return resp["val"]

if __name__ == "__main__":
    App().mainloop()