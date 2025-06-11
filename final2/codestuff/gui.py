import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import sys
from pathlib import Path

# adjust these imports if your PYTHONPATH is different
from final2.codestuff.retrieval.cli import get_context
from LLM.modelLoader import ModelLoader
from LLM.diagnosis import infer
from utils.prompts.validators.input_normalizer import (
    check_class_response,
    check_rank_response,
)
from utils.prompts.builders.build_prompts_from_json import (
    build_class_prompt,
    build_method_prompt,
    build_method_source_prompt,
)


def default_top5(json_path: Path) -> list[str]:
    # copy your default_top5 implementation here, or import if available
    ...


def step_class(json_path, tok, ctx, outdir, text):
    p = build_class_prompt(json_path, tok, ctx)[0]
    dst = outdir / f"{json_path.stem}_class.txt"
    dst.write_text(p, "utf-8")
    text.insert(tk.END, f"[class] prompt saved to {dst}\n")
    out = infer("class", dst, model, tok, device)
    text.insert(tk.END, f"[class] LLM → {out}\n")
    resp = check_class_response(out.read_text("utf-8"), json_path)
    text.insert(tk.END, f"[class] response → {resp}\n")


def step_rank(json_path, tok, ctx, outdir, text):
    prompts = build_method_prompt(json_path, tok, ctx)
    paths = []
    for idx, pr in enumerate(prompts, 1):
        dst = outdir / f"{json_path.stem}_rank_{idx}.txt"
        dst.write_text(pr, "utf-8")
        paths.append(dst)
    text.insert(tk.END, f"[rank] {len(paths)} prompts written\n")
    out = infer("rank", paths, model, tok, device)
    text.insert(tk.END, f"[rank] LLM → {out}\n")
    resp = check_rank_response(out.read_text("utf-8"), json_path)
    text.insert(tk.END, f"[rank] response → {resp}\n")


def step_source(json_path, tok, ctx, outdir, text):
    sigs = default_top5(json_path)
    prompts = build_method_source_prompt(json_path, sigs, tok, ctx)
    paths = []
    for idx, pr in enumerate(prompts, 1):
        dst = outdir / f"{json_path.stem}_source_{idx}.txt"
        dst.write_text(pr, "utf-8")
        paths.append(dst)
    text.insert(tk.END, f"[source] {len(paths)} prompts written\n")
    out = infer("source", paths, model, tok, device)
    text.insert(tk.END, f"[source] LLM → {out}\n")


class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ThesisPipeV2 GUI")
        self.geometry("700x500")

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.X)

        ttk.Label(frm, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.model_var).grid(row=0, column=1, sticky=tk.EW)

        ttk.Label(frm, text="Project:").grid(row=1, column=0, sticky=tk.W)
        self.project_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.project_var).grid(row=1, column=1, sticky=tk.EW)

        ttk.Label(frm, text="Bug ID:").grid(row=2, column=0, sticky=tk.W)
        self.bug_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.bug_var).grid(row=2, column=1, sticky=tk.EW)

        ttk.Label(frm, text="Output Dir:").grid(row=3, column=0, sticky=tk.W)
        self.out_var = tk.StringVar()
        out_entry = ttk.Entry(frm, textvariable=self.out_var)
        out_entry.grid(row=3, column=1, sticky=tk.EW)
        ttk.Button(frm, text="…", width=3, command=self._pick_dir).grid(row=3, column=2)

        frm.columnconfigure(1, weight=1)

        self.run_btn = ttk.Button(self, text="Run Pipeline", command=self._on_run)
        self.run_btn.pack(pady=5)

        self.log = scrolledtext.ScrolledText(self, state="normal")
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def _pick_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.out_var.set(d)

    def _on_run(self):
        self.run_btn.config(state=tk.DISABLED)
        self.log.delete("1.0", tk.END)
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self):
        try:
            model_name = self.model_var.get().strip()
            project = self.project_var.get().strip()
            bug_id = self.bug_var.get().strip()
            out_base = self.out_var.get().strip()
            if not (model_name and project and bug_id and out_base):
                raise ValueError("All fields are required")

            self.log.insert(tk.END, "▶ get_context…\n")
            get_context(project, bug_id, out_base, "")
            self.log.insert(tk.END, "✔ get_context\n")

            global model, tok, device
            loader = ModelLoader(model_name)
            tok, model, device = loader.loadModel()
            ctx = tok.model_max_length

            jp = Path(f"{out_base}.json").resolve()
            od = Path(out_base).resolve()
            od.mkdir(parents=True, exist_ok=True)

            # run 3 steps
            for step in (step_class, step_rank, step_source):
                step(jp, tok, ctx, od, self.log)
            self.log.insert(tk.END, "\n✅ Pipeline complete\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log.insert(tk.END, f"\n❌ {e}\n")
        finally:
            self.run_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()
