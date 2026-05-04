import json
import random
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QSettings, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class VideoSpec:
    width: int
    height: int
    duration: float


@dataclass
class ProcessConfig:
    input_videos: List[Path]
    output_dir: Path
    image_mode: str  # single|folder|none
    single_image: Optional[Path]
    image_folder: Optional[Path]
    focus_mode: str  # center|top|bottom
    overlap_pct_of_main: float
    auto_open: bool


class FFmpegRunner:
    def __init__(self) -> None:
        self.ffmpeg = self._resolve_binary("ffmpeg")
        self.ffprobe = self._resolve_binary("ffprobe")

    @staticmethod
    def _resolve_binary(name: str) -> str:
        exe = f"{name}.exe" if sys.platform.startswith("win") else name
        local = Path(sys.argv[0]).resolve().parent / exe
        return str(local) if local.exists() else exe

    def run(self, args: Sequence[str]) -> None:
        cmd = [self.ffmpeg, *args]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def run_capture(self, args: Sequence[str]) -> str:
        cmd = [self.ffprobe, *args]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout

    def probe_video(self, path: Path) -> VideoSpec:
        out = self.run_capture([
            "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
        ])
        data = json.loads(out)
        stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
        return VideoSpec(width=int(stream["width"]), height=int(stream["height"]), duration=float(data["format"]["duration"]))


class VideoEngine:
    def __init__(self, runner: FFmpegRunner, logger):
        self.runner = runner
        self.log = logger

    def process_one(self, input_video: Path, output_dir: Path, image_file: Optional[Path], focus_mode: str, overlap_pct_of_main: float) -> Path:
        td = tempfile.mkdtemp(prefix="shuffle_")
        work = Path(td)
        try:
            spec = self.runner.probe_video(input_video)
            self.log(f"[{input_video.name}] video={spec.width}x{spec.height} duration={spec.duration:.2f}s")
            audio = work / "audio.aac"
            shuffled = work / "shuffled.mp4"
            composed = work / "composed.mp4"
            final_out = output_dir / f"{input_video.stem}_processed.mp4"

            self._extract_audio(input_video, audio)
            scenes = self._detect_or_fallback_scenes(input_video, spec.duration)
            parts = self._cut_segments(input_video, scenes, work)
            shuffled_parts = self._shuffle_keep_first(parts)
            self._concat_segments(shuffled_parts, shuffled)

            if image_file:
                self._compose_layout(shuffled, image_file, spec, focus_mode, overlap_pct_of_main, composed)
            else:
                composed = shuffled

            self._mux_audio(composed, audio, final_out)
            return final_out
        finally:
            self._safe_cleanup_dir(work)

    def _safe_cleanup_dir(self, path: Path) -> None:
        import shutil
        import time

        for _ in range(8):
            try:
                shutil.rmtree(path, ignore_errors=False)
                return
            except PermissionError:
                time.sleep(0.25)
        shutil.rmtree(path, ignore_errors=True)

    def _extract_audio(self, video: Path, audio: Path) -> None:
        self.runner.run(["-y", "-i", str(video), "-vn", "-acodec", "copy", str(audio)])

    def _detect_or_fallback_scenes(self, video: Path, duration: float) -> List[Tuple[float, float]]:
        scene_list: List[Tuple[float, float]] = []
        try:
            v = open_video(str(video))
            manager = SceneManager()
            manager.add_detector(ContentDetector(threshold=27.0))
            manager.detect_scenes(v)
            raw = manager.get_scene_list()
            scene_list = [(s[0].get_seconds(), s[1].get_seconds()) for s in raw]
        except Exception as exc:
            self.log(f"Scene detection warning: {exc}")

        if len(scene_list) <= 1:
            self.log("Fallback random segments 3-5s")
            scene_list = []
            t = 0.0
            while t < duration:
                seg = random.uniform(3.0, 5.0)
                end = min(duration, t + seg)
                scene_list.append((t, end))
                t = end
        return scene_list

    def _cut_segments(self, video: Path, scenes: List[Tuple[float, float]], work: Path) -> List[Path]:
        out: List[Path] = []
        for idx, (start, end) in enumerate(scenes):
            seg = work / f"seg_{idx:04d}.mp4"
            self.runner.run([
                "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video), "-c", "copy", str(seg),
            ])
            out.append(seg)
        return out

    @staticmethod
    def _shuffle_keep_first(parts: List[Path]) -> List[Path]:
        if len(parts) <= 2:
            return parts
        head, tail = parts[0], parts[1:]
        random.shuffle(tail)
        return [head, *tail]

    def _concat_segments(self, parts: List[Path], out: Path) -> None:
        concat_file = out.parent / "concat.txt"
        concat_file.write_text("\n".join(f"file {shlex.quote(str(p))}" for p in parts), encoding="utf-8")
        self.runner.run(["-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(out)])

    def _compose_layout(self, video: Path, image: Path, spec: VideoSpec, focus: str, overlap_pct_of_main: float, out: Path) -> None:
        w, h = spec.width, spec.height
        main_h = int(h * 0.65)
        overlap_h = max(2, int(main_h * (overlap_pct_of_main / 100.0)))
        overlap_h = min(overlap_h, main_h)
        image_h = h - main_h
        visible_video_total = int(h * 0.70)
        offset_main = max(0, h - visible_video_total)

        if focus == "top":
            y_expr = "0"
        elif focus == "bottom":
            y_expr = "ih-oh"
        else:
            y_expr = "(ih-oh)/2"

        filter_complex = (
            f"[1:v]scale={w}:-1,crop={w}:{image_h}:0:{y_expr}[img];"
            f"[0:v]crop={w}:{main_h}:0:{offset_main}[main];"
            f"[main]crop={w}:{overlap_h}:0:{main_h-overlap_h},format=yuva420p,"
            f"geq=lum='p(X,Y)':a='255*(1-Y/{overlap_h})'[fade];"
            f"color=c=black:s={w}x{h}:d=1[base];"
            f"[base][img]overlay=0:{h-image_h}[tmp1];"
            f"[tmp1][main]overlay=0:0[tmp2];"
            f"[tmp2][fade]overlay=0:{main_h}[v]"
        )

        self.runner.run([
            "-y", "-i", str(video), "-loop", "1", "-i", str(image),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-shortest", "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(out)
        ])

    def _mux_audio(self, video: Path, audio: Path, out: Path) -> None:
        try:
            self.runner.run(["-y", "-i", str(video), "-i", str(audio), "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)])
        except subprocess.CalledProcessError:
            self.runner.run(["-y", "-i", str(video), "-i", str(audio), "-c:v", "libx264", "-c:a", "aac", "-shortest", str(out)])


class Worker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    done_signal = Signal(int, int)

    def __init__(self, cfg: ProcessConfig):
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        runner = FFmpegRunner()
        engine = VideoEngine(runner, self.log_signal.emit)
        ok = 0
        fail = 0
        for i, video in enumerate(self.cfg.input_videos, start=1):
            try:
                image = self.pick_image() if self.cfg.image_mode != "none" else None
                out = engine.process_one(video, self.cfg.output_dir, image, self.cfg.focus_mode, self.cfg.overlap_pct_of_main)
                self.log_signal.emit(f"OK: {out}")
                ok += 1
            except Exception as exc:
                self.log_signal.emit(f"FAIL: {video.name} -> {exc}")
                fail += 1
            self.progress_signal.emit(i, len(self.cfg.input_videos))
        self.done_signal.emit(ok, fail)

    def pick_image(self) -> Optional[Path]:
        if self.cfg.image_mode == "single":
            return self.cfg.single_image
        if self.cfg.image_mode == "folder" and self.cfg.image_folder:
            items = [p for p in self.cfg.image_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            return random.choice(items) if items else None
        return None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Auto Video Shuffle + Image Compositor")
        self.settings = QSettings("VideoSufle", "AutoShuffle")
        self.worker: Optional[Worker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        self.list_widget = QListWidget()
        layout.addWidget(QLabel("Input videos"))
        layout.addWidget(self.list_widget)

        row = QHBoxLayout()
        btn_files = QPushButton("Add Files")
        btn_files.clicked.connect(self.add_files)
        btn_folder = QPushButton("Add Folder")
        btn_folder.clicked.connect(self.add_folder)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self.remove_items)
        row.addWidget(btn_files); row.addWidget(btn_folder); row.addWidget(btn_remove)
        layout.addLayout(row)

        self.btn_single_image = QPushButton("Select Single Image")
        self.btn_single_image.clicked.connect(self.select_single_image)
        self.btn_folder_image = QPushButton("Select Image Folder")
        self.btn_folder_image.clicked.connect(self.select_image_folder)
        layout.addWidget(self.btn_single_image)
        layout.addWidget(self.btn_folder_image)
        self.image_label = QLabel("Image mode: none")
        layout.addWidget(self.image_label)

        focus_row = QHBoxLayout()
        focus_row.addWidget(QLabel("Image crop focus"))
        self.focus_combo = QComboBox()
        self.focus_combo.addItems(["center", "top", "bottom"])
        focus_row.addWidget(self.focus_combo)
        layout.addLayout(focus_row)

        overlap_row = QHBoxLayout()
        overlap_row.addWidget(QLabel("Fade overlap (% of main video height)"))
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(1.0, 50.0)
        self.overlap_spin.setDecimals(1)
        self.overlap_spin.setSingleStep(0.5)
        self.overlap_spin.setValue(7.7)
        overlap_row.addWidget(self.overlap_spin)
        layout.addLayout(overlap_row)

        self.auto_open = QCheckBox("Auto open output folder after processing")
        layout.addWidget(self.auto_open)

        out_row = QHBoxLayout()
        self.btn_out = QPushButton("Select Output Folder")
        self.btn_out.clicked.connect(self.select_output)
        self.btn_open_out = QPushButton("Open Output")
        self.btn_open_out.clicked.connect(self.open_output)
        out_row.addWidget(self.btn_out); out_row.addWidget(self.btn_open_out)
        layout.addLayout(out_row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.btn_start = QPushButton("Start Batch")
        self.btn_start.clicked.connect(self.start_batch)
        layout.addWidget(self.btn_start)

        self.single_image: Optional[Path] = None
        self.image_folder: Optional[Path] = None
        self.output_dir: Optional[Path] = None

        self.setCentralWidget(root)

    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Choose Videos", "", "Video (*.mp4 *.mov *.avi *.mkv)")
        for f in files:
            self.list_widget.addItem(QListWidgetItem(f))

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose Folder")
        if not folder:
            return
        for p in Path(folder).iterdir():
            if p.suffix.lower() in VIDEO_EXTS:
                self.list_widget.addItem(QListWidgetItem(str(p)))

    def remove_items(self) -> None:
        selected = self.list_widget.selectedItems()
        if selected:
            for item in selected:
                self.list_widget.takeItem(self.list_widget.row(item))
            return
        self.list_widget.clear()

    def select_single_image(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Image (*.jpg *.jpeg *.png *.webp)")
        if file:
            self.single_image = Path(file)
            self.image_folder = None
            self.image_label.setText(f"Image mode: single ({self.single_image.name})")

    def select_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose Image Folder")
        if folder:
            self.image_folder = Path(folder)
            self.single_image = None
            self.image_label.setText(f"Image mode: folder ({self.image_folder})")

    def select_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose Output Folder")
        if folder:
            self.output_dir = Path(folder)

    def open_output(self) -> None:
        path = self.settings.value("last_openable_output", "", str)
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Missing", "Output folder not found.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def start_batch(self) -> None:
        inputs = [Path(self.list_widget.item(i).text()) for i in range(self.list_widget.count())]
        if not inputs:
            QMessageBox.warning(self, "Missing", "Please add videos.")
            return
        output = self.output_dir or Path.cwd() / "output_processed"
        output.mkdir(parents=True, exist_ok=True)

        if self.single_image:
            image_mode = "single"
        elif self.image_folder:
            image_mode = "folder"
        else:
            image_mode = "none"

        focus = self.focus_combo.currentText()

        cfg = ProcessConfig(
            input_videos=inputs,
            output_dir=output,
            image_mode=image_mode,
            single_image=self.single_image,
            image_folder=self.image_folder,
            focus_mode=focus,
            overlap_pct_of_main=self.overlap_spin.value(),
            auto_open=self.auto_open.isChecked(),
        )
        self.worker = Worker(cfg)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.done_signal.connect(self.on_done)
        self.btn_start.setEnabled(False)
        self.worker.start()

    def log(self, text: str) -> None:
        self.log_box.append(text)

    def on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def on_done(self, ok: int, fail: int) -> None:
        self.btn_start.setEnabled(True)
        output = str(self.output_dir or (Path.cwd() / "output_processed"))
        self.settings.setValue("last_openable_output", output)
        self.log(f"Done. success={ok}, fail={fail}")
        if self.auto_open.isChecked() and Path(output).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(output))

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Processing", "Đang xử lý. Vui lòng chờ hoàn tất trước khi đóng.")
            event.ignore()
            return
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
