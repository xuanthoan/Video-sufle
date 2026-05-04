import json
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QSettings, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QProgressBar, QTextEdit, QVBoxLayout, QWidget,
)
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class VideoSpec:
    width: int
    height: int
    duration: float


@dataclass
class ProcessConfig:
    input_videos: List[Path]
    output_dir: Path
    image_files: List[Path]
    focus_mode: str
    overlap_pct_of_main: float
    auto_open: bool


class FFmpegRunner:
    def __init__(self) -> None:
        self.ffmpeg = self._resolve_binary("ffmpeg")
        self.ffprobe = self._resolve_binary("ffprobe")
        self.current_process: Optional[subprocess.Popen] = None

    @staticmethod
    def _resolve_binary(name: str) -> str:
        exe = f"{name}.exe" if sys.platform.startswith("win") else name
        local = Path(sys.argv[0]).resolve().parent / exe
        return str(local) if local.exists() else exe

    def run(self, args: Sequence[str], step: str = "") -> None:
        cmd = [self.ffmpeg, *args]
        self.current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = self.current_process.communicate()
        rc = self.current_process.returncode
        self.current_process = None
        if rc != 0:
            tail = "\n".join((err or "").splitlines()[-20:])
            raise RuntimeError(f"[{step}] FFmpeg lỗi (code={rc})\n{tail}")

    def stop_current(self) -> None:
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()

    def probe_video(self, path: Path) -> VideoSpec:
        cmd = [self.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
        return VideoSpec(int(stream["width"]), int(stream["height"]), float(data["format"]["duration"]))


class VideoEngine:
    def __init__(self, runner: FFmpegRunner, logger, stop_check):
        self.runner = runner
        self.log = logger
        self.stop_check = stop_check

    def ensure_not_stopped(self):
        if self.stop_check():
            raise RuntimeError("Đã dừng theo yêu cầu người dùng")

    def process_one(self, input_video: Path, output_dir: Path, image_file: Optional[Path], focus_mode: str, overlap_pct_of_main: float) -> Path:
        td = tempfile.mkdtemp(prefix="shuffle_")
        work = Path(td)
        try:
            spec = self.runner.probe_video(input_video)
            self.log(f"[{input_video.name}] {spec.width}x{spec.height} - {spec.duration:.2f}s")
            audio = work / "audio.aac"
            shuffled = work / "shuffled.mp4"
            composed = work / "composed.mp4"
            final_out = output_dir / f"{input_video.stem}_processed.mp4"

            self.ensure_not_stopped(); self.log("  Bước 1/6: Tách audio")
            self.runner.run(["-y", "-i", str(input_video), "-vn", "-acodec", "copy", str(audio)], "extract_audio")
            self.ensure_not_stopped(); self.log("  Bước 2/6: Detect cảnh")
            scenes = self._detect_or_fallback_scenes(input_video, spec.duration)
            self.log(f"  Số segment: {len(scenes)}")
            self.ensure_not_stopped(); self.log("  Bước 3/6: Cắt segment")
            parts = self._cut_segments(input_video, scenes, work)
            self.ensure_not_stopped(); self.log("  Bước 4/6: Shuffle + concat")
            self._concat_segments(self._shuffle_keep_first(parts), shuffled)

            if image_file:
                self.ensure_not_stopped(); self.log("  Bước 5/6: Ghép video + ảnh + overlap")
                self._compose_layout(shuffled, image_file, spec, focus_mode, overlap_pct_of_main, composed)
            else:
                composed = shuffled

            self.ensure_not_stopped(); self.log("  Bước 6/6: Gắn lại audio")
            self._mux_audio(composed, audio, final_out)
            return final_out
        finally:
            for _ in range(8):
                try:
                    shutil.rmtree(work, ignore_errors=False)
                    break
                except PermissionError:
                    time.sleep(0.25)

    def _detect_or_fallback_scenes(self, video: Path, duration: float) -> List[Tuple[float, float]]:
        scene_list = []
        try:
            v = open_video(str(video)); manager = SceneManager(); manager.add_detector(ContentDetector(threshold=27.0)); manager.detect_scenes(v)
            scene_list = [(s[0].get_seconds(), s[1].get_seconds()) for s in manager.get_scene_list()]
        except Exception as exc:
            self.log(f"  Cảnh báo detect cảnh: {exc}")
        if len(scene_list) <= 1:
            self.log("  Fallback: chia ngẫu nhiên 3-5 giây")
            t = 0.0
            while t < duration:
                e = min(duration, t + random.uniform(3.0, 5.0)); scene_list.append((t, e)); t = e
        return scene_list

    def _cut_segments(self, video: Path, scenes: List[Tuple[float, float]], work: Path) -> List[Path]:
        out = []
        for idx, (start, end) in enumerate(scenes):
            self.ensure_not_stopped()
            seg = work / f"seg_{idx:04d}.mp4"
            self.runner.run(["-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video), "-c", "copy", str(seg)], f"cut_{idx}")
            out.append(seg)
        return out

    @staticmethod
    def _shuffle_keep_first(parts: List[Path]) -> List[Path]:
        if len(parts) <= 2: return parts
        head, tail = parts[0], parts[1:]; random.shuffle(tail); return [head, *tail]

    def _concat_segments(self, parts: List[Path], out: Path) -> None:
        cf = out.parent / "concat.txt"
        cf.write_text("\n".join(f"file {shlex.quote(str(p))}" for p in parts), encoding="utf-8")
        self.runner.run(["-y", "-f", "concat", "-safe", "0", "-i", str(cf), "-c", "copy", str(out)], "concat")

    def _compose_layout(self, video: Path, image: Path, spec: VideoSpec, focus: str, overlap_pct_of_main: float, out: Path) -> None:
        w, h = spec.width, spec.height
        main_h = int(h * 0.65)
        overlap_h = min(main_h, max(2, int(main_h * overlap_pct_of_main / 100.0)))
        image_h = h - main_h
        offset_main = max(0, h - int(h * 0.70))
        y_expr = "0" if focus == "top" else ("ih-oh" if focus == "bottom" else "(ih-oh)/2")
        duration = max(0.1, spec.duration)

        # Quy tắc ảnh: nếu nhỏ hơn W -> scale tăng lên W; nếu >= W -> không scale, crop trực tiếp.
        img_chain = f"scale='if(lt(iw,{w}),{w},iw)':-1,crop={w}:{image_h}:0:{y_expr}"
        fc = (
            f"[1:v]{img_chain},trim=duration={duration:.3f}[img];"
            f"[0:v]crop={w}:{main_h}:0:{offset_main},trim=duration={duration:.3f}[main];"
            f"[main]crop={w}:{overlap_h}:0:{main_h-overlap_h},format=yuva420p,geq=lum='p(X,Y)':a='255*(1-Y/{overlap_h})'[fade];"
            f"color=c=black:s={w}x{h}:d={duration:.3f}[base];"
            f"[base][img]overlay=0:{h-image_h}[tmp1];"
            f"[tmp1][main]overlay=0:0[tmp2];"
            f"[tmp2][fade]overlay=0:{main_h}[v]"
        )
        self.runner.run([
            "-y", "-i", str(video), "-loop", "1", "-t", f"{duration:.3f}", "-i", str(image),
            "-filter_complex", fc, "-map", "[v]", "-t", f"{duration:.3f}", "-shortest",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(out)
        ], "compose")

    def _mux_audio(self, video: Path, audio: Path, out: Path) -> None:
        try:
            self.runner.run(["-y", "-i", str(video), "-i", str(audio), "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], "mux_copy")
        except Exception:
            self.runner.run(["-y", "-i", str(video), "-i", str(audio), "-c:v", "libx264", "-c:a", "aac", "-shortest", str(out)], "mux_fallback")


class Worker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    done_signal = Signal(int, int)

    def __init__(self, cfg: ProcessConfig):
        super().__init__(); self.cfg = cfg; self.stop_requested = False; self.runner: Optional[FFmpegRunner] = None

    def request_stop(self):
        self.stop_requested = True
        if self.runner: self.runner.stop_current()

    def run(self) -> None:
        self.runner = FFmpegRunner()
        engine = VideoEngine(self.runner, self.log_signal.emit, lambda: self.stop_requested)
        ok = fail = 0; total = len(self.cfg.input_videos)
        self.log_signal.emit(f"Bắt đầu batch: {total} video")
        for i, video in enumerate(self.cfg.input_videos, start=1):
            if self.stop_requested:
                self.log_signal.emit("Đã nhận lệnh dừng.")
                break
            self.log_signal.emit(f"=== [{i}/{total}] {video.name} ===")
            try:
                img = random.choice(self.cfg.image_files) if self.cfg.image_files else None
                out = engine.process_one(video, self.cfg.output_dir, img, self.cfg.focus_mode, self.cfg.overlap_pct_of_main)
                self.log_signal.emit(f"OK: {out}"); ok += 1
            except Exception as exc:
                self.log_signal.emit(f"LỖI: {video.name} -> {exc}"); fail += 1
            self.progress_signal.emit(i, total)
        self.log_signal.emit("Kết thúc batch")
        self.done_signal.emit(ok, fail)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Xáo Trộn Video + Ghép Ảnh Tự Động")
        self.settings = QSettings("VideoSufle", "AutoShuffle")
        self.worker: Optional[Worker] = None
        self.image_files: List[Path] = []
        self.output_dir: Optional[Path] = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); layout = QVBoxLayout(root)
        self.list_widget = QListWidget(); layout.addWidget(QLabel("Danh sách video đầu vào")); layout.addWidget(self.list_widget)
        r=QHBoxLayout()
        for t,fn in [("Thêm file",self.add_files),("Thêm thư mục",self.add_folder),("Xóa",self.remove_items)]:
            b=QPushButton(t); b.clicked.connect(fn); r.addWidget(b)
        layout.addLayout(r)

        self.btn_select_images = QPushButton("Chọn 1 hoặc nhiều ảnh")
        self.btn_select_images.clicked.connect(self.select_images)
        layout.addWidget(self.btn_select_images)
        self.image_label = QLabel("Chưa chọn ảnh")
        layout.addWidget(self.image_label)

        fr=QHBoxLayout(); fr.addWidget(QLabel("Điểm crop ảnh")); self.focus_combo=QComboBox(); self.focus_combo.addItems(["center","top","bottom"]); fr.addWidget(self.focus_combo); layout.addLayout(fr)
        orow=QHBoxLayout(); orow.addWidget(QLabel("Overlap fade (% chiều cao video chính)")); self.overlap_spin=QDoubleSpinBox(); self.overlap_spin.setRange(1.0,50.0); self.overlap_spin.setValue(7.7); orow.addWidget(self.overlap_spin); layout.addLayout(orow)

        self.auto_open = QCheckBox("Tự mở thư mục output sau khi xử lý"); layout.addWidget(self.auto_open)
        out=QHBoxLayout()
        bo=QPushButton("Chọn thư mục output"); bo.clicked.connect(self.select_output)
        bop=QPushButton("Mở output"); bop.clicked.connect(self.open_output)
        out.addWidget(bo); out.addWidget(bop); layout.addLayout(out)

        self.progress = QProgressBar(); layout.addWidget(self.progress)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); layout.addWidget(self.log_box)
        br=QHBoxLayout()
        self.btn_start = QPushButton("Bắt đầu batch"); self.btn_start.clicked.connect(self.start_batch)
        self.btn_end = QPushButton("End / Dừng tất cả"); self.btn_end.clicked.connect(self.stop_batch)
        br.addWidget(self.btn_start); br.addWidget(self.btn_end); layout.addLayout(br)
        self.setCentralWidget(root)

    def add_files(self):
        files,_=QFileDialog.getOpenFileNames(self,"Chọn video","","Video (*.mp4 *.mov *.avi *.mkv)")
        for f in files: self.list_widget.addItem(QListWidgetItem(f))

    def add_folder(self):
        folder=QFileDialog.getExistingDirectory(self,"Chọn thư mục")
        if folder:
            for p in Path(folder).iterdir():
                if p.suffix.lower() in VIDEO_EXTS: self.list_widget.addItem(QListWidgetItem(str(p)))

    def remove_items(self):
        selected=self.list_widget.selectedItems()
        if selected:
            for it in selected: self.list_widget.takeItem(self.list_widget.row(it))
        else: self.list_widget.clear()

    def select_images(self):
        files,_=QFileDialog.getOpenFileNames(self,"Chọn 1 hoặc nhiều ảnh","","Ảnh (*.jpg *.jpeg *.png *.webp)")
        self.image_files=[Path(f) for f in files]
        self.image_label.setText(f"Đã chọn {len(self.image_files)} ảnh" if self.image_files else "Chưa chọn ảnh")

    def select_output(self):
        folder=QFileDialog.getExistingDirectory(self,"Chọn thư mục output")
        if folder: self.output_dir=Path(folder)

    def open_output(self):
        path=self.settings.value("last_openable_output","",str)
        if not path or not Path(path).exists():
            QMessageBox.warning(self,"Thiếu dữ liệu","Không tìm thấy thư mục output."); return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def start_batch(self):
        inputs=[Path(self.list_widget.item(i).text()) for i in range(self.list_widget.count())]
        if not inputs:
            QMessageBox.warning(self,"Thiếu dữ liệu","Vui lòng thêm video."); return
        out=self.output_dir or Path.cwd()/"output_processed"; out.mkdir(parents=True, exist_ok=True)
        cfg=ProcessConfig(inputs,out,self.image_files,self.focus_combo.currentText(),self.overlap_spin.value(),self.auto_open.isChecked())
        self.worker=Worker(cfg)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.done_signal.connect(self.on_done)
        self.btn_start.setEnabled(False)
        self.worker.start()

    def stop_batch(self):
        if self.worker and self.worker.isRunning():
            self.log("Đang gửi lệnh dừng...")
            self.worker.request_stop()

    def log(self, text: str): self.log_box.append(text)
    def on_progress(self, done: int, total: int): self.progress.setMaximum(total); self.progress.setValue(done)

    def on_done(self, ok: int, fail: int):
        self.btn_start.setEnabled(True)
        output=str(self.output_dir or (Path.cwd()/"output_processed"))
        self.settings.setValue("last_openable_output",output)
        self.log(f"Hoàn tất. thành công={ok}, lỗi={fail}")
        if self.auto_open.isChecked() and Path(output).exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(output))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Đang xử lý", "Đang xử lý, vui lòng bấm End và chờ dừng trước khi đóng.")
            event.ignore(); return
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.resize(920, 760); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
