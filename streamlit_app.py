from __future__ import annotations

import csv
import io
import json
import random
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ijson
import streamlit as st

ARCHIVE = Path.home() / "Downloads" / "archive"
OUTPUT = Path.home() / "Downloads" / "poools_generated"
ROADSIFT_CLASS_MAP = {"pedestrian": "person", "bicycle": "bike", "motorcycle": "motor"}


def image_names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.jpg") if p.is_file())


def sample_by_index(names: list[str], count: int, seed: int) -> list[tuple[int, str]]:
    # Indexes are 1-based for the manifest; Python maps them with index - 1.
    indexes = random.Random(seed).sample(range(1, len(names) + 1), count)
    return [(index, names[index - 1]) for index in indexes]


def sequence(name: str) -> str:
    # BDD100K image names start with the video/sequence ID.
    return Path(name).stem.split("-", 1)[0]


def records_for(json_path: Path, wanted: set[str]):
    pending = set(wanted)
    if not pending:
        return
    with json_path.open("rb") as source:
        for row in ijson.items(source, "item", use_float=True):
            if row.get("name") in pending:
                yield row
                pending.remove(row["name"])
                if not pending:
                    return


def manifest_sequences(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["sequence_id"] for row in csv.DictReader(f)}


def write_manifest(path: Path, indexed_names: list[tuple[int, str]], split: str):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "sample_id", "filename", "source_name", "sequence_id", "source_split"])
        writer.writerows((i, Path(name).stem, name, name, sequence(name), split)
                         for i, name in indexed_names)


def generated_sequences(output: Path, kind: str) -> set[str]:
    sequences = set()
    for manifest in output.glob(f"{kind}*/manifest.csv"):
        sequences.update(manifest_sequences(manifest))
    return sequences


def create_run_folder(output: Path, kind: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"{kind}_{timestamp}"
    suffix = 0
    while True:
        folder = output / (base_name if suffix == 0 else f"{base_name}_{suffix}")
        try:
            folder.mkdir(exist_ok=False)
            return folder
        except FileExistsError:
            suffix += 1


def build_pool(root: Path, output: Path, count: int, seed: int) -> tuple[list[str], Path]:
    train_dir = root / "train" / "images"
    train_names = image_names(train_dir)
    excluded_sequences = generated_sequences(output, "holdout")
    candidates = [name for name in train_names if sequence(name) not in excluded_sequences]
    if len(candidates) < count:
        raise ValueError(f"Chỉ còn {len(candidates)} ảnh train đủ điều kiện; cần {count}.")

    selected = sample_by_index(candidates, count, seed)
    pool_dir = create_run_folder(output, "pool")
    images_dir = pool_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for _, name in selected:
        shutil.copy2(train_dir / name, images_dir / name)
    write_manifest(pool_dir / "manifest.csv", selected, "train")
    return [name for _, name in selected], pool_dir


def build_holdout(root: Path, output: Path, count: int, seed: int) -> tuple[list[str], Path]:
    val_dir = root / "val" / "images"
    excluded_sequences = generated_sequences(output, "pool")
    one_per_sequence: dict[str, str] = {}
    for name in image_names(val_dir):
        seq = sequence(name)
        if seq not in excluded_sequences:
            one_per_sequence.setdefault(seq, name)
    candidates = list(one_per_sequence.values())
    if len(candidates) < count:
        raise ValueError(f"Chỉ còn {len(candidates)} chuỗi val đủ điều kiện; cần {count}.")

    selected = sample_by_index(candidates, count, seed)
    holdout_dir = create_run_folder(output, "holdout")
    images_dir = holdout_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for _, name in selected:
        shutil.copy2(val_dir / name, images_dir / name)
    write_manifest(holdout_dir / "manifest.csv", selected, "val")

    label_path = root / "val" / "annotations" / "bdd100k_labels_images_val.json"
    labels = list(records_for(label_path, {name for _, name in selected}))
    with (holdout_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    return [name for _, name in selected], holdout_dir


def parse_batch_mapping(text: str) -> tuple[dict[str, str], str | None]:
    rows = csv.DictReader(io.StringIO(text))
    if not {"source_name", "sample_id"}.issubset(rows.fieldnames or []):
        raise ValueError("Manifest batch cần có hai cột source_name,sample_id.")
    mapping = {}
    splits = set()
    for row in rows:
        source_name = (row.get("source_name") or "").strip()
        sample_id = (row.get("sample_id") or "").strip()
        if not source_name or not sample_id:
            continue
        previous = mapping.setdefault(source_name, sample_id)
        if previous != sample_id:
            raise ValueError(f"Manifest có nhiều sample_id cho cùng source_name: {source_name}")
        split = (row.get("source_split") or "").strip().lower()
        if split:
            splits.add(split)
    if not mapping:
        raise ValueError("Manifest batch không có dòng ánh xạ hợp lệ.")
    if len(splits) > 1:
        raise ValueError("Batch chứa nhiều source_split; hãy nhập riêng từng batch train hoặc val.")
    detected_split = next(iter(splits), None)
    if detected_split and detected_split not in {"train", "val"}:
        raise ValueError(f"source_split không được hỗ trợ: {detected_split}")
    return mapping, detected_split


def _json_batch_mapping(text: str) -> tuple[dict[str, str], str | None]:
    document = json.loads(text)
    rows = document if isinstance(document, list) else document.get("items", [])
    mapping = {}
    splits = set()
    for row in rows:
        sample = row.get("sample", row) if isinstance(row, dict) else {}
        attributes = sample.get("attributes") or {}
        sample_id = sample.get("id") or row.get("sample_id")
        source_name = attributes.get("source_name") or row.get("source_name")
        if source_name and sample_id:
            mapping[str(source_name)] = str(sample_id)
            split = attributes.get("source_split") or row.get("source_split")
            if split:
                splits.add(str(split).strip().lower())
    if not mapping:
        raise ValueError("JSON batch không có source_name và sample_id để đối chiếu RoadSift.")
    if len(splits) > 1:
        raise ValueError("Batch RoadSift chứa nhiều source_split; hãy nhập riêng từng batch train hoặc val.")
    detected_split = next(iter(splits), None)
    if detected_split and detected_split not in {"train", "val"}:
        raise ValueError(f"source_split không được hỗ trợ: {detected_split}")
    return mapping, detected_split


def batch_input_from_path(raw_path: str) -> tuple[list[str], dict[str, str], str | None]:
    """Read a local batch folder/archive or a RoadSift samples export without extracting files."""
    cleaned = raw_path.strip().strip('"').strip("'")
    if not cleaned:
        return [], {}, None
    path = Path(cleaned).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy đường dẫn batch: {path}")

    image_names: list[str] = []
    mapping: dict[str, str] = {}
    detected_split = None
    if path.is_dir():
        image_names = [p.name for p in path.rglob("*")
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
        manifest = path / "manifest.csv"
        if manifest.is_file():
            mapping, detected_split = parse_batch_mapping(manifest.read_text(encoding="utf-8-sig"))
        else:
            batch_json = path / "batch-samples.json"
            if batch_json.is_file():
                mapping, detected_split = _json_batch_mapping(batch_json.read_text(encoding="utf-8-sig"))
    elif path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
        image_names = [path.name]
    elif path.is_file() and path.suffix.lower() == ".csv":
        mapping, detected_split = parse_batch_mapping(path.read_text(encoding="utf-8-sig"))
    elif path.is_file() and path.suffix.lower() == ".json":
        mapping, detected_split = _json_batch_mapping(path.read_text(encoding="utf-8-sig"))
    elif path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            image_names = [Path(name.replace("\\", "/")).name for name in members
                           if Path(name.replace("\\", "/")).suffix.lower() in {".jpg", ".jpeg"}]
            manifests = [name for name in members if Path(name.replace("\\", "/")).name.lower() == "manifest.csv"]
            json_manifests = [name for name in members
                              if Path(name.replace("\\", "/")).name.lower() == "batch-samples.json"]
            if len(manifests) > 1 or len(json_manifests) > 1 or (manifests and json_manifests):
                raise ValueError("Batch ZIP chỉ nên có một manifest.csv hoặc batch-samples.json.")
            if manifests:
                mapping, detected_split = parse_batch_mapping(archive.read(manifests[0]).decode("utf-8-sig"))
            elif json_manifests:
                mapping, detected_split = _json_batch_mapping(archive.read(json_manifests[0]).decode("utf-8-sig"))
    else:
        raise ValueError("Dán đường dẫn tới thư mục batch, ZIP, CSV manifest, JSON batch hoặc ảnh JPG.")

    if len(image_names) != len(set(image_names)):
        raise ValueError("Batch có tên ảnh trùng nhau; cần giữ tên nguồn duy nhất để dò nhãn.")
    if mapping and image_names:
        image_names_set = set(image_names)
        image_stems = {Path(name).stem for name in image_names}
        mapping = {source_name: sample_id for source_name, sample_id in mapping.items()
                   if source_name in image_names_set or Path(source_name).stem in image_stems
                   or sample_id in image_stems}
        if not mapping:
            raise ValueError("Ảnh trong batch không khớp source_name hoặc sample_id trong manifest.")
    names = sorted(mapping) if mapping else sorted(image_names)
    if not names:
        raise ValueError("Không tìm thấy ảnh JPG/JPEG hoặc manifest batch hợp lệ tại đường dẫn này.")
    return names, mapping, detected_split


def export_batch_labels(root: Path, names: list[str], mapping: dict[str, str], split: str):
    annotation_path = root / split / "annotations" / f"bdd100k_labels_images_{split}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhãn BDD100K: {annotation_path}")

    records = {record["name"]: record for record in records_for(annotation_path, set(names))}
    exported, missing = [], []
    for name in names:
        record = records.get(name)
        if record is None:
            missing.append(name)
            continue
        output_record = dict(record)
        output_record["name"] = mapping.get(name, Path(name).stem)
        output_record["labels"] = [
            {**label, "category": ROADSIFT_CLASS_MAP.get(label.get("category"), label.get("category"))}
            if isinstance(label, dict) else label
            for label in (record.get("labels") or [])
        ]
        exported.append(output_record)
    return exported, missing


st.set_page_config(page_title="BDD100K Pool Tool", page_icon="B", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at top, #11263a, #0b1320 58%, #080d16);}
.block-container {max-width: 1120px; padding-top: 2rem;}
button[kind="primary"] {background:#087eae;border:1px solid #23b8e4;color:white;}
</style>
""", unsafe_allow_html=True)

st.title("BDD100K Pool Builder")
st.caption("Tool cục bộ: tạo Pool, tạo Holdout độc lập và nhập batch để xuất nhãn.")
root = Path(st.text_input("Thư mục archive", str(ARCHIVE))).expanduser()
output = Path(st.text_input("Thư mục output", str(OUTPUT))).expanduser()
st.caption("Mỗi lần tạo sẽ lưu vào thư mục con mới: pool_<timestamp> hoặc holdout_<timestamp> (giờ Việt Nam).")

tab_pool, tab_holdout, tab_import_labels = st.tabs(
    ["Tạo Pool", "Tạo Holdout", "Nhập batch · Xuất nhãn"]
)

with tab_pool:
    st.subheader("Tạo Pool từ ảnh train")
    train_count = len(image_names(root / "train" / "images"))
    st.caption(f"Có {train_count:,} ảnh JPG trong train/images. Pool chỉ chứa ảnh, không kèm nhãn.")
    pool_n = st.number_input("Số ảnh Pool", 1, 5000, 200, 20)
    pool_seed = st.number_input("Seed Pool", 0, 2147483647, 42)
    if st.button("Tạo Pool", type="primary"):
        try:
            pool, pool_dir = build_pool(root, output, pool_n, pool_seed)
            st.success(f"Đã tạo Pool gồm {len(pool)} ảnh.")
            st.caption(f"Đã lưu tại: {pool_dir}")
        except (OSError, ValueError) as exc:
            st.error(str(exc))

with tab_holdout:
    st.subheader("Tạo Holdout từ ảnh validation")
    val_count = len(image_names(root / "val" / "images"))
    st.caption(f"Có {val_count:,} ảnh JPG trong val/images. Mỗi chuỗi hành trình lấy tối đa một ảnh.")
    holdout_n = st.number_input("Số ảnh Holdout", 1, 2000, 50, 10)
    holdout_seed = st.number_input("Seed Holdout", 0, 2147483647, 43)
    if st.button("Tạo Holdout", type="primary"):
        try:
            holdout, holdout_dir = build_holdout(root, output, holdout_n, holdout_seed)
            st.success(f"Đã tạo Holdout gồm {len(holdout)} ảnh; nhãn gốc nằm riêng trong labels.json.")
            st.caption(f"Đã lưu tại: {holdout_dir}")
        except (OSError, ValueError) as exc:
            st.error(str(exc))

with tab_import_labels:
    st.subheader("Nhập batch và xuất nhãn tương ứng")
    st.caption("Dán đường dẫn thư mục/ZIP batch hoặc manifest RoadSift; nhãn xuất ra dùng sample ID của batch và taxonomy RoadSift.")
    batch_path = st.text_input(
        "Đường dẫn batch",
        placeholder=r"C:\duong-dan\batch hoặc C:\duong-dan\batch.zip / manifest.csv / batch-samples.json",
        key="import_batch_path",
    )
    try:
        names, mapping, detected_split = batch_input_from_path(batch_path)
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        names, mapping, detected_split = [], {}, None
        if batch_path.strip():
            st.error(str(exc))
    if detected_split:
        split = detected_split
        st.caption(f"Tự nhận nguồn annotation: {split} từ manifest RoadSift.")
    else:
        split = st.selectbox(
            "Nguồn của batch", ["train", "val"],
            format_func=lambda value: "Train / Pool" if value == "train" else "Val / Holdout",
        )
    if names:
        st.caption(f"Đã nhận {len(names)} ảnh trong batch." +
                   (f" Đã ánh xạ {len(mapping)} sample ID RoadSift." if mapping else ""))
    if st.button("Xuất nhãn JSON", type="primary", disabled=not names):
        try:
            labels, missing = export_batch_labels(root, names, mapping, split)
            if labels:
                payload = json.dumps(labels, ensure_ascii=False, indent=2).encode("utf-8")
                st.success(f"Đã tìm thấy nhãn cho {len(labels)}/{len(names)} ảnh.")
                st.download_button(
                    "Tải file nhãn JSON", payload,
                    f"bdd100k_{split}_batch_labels.json", "application/json",
                )
            else:
                st.error("Không tìm thấy nhãn nào khớp tên ảnh trong batch.")
            if missing:
                sample = ", ".join(missing[:5])
                st.warning(f"Không khớp {len(missing)} ảnh. Ví dụ: {sample}")
        except (OSError, ValueError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
            st.error(str(exc))

st.caption("Công cụ chạy cục bộ; chỉ đọc archive và ghi các file trong thư mục output.")
