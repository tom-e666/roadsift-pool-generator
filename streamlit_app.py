from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ijson
import streamlit as st
from PIL import Image, ImageDraw

ARCHIVE = Path.home() / "Downloads" / "archive"
OUTPUT = Path.home() / "Downloads" / "poools_generated"

BDD100K_CANONICAL_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
]

# Map archive categories (person, bike, motor) to RoadSift canonical classes
CANONICAL_P021_MAP = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle",
}

# Reverse map for pools configured with person, bike, motor
LEGACY_ALIAS_MAP = {
    "pedestrian": "person",
    "bicycle": "bike",
    "motorcycle": "motor",
}

CLASS_COLORS = {
    "car": "#23b8e4", "pedestrian": "#00ff88", "person": "#00ff88",
    "truck": "#f59e0b", "bus": "#ec4899", "rider": "#a855f7",
    "bike": "#10b981", "bicycle": "#10b981", "motor": "#ef4444", "motorcycle": "#ef4444",
    "traffic light": "#eab308", "traffic sign": "#06b6d4", "train": "#6366f1",
}


def draw_boxes(img_path: Path, labels: list[dict], class_map: dict[str, str] | None = None) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    class_map = class_map or {}

    for label in labels:
        box = label.get("box2d")
        if not box:
            continue
        raw_cat = str(label.get("category", ""))
        cat = class_map.get(raw_cat, raw_cat)
        color = CLASS_COLORS.get(cat, "#ffffff")
        try:
            x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 3, max(2, y1 - 12)), cat, fill=color)
        except (KeyError, ValueError, TypeError):
            continue
    return img


def render_image(image, caption: str | None = None) -> None:
    try:
        st.image(image, width="stretch", caption=caption)
    except (TypeError, ValueError):
        st.image(image, use_container_width=True, caption=caption)


def render_dataframe(data, height: int | None = None) -> None:
    try:
        st.dataframe(data, width="stretch", height=height)
    except (TypeError, ValueError):
        st.dataframe(data, use_container_width=True, height=height)


def locate_image_file(image_name: str, root: Path, output: Path) -> Path | None:
    for sub in ("train", "val"):
        p = root / sub / "images" / image_name
        if p.is_file():
            return p
    for p in output.glob(f"**/images/{image_name}"):
        if p.is_file():
            return p
    return None


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def image_names(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})


def sample_by_index(names: list[str], count: int, seed: int) -> list[tuple[int, str]]:
    indexes = sorted(random.Random(seed).sample(range(1, len(names) + 1), count))
    return [(index, names[index - 1]) for index in indexes]


def sequence(name: str) -> str:
    return Path(name).stem.split("-", 1)[0]


def records_for(json_path: Path, wanted: set[str]):
    pending = set(wanted)
    if not pending or not json_path.exists():
        return

    # Use fast json.load for smaller annotation files (like val 208MB)
    if json_path.stat().st_size < 350 * 1024 * 1024:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data:
                name = row.get("name")
                if name in pending:
                    yield row
                    pending.remove(name)
                    if not pending:
                        return
            return
        except (MemoryError, json.JSONDecodeError):
            pass

    # Stream large files with ijson
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
        return {row.get("sequence_id") for row in csv.DictReader(f) if row.get("sequence_id")}


def write_manifest(path: Path, indexed_data: list[tuple[int, str, str]], split: str):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "sample_id", "filename", "source_name", "sequence_id", "source_split", "content_sha256"])
        writer.writerows(
            (i, Path(name).stem, name, name, sequence(name), split, sha)
            for i, name, sha in indexed_data
        )


def generated_sequences(output: Path, kind: str) -> set[str]:
    sequences = set()
    for manifest in output.glob(f"{kind}*/manifest.csv"):
        sequences.update(manifest_sequences(manifest))
    return sequences


def load_known_manifest_index(output_dir: Path) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    if not output_dir.exists():
        return index
    for manifest_path in output_dir.glob("**/manifest.csv"):
        try:
            with manifest_path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sha = (row.get("content_sha256") or "").strip().lower()
                    source_name = (row.get("source_name") or row.get("filename") or "").strip()
                    split = (row.get("source_split") or "train").strip().lower()
                    if sha and source_name:
                        index[sha] = (source_name, split)
        except Exception:
            continue
    return index


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

    indexed_data = []
    for index, name in selected:
        dst = images_dir / name
        shutil.copy2(train_dir / name, dst)
        sha = compute_sha256(dst)
        indexed_data.append((index, name, sha))

    write_manifest(pool_dir / "manifest.csv", indexed_data, "train")
    return [name for _, name, _ in indexed_data], pool_dir


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

    indexed_data = []
    for index, name in selected:
        dst = images_dir / name
        shutil.copy2(val_dir / name, dst)
        sha = compute_sha256(dst)
        indexed_data.append((index, name, sha))

    write_manifest(holdout_dir / "manifest.csv", indexed_data, "val")

    # Save sha256 list for direct RoadSift holdout registration
    sha_list = [sha for _, _, sha in indexed_data]
    with (holdout_dir / "content_sha256.json").open("w", encoding="utf-8") as f:
        json.dump(sha_list, f, indent=2)

    label_path = root / "val" / "annotations" / "bdd100k_labels_images_val.json"
    labels = list(records_for(label_path, {name for _, name, _ in indexed_data}))
    with (holdout_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)

    return [name for _, name, _ in indexed_data], holdout_dir


def parse_batch_mapping(text: str) -> tuple[dict[str, str], str | None]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("File manifest CSV trống.")

    fieldnames = [f.strip() for f in (rows[0].keys() if rows else [])]
    # Identify sample_id and source_name columns
    sample_id_candidates = ["sample_id", "id", "sampleId", "sample"]
    source_name_candidates = ["source_name", "filename", "file_name", "sourceName", "name"]
    split_candidates = ["source_split", "split", "sourceSplit"]

    sid_col = next((c for c in sample_id_candidates if c in fieldnames), None)
    src_col = next((c for c in source_name_candidates if c in fieldnames), None)
    split_col = next((c for c in split_candidates if c in fieldnames), None)

    if not sid_col or not src_col:
        raise ValueError("Manifest batch cần có cột sample_id và source_name (hoặc filename).")

    mapping = {}
    splits = set()
    for row in rows:
        source_name = (row.get(src_col) or "").strip()
        sample_id = (row.get(sid_col) or "").strip()
        if not source_name or not sample_id:
            continue
        mapping[source_name] = sample_id
        if split_col:
            split = (row.get(split_col) or "").strip().lower()
            if split:
                splits.add(split)

    if not mapping:
        raise ValueError("Manifest batch không có dòng ánh xạ hợp lệ.")
    detected_split = next(iter(splits), None) if len(splits) == 1 else None
    return mapping, detected_split


def _json_batch_mapping(text: str, output_dir: Path | None = None) -> tuple[dict[str, str], str | None]:
    document = json.loads(text)
    items = document if isinstance(document, list) else document.get("items", [])
    if isinstance(document, dict) and not items and "samples" in document:
        items = document.get("samples", [])

    mapping = {}
    splits = set()
    sha_needed = []

    for row in items:
        if not isinstance(row, dict):
            continue
        sample = row.get("sample", row) if isinstance(row.get("sample"), dict) else row
        attributes = sample.get("attributes") or {}
        sample_id = sample.get("id") or row.get("sample_id")

        source_name = (
            attributes.get("source_name")
            or attributes.get("filename")
            or row.get("source_name")
            or row.get("filename")
            or sample.get("source_name")
        )

        sha = row.get("sha256") or sample.get("content_sha256") or sample.get("sha256")

        if source_name and sample_id:
            mapping[str(source_name)] = str(sample_id)
            split = attributes.get("source_split") or row.get("source_split") or sample.get("source_split")
            if split:
                splits.add(str(split).strip().lower())
        elif sha and sample_id:
            sha_needed.append((str(sha).lower(), str(sample_id)))

    # Resolve missing source_names using sha256 index from output manifests
    if sha_needed and output_dir:
        known_index = load_known_manifest_index(output_dir)
        for sha, sample_id in sha_needed:
            if sha in known_index:
                src_name, split = known_index[sha]
                mapping[src_name] = sample_id
                if split:
                    splits.add(split)

    if not mapping:
        raise ValueError("JSON batch không tìm thấy ánh xạ source_name -> sample_id tương ứng.")
    detected_split = next(iter(splits), None) if len(splits) == 1 else None
    return mapping, detected_split


def batch_input_from_path(raw_path: str, output_dir: Path | None = None) -> tuple[list[str], dict[str, str], str | None]:
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
        manifest_csv = path / "manifest.csv"
        if manifest_csv.is_file():
            mapping, detected_split = parse_batch_mapping(manifest_csv.read_text(encoding="utf-8-sig"))
        else:
            json_candidates = list(path.glob("*.json"))
            if json_candidates:
                mapping, detected_split = _json_batch_mapping(json_candidates[0].read_text(encoding="utf-8-sig"), output_dir)
    elif path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
        image_names = [path.name]
    elif path.is_file() and path.suffix.lower() == ".csv":
        mapping, detected_split = parse_batch_mapping(path.read_text(encoding="utf-8-sig"))
    elif path.is_file() and path.suffix.lower() == ".json":
        mapping, detected_split = _json_batch_mapping(path.read_text(encoding="utf-8-sig"), output_dir)
    elif path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            image_names = [Path(m.replace("\\", "/")).name for m in members
                           if Path(m.replace("\\", "/")).suffix.lower() in {".jpg", ".jpeg"}]
            manifests = [m for m in members if Path(m.replace("\\", "/")).name.lower() == "manifest.csv"]
            json_manifests = [m for m in members if Path(m.replace("\\", "/")).suffix.lower() == ".json"]

            if manifests:
                mapping, detected_split = parse_batch_mapping(archive.read(manifests[0]).decode("utf-8-sig"))
            elif json_manifests:
                mapping, detected_split = _json_batch_mapping(archive.read(json_manifests[0]).decode("utf-8-sig"), output_dir)
    else:
        raise ValueError("Đường dẫn cần là thư mục batch, tệp ZIP, CSV manifest hoặc JSON batch.")

    if len(image_names) != len(set(image_names)):
        raise ValueError("Batch có tên ảnh trùng nhau; cần giữ tên nguồn duy nhất.")

    if mapping and image_names:
        image_names_set = set(image_names)
        image_stems = {Path(name).stem for name in image_names}
        mapping = {
            src: sid for src, sid in mapping.items()
            if src in image_names_set or Path(src).stem in image_stems or sid in image_stems or f"{sid}.jpg" in image_names_set
        }

    names = sorted(mapping) if mapping else sorted(image_names)
    if not names:
        raise ValueError("Không tìm thấy ảnh hoặc danh sách sample hợp lệ trong batch.")
    return names, mapping, detected_split


def export_batch_labels(
    root: Path,
    names: list[str],
    mapping: dict[str, str],
    split: str,
    class_map: dict[str, str] | None = None,
    target_taxonomy: list[str] | None = None,
    filter_to_taxonomy: bool = True,
) -> tuple[list[dict], list[str]]:
    annotation_path = root / split / "annotations" / f"bdd100k_labels_images_{split}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhãn BDD100K: {annotation_path}")

    class_map = class_map or {}
    allowed_classes = set(target_taxonomy) if (filter_to_taxonomy and target_taxonomy) else None

    records = {record["name"]: record for record in records_for(annotation_path, set(names))}
    exported, missing = [], []

    for name in names:
        record = records.get(name)
        if record is None:
            missing.append(name)
            continue

        clean_labels = []
        for label in record.get("labels") or []:
            if not isinstance(label, dict):
                continue
            box2d = label.get("box2d")
            if not box2d:
                continue
            category = str(label.get("category", ""))
            mapped_cat = class_map.get(category, category)
            if allowed_classes is not None and mapped_cat not in allowed_classes:
                continue

            clean_label = dict(label)
            clean_label["category"] = mapped_cat
            clean_labels.append(clean_label)

        output_record = dict(record)
        # Use RoadSift sample ID as frame name
        output_record["name"] = mapping.get(name, Path(name).stem)
        output_record["labels"] = clean_labels
        exported.append(output_record)

    return exported, missing


st.set_page_config(
    page_title="BDD100K Pool Builder · RoadSift",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at top, #11263a, #0b1320 58%, #080d16);}
.block-container {max-width: 1200px; padding-top: 1.5rem;}
.stButton>button {
    font-weight: 600;
    border-radius: 6px;
    height: 2.85rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR: CẤU HÌNH HỆ THỐNG -----------------
with st.sidebar:
    st.title("⚙️ Cấu hình thư mục")
    st.caption("RoadSift P-021 RAV-05")
    st.divider()

    archive_input = st.text_input(
        "Thư mục Archive gốc",
        value=str(ARCHIVE),
        help="Đường dẫn chứa thư mục `train/images`, `val/images` và annotations.",
    )
    output_input = st.text_input(
        "Thư mục Output",
        value=str(OUTPUT),
        help="Dữ liệu sinh ra sẽ lưu theo format: pool_<timestamp> hoặc holdout_<timestamp>",
    )

    root = Path(archive_input).expanduser()
    output = Path(output_input).expanduser()

    archive_ok = root.is_dir()
    if archive_ok:
        st.success("✓ Đường dẫn Archive hợp lệ", icon="✅")
    else:
        st.error("✗ Không tìm thấy thư mục Archive", icon="⚠️")

    st.caption(f"📁 Output đích: `{output}`")

    st.divider()
    st.markdown("### 📊 Dataset Info")
    train_dir = root / "train" / "images"
    val_dir = root / "val" / "images"

    n_train = len(image_names(train_dir)) if archive_ok and train_dir.is_dir() else 0
    n_val = len(image_names(val_dir)) if archive_ok and val_dir.is_dir() else 0

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Train images", f"{n_train:,}" if n_train else "0")
    with col_m2:
        st.metric("Val images", f"{n_val:,}" if n_val else "0")

    st.caption("🔒 Checksum: SHA-256 enabled")
    st.caption("🛡️ Anti-leakage: Sequence exclusion active")

# ----------------- MAIN VIEW -----------------
st.title("🚗 BDD100K Pool Builder")
st.caption("Công cụ chuẩn bị dữ liệu, phân tầng theo sequence và xuất nhãn cho RoadSift (P-021).")

tabs = st.tabs([
    "🎯 1. Tạo Pool (Train)",
    "🧪 2. Tạo Holdout (Val)",
    "🏷️ 3. Gán nhãn batch (Handoff)",
    "👁️ 4. Xem trước (Preview & Metadata)",
])

# TAB 1: TẠO POOL
with tabs[0]:
    st.subheader("Trích xuất Pool từ tập Train")
    st.info("Manifest sinh ra sẽ đính kèm mã băm SHA-256 cho từng file và tự động loại trừ các sequence trong Holdout để ngăn data leakage.", icon="ℹ️")

    with st.container(border=True):
        st.markdown("##### ⚙️ Tham số lấy mẫu")
        col1, col2 = st.columns(2)
        with col1:
            pool_n = st.number_input(
                "Số lượng ảnh cần trích xuất",
                min_value=1,
                max_value=max(10, n_train) if n_train else 70000,
                value=min(200, n_train) if n_train else 200,
                step=50,
                help="Kích thước pool dữ liệu đưa vào labeling loop.",
            )
        with col2:
            pool_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=2147483647,
                value=42,
                step=1,
                help="Cố định seed để tái lập chính xác tập mẫu.",
            )

        submit_pool = st.button("🚀 Khởi tạo Pool ngay", type="primary")

    if submit_pool:
        if not archive_ok:
            st.error("Không thể chạy: Thư mục nguồn Archive không tồn tại. Vui lòng kiểm tra lại ở thanh bên (Sidebar).")
        else:
            try:
                with st.status("Đang xử lý trích xuất Pool...", expanded=True) as status:
                    st.write("1. Quét danh sách ảnh train và kiểm tra sequence...")
                    st.write("2. Lọc bỏ các sequence đã xuất hiện trong tập Holdout...")
                    st.write(f"3. Lấy mẫu ngẫu nhiên {pool_n} ảnh (seed={pool_seed})...")
                    st.write("4. Sao chép ảnh, tính toán checksum SHA-256 và ghi manifest.csv...")
                    pool, pool_dir = build_pool(root, output, pool_n, pool_seed)
                    status.update(label=f"✓ Đã tạo thành công Pool gồm {len(pool)} ảnh!", state="complete", expanded=False)

                with st.container(border=True):
                    st.success("Dữ liệu Pool đã sẵn sàng:")
                    st.code(str(pool_dir), language="bash")
                    st.markdown("""
                    **💡 Bước tiếp theo trên RoadSift:**
                    1. Mở RoadSift > **Import dữ liệu** (`/import`).
                    2. Tạo pool mới, chọn tải lên thư mục ảnh `images/` và tệp `manifest.csv`.
                    """)
            except (OSError, ValueError) as exc:
                st.error(f"Lỗi khi tạo Pool: {exc}")

# TAB 2: TẠO HOLDOUT
with tabs[1]:
    st.subheader("Trích xuất Holdout Set từ tập Val")
    st.info("Mỗi sequence hành trình chỉ lấy tối đa 1 frame và loại trừ các sequence đã xuất hiện trong Pool. Tệp `content_sha256.json` dùng để đăng ký holdout với RoadSift.", icon="ℹ️")

    with st.container(border=True):
        st.markdown("##### ⚙️ Tham số lấy mẫu Holdout")
        col1, col2 = st.columns(2)
        with col1:
            holdout_n = st.number_input(
                "Số lượng ảnh Holdout",
                min_value=1,
                max_value=max(10, n_val) if n_val else 10000,
                value=min(50, n_val) if n_val else 50,
                step=10,
                help="Kích thước tập Holdout để benchmark và tính metrics mAP.",
            )
        with col2:
            holdout_seed = st.number_input(
                "Random Seed Holdout",
                min_value=0,
                max_value=2147483647,
                value=43,
                step=1,
            )

        submit_holdout = st.button("🧪 Khởi tạo Holdout ngay", type="primary")

    if submit_holdout:
        if not archive_ok:
            st.error("Không thể chạy: Thư mục nguồn Archive không tồn tại. Vui lòng kiểm tra lại ở thanh bên.")
        else:
            try:
                with st.status("Đang xử lý trích xuất Holdout...", expanded=True) as status:
                    st.write("1. Quét danh sách ảnh val/images và gom nhóm theo sequence...")
                    st.write("2. Loại trừ các sequence đã xuất hiện trong các Pool trước...")
                    st.write(f"3. Lấy mẫu phân tầng {holdout_n} ảnh (1 ảnh/sequence, seed={holdout_seed})...")
                    st.write("4. Sao chép ảnh, tính SHA-256, trích xuất ground-truth labels.json...")
                    holdout, holdout_dir = build_holdout(root, output, holdout_n, holdout_seed)
                    status.update(label=f"✓ Đã tạo thành công Holdout gồm {len(holdout)} ảnh!", state="complete", expanded=False)

                with st.container(border=True):
                    st.success("Dữ liệu Holdout đã lưu tại:")
                    st.code(str(holdout_dir), language="bash")
                    sha_file = holdout_dir / "content_sha256.json"
                    if sha_file.exists():
                        sha_data = sha_file.read_text(encoding="utf-8")
                        st.markdown("**📋 Danh sách SHA-256 để đăng ký Holdout trên RoadSift:**")
                        st.code(sha_data, language="json")
                        st.caption("Dùng danh sách này nộp vào `POST /api/v2/pools/{id}/holdouts`.")
            except (OSError, ValueError) as exc:
                st.error(f"Lỗi khi tạo Holdout: {exc}")

# TAB 3: LABELING HANDOFF
with tabs[2]:
    st.subheader("Nhập batch và xuất nhãn cho RoadSift")
    st.info("Nhận diện batch xuất từ RoadSift (JSON sample, handoff manifest, CSV hoặc ZIP), tự động chuẩn hóa taxonomy và xuất nhãn BDD100K.", icon="ℹ️")

    with st.container(border=True):
        st.markdown("##### 📁 Thông tin Batch từ RoadSift")
        batch_path = st.text_input(
            "Đường dẫn file hoặc thư mục Batch",
            placeholder=r"C:\duong-dan\batch hoặc C:\duong-dan\handoff_manifest.json / batch-samples.json / batch.zip",
            key="import_batch_path",
        )

        names, mapping, detected_split = [], {}, None
        if batch_path.strip():
            try:
                names, mapping, detected_split = batch_input_from_path(batch_path, output)
            except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
                st.error(str(exc))

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            split = detected_split or st.selectbox(
                "Nguồn annotation gốc", ["train", "val"],
                format_func=lambda value: "Train (Pool)" if value == "train" else "Val (Holdout)",
            )
        with col_s2:
            taxonomy_mode = st.selectbox(
                "Chế độ Taxonomy",
                [
                    "Canonical RoadSift P-021 (pedestrian, bicycle, motorcycle)",
                    "Pool dùng person, bike, motor (nhãn gốc BDD100K)",
                    "Giữ nguyên không lọc nhãn",
                ],
                index=0,
                help="P-021 chuẩn dùng 10 lớp canonical (pedestrian, bicycle, motorcycle). Archive BDD100K dùng person/bike/motor sẽ được tự động chuẩn hóa.",
            )

        if taxonomy_mode.startswith("Canonical RoadSift P-021"):
            active_class_map = CANONICAL_P021_MAP
            active_target_taxonomy = BDD100K_CANONICAL_CLASSES
        elif taxonomy_mode.startswith("Pool dùng person"):
            active_class_map = {}
            active_target_taxonomy = ["person", "bike", "motor", "car", "truck", "bus", "train", "rider", "traffic light", "traffic sign"]
        else:
            active_class_map = {}
            active_target_taxonomy = None

        filter_taxonomy = st.checkbox(
            "Lọc bỏ các nhãn không nằm trong taxonomy đích (tránh lỗi unknown_class khi nhập vào RoadSift)",
            value=True,
        )

        if names:
            st.success(f"✓ Đã nhận diện {len(names)} ảnh. Khớp được {len(mapping)} sample ID RoadSift.")
            with st.expander("🔍 Xem trước ảnh & nhãn của Batch", expanded=False):
                preview_img_name = st.selectbox(
                    "Chọn ảnh xem trước",
                    names,
                    format_func=lambda n: f"{n} (RoadSift ID: {mapping.get(n, 'N/A')})",
                    key="tab3_preview_img",
                )
                img_path = locate_image_file(preview_img_name, root, output)
                if img_path and img_path.is_file():
                    annotation_path = root / split / "annotations" / f"bdd100k_labels_images_{split}.json"
                    matched_records = list(records_for(annotation_path, {preview_img_name}))
                    raw_labels = (matched_records[0].get("labels") or []) if matched_records else []
                    valid_boxes = [l for l in raw_labels if l.get("box2d")]
                    
                    show_b = st.checkbox("Hiển thị Bounding Box", value=True, key="tab3_show_b")
                    cp1, cp2 = st.columns([3, 2])
                    with cp1:
                        if show_b and valid_boxes:
                            drawn = draw_boxes(img_path, valid_boxes, class_map=active_class_map)
                            render_image(drawn, caption=f"{preview_img_name} ({len(valid_boxes)} boxes)")
                        else:
                            render_image(str(img_path), caption=preview_img_name)
                    with cp2:
                        st.markdown(f"**RoadSift Sample ID:** `{mapping.get(preview_img_name, 'N/A')}`")
                        st.markdown(f"**Sequence ID:** `{sequence(preview_img_name)}`")
                        st.markdown(f"**Số box nhãn:** {len(valid_boxes)}")
                        if valid_boxes:
                            box_data = [
                                {
                                    "Lớp": active_class_map.get(b.get("category"), b.get("category")),
                                    "x1": round(float(b["box2d"]["x1"]), 1),
                                    "y1": round(float(b["box2d"]["y1"]), 1),
                                    "x2": round(float(b["box2d"]["x2"]), 1),
                                    "y2": round(float(b["box2d"]["y2"]), 1),
                                }
                                for b in valid_boxes[:20]
                            ]
                            render_dataframe(box_data, height=220)
                else:
                    st.info(f"Không tìm thấy file ảnh `{preview_img_name}` cục bộ trong thư mục archive hoặc output.")

        submit_labels = st.button("🏷️ Xuất nhãn JSON cho Batch", type="primary", disabled=not names)

    if submit_labels and names:
        try:
            with st.status("Đang tìm và chuẩn hóa nhãn ground-truth...", expanded=True) as status:
                st.write(f"1. Quét annotation từ bdd100k_labels_images_{split}.json...")
                st.write("2. Ánh xạ danh mục sang taxonomy chuẩn P-021...")
                st.write("3. Đổi tên frame thành RoadSift sample_id...")
                labels, missing = export_batch_labels(
                    root, names, mapping, split,
                    class_map=active_class_map,
                    target_taxonomy=active_target_taxonomy,
                    filter_to_taxonomy=filter_taxonomy,
                )
                total_boxes = sum(len(f.get("labels", [])) for f in labels)
                status.update(label=f"✓ Đã xuất nhãn thành công ({len(labels)} ảnh, {total_boxes} boxes)!", state="complete", expanded=False)

            if labels:
                payload = json.dumps(labels, ensure_ascii=False, indent=2).encode("utf-8")
                with st.container(border=True):
                    st.success(f"Đã tạo file nhãn cho {len(labels)}/{len(names)} ảnh ({total_boxes} bounding boxes).")
                    st.download_button(
                        "📥 Tải file nhãn JSON (RoadSift BDD100K)",
                        payload,
                        f"bdd100k_{split}_batch_labels.json",
                        "application/json",
                    )
                    st.markdown("💡 **Bước tiếp theo:** Vào RoadSift > **Nhập annotation**, chọn batch và tải file JSON này lên.")
            else:
                st.error("Không tìm thấy nhãn nào khớp với ảnh trong batch.")

            if missing:
                st.warning(f"Không tìm thấy nhãn cho {len(missing)} ảnh: {', '.join(missing[:5])}")
        except (OSError, ValueError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
            st.error(str(exc))

# TAB 4: PREVIEW & METADATA
with tabs[3]:
    st.subheader("Xem trước ảnh, Metadata và Nhãn BDD100K")
    st.caption("Tra cứu trực quan bất kỳ khung hình nào kèm bounding boxes, thông số thời tiết và thuộc tính camera.")

    with st.container(border=True):
        col_src1, col_src2 = st.columns([1, 2])
        with col_src1:
            pv_source = st.radio("Nguồn ảnh", ["Val (Archive)", "Train (Archive)", "Thư mục output đã tạo"], index=0)
            if pv_source.startswith("Val"):
                pv_folder = root / "val" / "images"
                pv_split = "val"
            elif pv_source.startswith("Train"):
                pv_folder = root / "train" / "images"
                pv_split = "train"
            else:
                pv_folder = output
                pv_split = "train"

            if pv_source.startswith("Thư mục output"):
                pv_img_list = sorted(p.name for p in output.glob("**/images/*.jpg"))
            else:
                pv_img_list = image_names(pv_folder)

        with col_src2:
            if not pv_img_list:
                st.info("Chưa có ảnh nào trong nguồn này.")
                pv_chosen_img = None
            else:
                pv_chosen_img = st.selectbox("Chọn ảnh trong danh sách", pv_img_list[:500], index=0)
                pv_custom_name = st.text_input("Hoặc nhập tên tệp ảnh (vd: b1c66a42-6f7d68ca.jpg):", "")
                if pv_custom_name.strip():
                    pv_chosen_img = pv_custom_name.strip()

    if pv_chosen_img:
        found_file = locate_image_file(pv_chosen_img, root, output)
        if found_file and found_file.is_file():
            ann_file = root / pv_split / "annotations" / f"bdd100k_labels_images_{pv_split}.json"
            recs = list(records_for(ann_file, {pv_chosen_img}))
            attr = recs[0].get("attributes") or {} if recs else {}
            all_boxes = [l for l in (recs[0].get("labels") or []) if l.get("box2d")] if recs else []

            with st.container(border=True):
                col_view, col_meta = st.columns([3, 2])
                with col_view:
                    toggle_b = st.checkbox("Hiển thị Bounding Box", value=True, key="pv_tab_boxes")
                    if toggle_b and all_boxes:
                        rendered = draw_boxes(found_file, all_boxes)
                        render_image(rendered, caption=f"{pv_chosen_img} ({len(all_boxes)} boxes)")
                    else:
                        render_image(str(found_file), caption=pv_chosen_img)

                with col_meta:
                    st.markdown("#### ℹ️ Metadata")
                    st.markdown(f"- **Tên file:** `{pv_chosen_img}`")
                    st.markdown(f"- **Sequence:** `{sequence(pv_chosen_img)}`")
                    st.markdown(f"- **Thời tiết (weather):** `{attr.get('weather', 'unknown')}`")
                    st.markdown(f"- **Thời điểm (timeofday):** `{attr.get('timeofday', 'unknown')}`")
                    st.markdown(f"- **Bối cảnh (scene):** `{attr.get('scene', 'unknown')}`")
                    st.markdown(f"- **Dung lượng:** `{found_file.stat().st_size:,} bytes`")
                    file_sha = compute_sha256(found_file)
                    st.markdown(f"- **SHA-256:** `{file_sha[:16]}...`")

                    st.markdown(f"#### 🏷️ Bounding Boxes ({len(all_boxes)})")
                    if all_boxes:
                        table_rows = [
                            {
                                "Lớp": b.get("category"),
                                "x1": round(float(b["box2d"]["x1"]), 1),
                                "y1": round(float(b["box2d"]["y1"]), 1),
                                "x2": round(float(b["box2d"]["x2"]), 1),
                                "y2": round(float(b["box2d"]["y2"]), 1),
                            }
                            for b in all_boxes
                        ]
                        render_dataframe(table_rows, height=240)
                    else:
                        st.caption("Ảnh không có nhãn bounding box (explicit negative).")
        else:
            st.warning(f"Không tìm thấy file ảnh `{pv_chosen_img}` trên ổ đĩa.")

