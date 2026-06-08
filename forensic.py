#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║       FORENSIC IMAGE ENHANCEMENT TOOL v2.0                  ║
║       GitHub Actions Edition - Full GPU/CPU Power           ║
║       للاستخدام الجنائي الرسمي فقط                         ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import hashlib
import datetime
import argparse
import shutil
from pathlib import Path
from glob import glob

import cv2
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────
# IMAGE METADATA & FORENSIC ANALYSIS
# ──────────────────────────────────────────────
def extract_metadata(image_path):
    path = Path(image_path)
    stat = path.stat()

    img_cv = cv2.imread(str(image_path))
    img_pil = Image.open(image_path)

    with open(image_path, "rb") as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
        sha256_hash = hashlib.sha256(data).hexdigest()

    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # حدة الصورة
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # مستوى الضوضاء
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    noise = np.sum(np.abs(cv2.filter2D(gray.astype(float), -1, kernel)))
    noise = noise * (6 / (h * w))

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # EXIF
    exif_data = {}
    try:
        exif = img_pil._getexif()
        if exif:
            from PIL.ExifTags import TAGS
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[str(tag)] = str(value)[:200]
    except:
        exif_data = {"note": "لا توجد بيانات EXIF"}

    return {
        "file_name": path.name,
        "file_size_kb": round(stat.st_size / 1024, 2),
        "file_modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "md5": md5_hash,
        "sha256": sha256_hash,
        "width_px": w,
        "height_px": h,
        "resolution": f"{w}x{h}",
        "sharpness_score": round(laplacian_var, 2),
        "sharpness_level": classify_sharpness(laplacian_var),
        "noise_level": round(noise, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "exif": exif_data,
        "analysis_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def classify_sharpness(score):
    if score < 50:   return "منخفض جداً - ضبابية شديدة"
    elif score < 200: return "منخفض - يحتاج تحسيناً"
    elif score < 500: return "متوسط"
    elif score < 1000: return "جيد"
    else: return "ممتاز"

# ──────────────────────────────────────────────
# ENHANCEMENT ENGINE
# ──────────────────────────────────────────────
def enhance_image(input_path, output_dir, scale=4):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"\n{'='*60}")
    print(f"  معالجة: {input_path.name}")
    print(f"{'='*60}")

    print("[1/4] تحليل الصورة الأصلية...")
    original_meta = extract_metadata(input_path)
    print(f"  الدقة: {original_meta['resolution']}")
    print(f"  الحدة: {original_meta['sharpness_score']} ({original_meta['sharpness_level']})")

    # ── Real-ESRGAN
    print("[2/4] Real-ESRGAN - رفع الدقة...")
    enhanced_path = output_dir / f"{stem}_enhanced.png"

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32, scale=4
        )

        upsampler = RealESRGANer(
            scale=scale,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        output, _ = upsampler.enhance(img, outscale=scale)
        cv2.imwrite(str(enhanced_path), output)
        print(f"  [✓] {original_meta['resolution']} → {scale}x تم")

    except Exception as e:
        print(f"  [!] Real-ESRGAN: {e}")
        print("  [→] OpenCV Bicubic + تحسين الحواف...")
        img = cv2.imread(str(input_path))
        h, w = img.shape[:2]
        enhanced = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        cv2.imwrite(str(enhanced_path), enhanced)

    # ── GFPGAN - تحسين الوجوه
    print("[3/4] GFPGAN - تحسين الوجوه...")
    face_path = output_dir / f"{stem}_face_enhanced.png"

    try:
        from gfpgan import GFPGANer

        restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
        )

        img_in = cv2.imread(str(enhanced_path), cv2.IMREAD_COLOR)
        _, _, restored = restorer.enhance(
            img_in,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        if restored is not None:
            cv2.imwrite(str(face_path), restored)
            final_path = face_path
            print("  [✓] تم تحسين الوجوه")
        else:
            final_path = enhanced_path
            print("  [!] لم يُكتشف وجه - الاحتفاظ بنتيجة Real-ESRGAN")

    except Exception as e:
        print(f"  [!] GFPGAN: {e}")
        final_path = enhanced_path

    # ── صورة المقارنة
    print("[4/4] إنشاء صورة المقارنة...")
    comparison_path = output_dir / f"{stem}_comparison.jpg"
    create_comparison(input_path, final_path, comparison_path)

    # ── تحليل النتيجة
    enhanced_meta = extract_metadata(final_path)

    return {
        "original": original_meta,
        "enhanced": enhanced_meta,
        "files": {
            "original": str(input_path),
            "enhanced": str(final_path),
            "comparison": str(comparison_path),
        }
    }

# ──────────────────────────────────────────────
# COMPARISON IMAGE
# ──────────────────────────────────────────────
def create_comparison(orig_path, enh_path, out_path):
    img1 = cv2.imread(str(orig_path))
    img2 = cv2.imread(str(enh_path))

    target_h = min(max(img1.shape[0], img2.shape[0]), 1400)

    w1 = int(img1.shape[1] * target_h / img1.shape[0])
    w2 = int(img2.shape[1] * target_h / img2.shape[0])

    img1 = cv2.resize(img1, (w1, target_h))
    img2 = cv2.resize(img2, (w2, target_h))

    label_h = 60
    bar_color = [20, 20, 20]

    def make_label(w, text, color):
        bar = np.full((label_h, w, 3), bar_color, dtype=np.uint8)
        cv2.putText(bar, text, (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2, cv2.LINE_AA)
        return bar

    l1 = make_label(w1, "ORIGINAL", (80, 80, 255))
    l2 = make_label(w2, "ENHANCED - FORENSIC", (80, 255, 80))

    divider = np.full((target_h + label_h, 8, 3), [0, 0, 200], dtype=np.uint8)

    col1 = np.vstack([l1, img1])
    col2 = np.vstack([l2, img2])

    comparison = np.hstack([col1, divider, col2])
    cv2.imwrite(str(out_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"  [✓] المقارنة: {Path(out_path).name}")

# ──────────────────────────────────────────────
# HTML FORENSIC REPORT
# ──────────────────────────────────────────────
def generate_report(results, output_path, case_number=""):
    now = datetime.datetime.now()
    orig = results["original"]
    enh = results["enhanced"]
    files = results["files"]

    improvement = round(enh["sharpness_score"] - orig["sharpness_score"], 2)
    improvement_pct = round((improvement / max(orig["sharpness_score"], 0.1)) * 100, 1)

    exif_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in orig["exif"].items()
    ) or "<tr><td colspan='2'>لا توجد بيانات EXIF</td></tr>"

    badge = lambda s: (
        "badge-bad" if s < 50 else
        "badge-warn" if s < 200 else
        "badge-ok"
    )
    badge_text = lambda s: (
        "ضعيف" if s < 50 else
        "متوسط" if s < 200 else
        "جيد"
    )

    html = f"""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
<meta charset="UTF-8">
<title>تقرير جنائي - {orig['file_name']}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0a0a0a;--panel:#111;--border:#222;
  --red:#c0392b;--green:#27ae60;--blue:#2980b9;
  --text:#ddd;--dim:#666;--mono:'Courier New',monospace;
}}
body{{font-family:'IBM Plex Sans Arabic',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.7}}
.header{{background:#0f0f0f;border-bottom:3px solid var(--red);padding:28px 40px;display:flex;justify-content:space-between;align-items:center}}
.title{{font-size:20px;font-weight:700;color:#fff;letter-spacing:1px}}
.subtitle{{font-size:12px;color:var(--dim);margin-top:4px;font-family:var(--mono)}}
.stamp{{border:2px solid var(--red);padding:8px 18px;color:var(--red);font-weight:700;font-size:12px;letter-spacing:3px;text-align:center}}
.container{{max-width:1200px;margin:0 auto;padding:30px 40px}}
.warn{{background:#1a1000;border:1px solid #e67e22;padding:14px 18px;color:#e67e22;font-size:12px;margin-bottom:24px;border-radius:2px}}
.section{{margin-bottom:28px}}
.sec-title{{font-size:11px;font-weight:700;color:var(--red);letter-spacing:3px;font-family:var(--mono);border-bottom:1px solid var(--border);padding-bottom:8px;margin-bottom:16px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:var(--panel);border:1px solid var(--border);padding:18px}}
.card-label{{font-size:10px;color:var(--dim);letter-spacing:2px;font-family:var(--mono);margin-bottom:12px}}
.row{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1a1a1a;font-size:13px}}
.row:last-child{{border:none}}
.lbl{{color:var(--dim)}}
.val{{font-family:var(--mono);font-size:12px;color:#fff}}
.hash{{background:#080808;border:1px solid #1a1a1a;padding:10px;font-family:var(--mono);font-size:11px;color:#4ec9b0;word-break:break-all;margin-top:6px}}
.metrics{{background:#050f05;border:1px solid var(--green);padding:20px;display:flex;justify-content:space-around;text-align:center;margin-bottom:16px}}
.mv{{font-size:30px;font-weight:700;color:var(--green);font-family:var(--mono)}}
.ml{{font-size:10px;color:var(--dim);margin-top:4px;letter-spacing:1px}}
.badge{{display:inline-block;padding:2px 8px;font-size:10px;font-family:var(--mono);font-weight:700;margin-right:6px}}
.badge-ok{{background:#0a2a0a;color:#4caf50}}
.badge-warn{{background:#2a1a00;color:#ff9800}}
.badge-bad{{background:#2a0505;color:#f44336}}
.cmp-img{{width:100%;border:1px solid var(--border);display:block}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#1a1a1a;color:var(--dim);padding:10px;text-align:right;font-family:var(--mono);letter-spacing:1px}}
td{{padding:8px 10px;border-bottom:1px solid #151515;color:#bbb;font-family:var(--mono)}}
.footer{{background:#0f0f0f;border-top:1px solid var(--border);padding:16px 40px;font-size:11px;color:#444;display:flex;justify-content:space-between;font-family:var(--mono)}}
</style>
</head>
<body>
<div class="header">
  <div>
    <div class="title">⬛ FORENSIC IMAGE ANALYSIS REPORT</div>
    <div class="title" style="font-size:15px;color:#888;margin-top:4px">تقرير التحليل الجنائي للصور</div>
    <div class="subtitle">
      التاريخ: {now.strftime("%Y-%m-%d")} &nbsp;|&nbsp;
      الوقت: {now.strftime("%H:%M:%S")} &nbsp;|&nbsp;
      القضية: {case_number or 'غير محدد'} &nbsp;|&nbsp;
      الإصدار: v2.0 GitHub Actions
    </div>
  </div>
  <div class="stamp">سري<br>CONFIDENTIAL</div>
</div>

<div class="container">

<div class="warn">
⚠ تنبيه جنائي: الصور المحسنة بالذكاء الاصطناعي أداة استدلالية لتوجيه التحقيق.
لا تُقدَّم وحدها دليلاً قانونياً نهائياً دون دعمها بأدلة مستقلة.
</div>

<div class="section">
  <div class="sec-title">// هوية الملف / FILE IDENTITY</div>
  <div class="grid2">
    <div class="card">
      <div class="card-label">ORIGINAL FILE</div>
      <div class="row"><span class="lbl">اسم الملف</span><span class="val">{orig['file_name']}</span></div>
      <div class="row"><span class="lbl">الحجم</span><span class="val">{orig['file_size_kb']} KB</span></div>
      <div class="row"><span class="lbl">الدقة</span><span class="val">{orig['resolution']}</span></div>
      <div class="row"><span class="lbl">آخر تعديل</span><span class="val">{orig['file_modified']}</span></div>
      <div class="row"><span class="lbl">MD5</span></div>
      <div class="hash">{orig['md5']}</div>
      <div class="row" style="margin-top:8px"><span class="lbl">SHA-256</span></div>
      <div class="hash">{orig['sha256']}</div>
    </div>
    <div class="card">
      <div class="card-label">ENHANCED FILE</div>
      <div class="row"><span class="lbl">اسم الملف</span><span class="val">{enh['file_name']}</span></div>
      <div class="row"><span class="lbl">الحجم</span><span class="val">{enh['file_size_kb']} KB</span></div>
      <div class="row"><span class="lbl">الدقة</span><span class="val">{enh['resolution']}</span></div>
      <div class="row"><span class="lbl">وقت المعالجة</span><span class="val">{enh['analysis_time']}</span></div>
      <div class="row"><span class="lbl">MD5</span></div>
      <div class="hash">{enh['md5']}</div>
      <div class="row" style="margin-top:8px"><span class="lbl">SHA-256</span></div>
      <div class="hash">{enh['sha256']}</div>
    </div>
  </div>
</div>

<div class="section">
  <div class="sec-title">// تحليل الجودة / QUALITY ANALYSIS</div>
  <div class="metrics">
    <div><div class="mv">{orig['sharpness_score']}</div><div class="ml">حدة الأصل</div></div>
    <div><div class="mv" style="color:var(--blue)">→</div></div>
    <div><div class="mv">{enh['sharpness_score']}</div><div class="ml">حدة المحسنة</div></div>
    <div><div class="mv">+{improvement_pct}%</div><div class="ml">نسبة التحسين</div></div>
    <div><div class="mv" style="font-size:16px">{orig['resolution']}<br>↓<br>{enh['resolution']}</div><div class="ml">الدقة</div></div>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="card-label">ORIGINAL METRICS</div>
      <div class="row">
        <span class="lbl">مستوى الحدة</span>
        <span class="val"><span class="badge {badge(orig['sharpness_score'])}">{badge_text(orig['sharpness_score'])}</span>{orig['sharpness_level']}</span>
      </div>
      <div class="row"><span class="lbl">الضوضاء</span><span class="val">{orig['noise_level']}</span></div>
      <div class="row"><span class="lbl">السطوع</span><span class="val">{orig['brightness']} / 255</span></div>
      <div class="row"><span class="lbl">التباين</span><span class="val">{orig['contrast']}</span></div>
    </div>
    <div class="card">
      <div class="card-label">ENHANCED METRICS</div>
      <div class="row">
        <span class="lbl">مستوى الحدة</span>
        <span class="val"><span class="badge {badge(enh['sharpness_score'])}">{badge_text(enh['sharpness_score'])}</span>{enh['sharpness_level']}</span>
      </div>
      <div class="row"><span class="lbl">الضوضاء</span><span class="val">{enh['noise_level']}</span></div>
      <div class="row"><span class="lbl">السطوع</span><span class="val">{enh['brightness']} / 255</span></div>
      <div class="row"><span class="lbl">التباين</span><span class="val">{enh['contrast']}</span></div>
    </div>
  </div>
</div>

<div class="section">
  <div class="sec-title">// مقارنة بصرية / VISUAL COMPARISON</div>
  <img class="cmp-img" src="{Path(files['comparison']).name}" alt="مقارنة">
</div>

<div class="section">
  <div class="sec-title">// بيانات EXIF</div>
  <div class="card">
    <table>
      <tr><th>الحقل</th><th>القيمة</th></tr>
      {exif_rows}
    </table>
  </div>
</div>

</div>
<div class="footer">
  <span>Forensic Image Enhancement Tool v2.0 - GitHub Actions Edition</span>
  <span>Generated: {now.strftime("%Y-%m-%d %H:%M:%S")}</span>
  <span>للاستخدام الجنائي الرسمي فقط - OFFICIAL USE ONLY</span>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [✓] التقرير: {Path(output_path).name}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Forensic Image Enhancement Tool v2.0")
    parser.add_argument("--input",  "-i", default="./input",  help="مجلد الصور")
    parser.add_argument("--output", "-o", default="./output", help="مجلد النتائج")
    parser.add_argument("--case",   "-c", default="",         help="رقم القضية")
    parser.add_argument("--scale",  "-s", type=int, default=4, choices=[2, 4])
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║     FORENSIC IMAGE ENHANCEMENT TOOL v2.0                    ║
║     GitHub Actions Edition                                  ║
╚══════════════════════════════════════════════════════════════╝""")

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    images = []
    for ext in extensions:
        images.extend(glob(str(input_dir / ext)))
        images.extend(glob(str(input_dir / ext.upper())))

    if not images:
        print(f"[✗] لا توجد صور في: {input_dir.absolute()}")
        sys.exit(1)

    print(f"\n[→] صور مكتشفة: {len(images)}")
    print(f"[→] النتائج في: {output_dir.absolute()}")
    print(f"[→] معامل التكبير: {args.scale}x")

    success = 0
    for img_path in images:
        try:
            img_path = Path(img_path)
            img_out  = output_dir / img_path.stem
            img_out.mkdir(exist_ok=True)

            shutil.copy(img_path, img_out / img_path.name)

            results = enhance_image(img_path, img_out, scale=args.scale)

            report_path = img_out / f"{img_path.stem}_report.html"
            generate_report(results, report_path, case_number=args.case)

            json_path = img_out / f"{img_path.stem}_data.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"\n[✓] {img_path.name} → {img_out.name}/")
            success += 1

        except Exception as e:
            print(f"\n[✗] خطأ: {img_path} — {e}")

    print(f"\n{'='*60}")
    print(f"  اكتمل: {success}/{len(images)} صورة")
    print(f"  النتائج: {output_dir.absolute()}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
