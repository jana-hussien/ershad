#!/usr/bin/env python3
"""Build the Ershad / DareDevil workflow figure for the IEEE paper."""

from __future__ import annotations

import base64
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "figures" / "workflow_assets"
OUT_SVG = ROOT / "figures" / "ershad_workflow.svg"
NB_SAMPLES = ROOT / "assets" / "nb_images"
POSTER_RIG = ROOT / "assets" / "poster_rig.png"


def ensure_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    person = Image.open(NB_SAMPLES / "img_1.png")
    outdoor = Image.open(NB_SAMPLES / "img_0.png")
    rig = Image.open(POSTER_RIG)

    # sample panels are 4 equal-width subplots in the notebook export
    def split_panels(img: Image.Image, prefix: str) -> None:
        w, h = img.size
        panel_w = w // 4
        rgb = img.crop((0, 0, panel_w, h))
        thermal = img.crop((panel_w, 0, w, h))
        rgb.save(ASSETS / f"{prefix}_rgb.png")
        thermal.save(ASSETS / f"{prefix}_thermal_trio.png")

    split_panels(person, "person")
    split_panels(outdoor, "outdoor")

    # assembled rig render (middle panel of poster crop)
    rw, rh = rig.size
    assembled = rig.crop((int(rw * 0.30), int(rh * 0.02), int(rw * 0.62), int(rh * 0.58)))
    assembled.save(ASSETS / "rig_assembled.png")

    _draw_feature_fusion_schematic(ASSETS / "feature_fusion.png")
    _draw_yolo_chip(ASSETS / "yolo_chip.png")


def _draw_yolo_chip(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.6, 1.0), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.4, 0.8), 9.2, 4.4, boxstyle="round,pad=0.08,rounding_size=0.35",
        linewidth=1.2, edgecolor="#C56A00", facecolor="#FFF1DC",
    )
    ax.add_patch(box)
    ax.text(5, 3.6, "YOLOv8", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(5, 2.2, "RGB detections", ha="center", va="center", fontsize=8.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def _draw_feature_fusion_schematic(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.4, 1.35), dpi=150)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)
    ax.axis("off")

    cam_x = [2.2, 9.0, 15.8]
    colors = ["#6C8EBF", "#6C8EBF", "#6C8EBF"]
    for x, c in zip(cam_x, colors):
        tile = FancyBboxPatch(
            (x, 7.0), 3.0, 2.2, boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.0, edgecolor="#2E5A8C", facecolor="#D9E6F5",
        )
        ax.add_patch(tile)
        ax.text(x + 1.5, 8.1, "32×24", ha="center", va="center", fontsize=6.5)
        bb = FancyBboxPatch(
            (x, 4.0), 3.0, 2.0, boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.0, edgecolor="#444", facecolor="#EFEFEF",
        )
        ax.add_patch(bb)
        ax.text(x + 1.5, 5.0, "Backbone", ha="center", va="center", fontsize=6.5)
        ax.annotate("", xy=(x + 1.5, 4.0), xytext=(x + 1.5, 7.0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333"))

    fuse = FancyBboxPatch(
        (6.0, 1.5), 8.0, 1.8, boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.0, edgecolor="#1B5E20", facecolor="#E8F5E9",
    )
    ax.add_patch(fuse)
    ax.text(10.0, 2.4, "1×1 fusion + head", ha="center", va="center", fontsize=7.0, fontweight="bold")
    for x in cam_x:
        ax.annotate("", xy=(10.0, 3.3), xytext=(x + 1.5, 4.0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#555"))

    outs = ["left", "mid", "right"]
    for i, (x, label) in enumerate(zip(cam_x, outs)):
        ax.text(x + 1.5, 0.7, label, ha="center", va="center", fontsize=6.3)
        dot = mpatches.Circle((x + 1.5, 1.2), 0.18, facecolor="#2E7D32", edgecolor="none")
        ax.add_patch(dot)

    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def _embed(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    return f"data:image/{mime};base64,{data}"


def _stage_box(x: int, y: int, w: int, h: int, title: str, subtitle: str = "") -> str:
    lines = [f'<text x="{x + w // 2}" y="{y + 22}" text-anchor="middle" '
             f'font-family="Helvetica, Arial, sans-serif" font-size="11" font-weight="600">{title}</text>']
    if subtitle:
        lines.append(
            f'<text x="{x + w // 2}" y="{y + 36}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="8.5" fill="#444">{subtitle}</text>'
        )
    return "\n".join([
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" '
        f'fill="#FFFFFF" stroke="#222222" stroke-width="1.2"/>',
        *lines,
    ])


def _arrow(x1: int, y1: int, x2: int, y2: int, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#222" stroke-width="1.4" '
        f'marker-end="url(#arrow)"{dash}/>'
    )


def _group_brace(x: int, y: int, h: int, label: str) -> str:
    mid = y + h // 2
    return "\n".join([
        f'<path d="M {x} {y + 8} Q {x - 8} {mid} {x} {y + h - 8}" fill="none" stroke="#666" stroke-width="1.1"/>',
        f'<text transform="rotate(-90 {x - 18} {mid})" x="{x - 18}" y="{mid}" text-anchor="middle" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="11" font-weight="700" fill="#333">{label}</text>',
    ])


def _output_pill(x: int, y: int, w: int, h: int, text: str, fill: str) -> str:
    return "\n".join([
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h // 2}" ry="{h // 2}" fill="{fill}" '
        f'stroke="#222" stroke-width="0.8"/>',
        f'<text x="{x + w // 2}" y="{y + h // 2 + 4}" text-anchor="middle" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="8.5" font-weight="600" fill="#111">{text}</text>',
    ])


def build_svg() -> str:
    person_rgb = _embed(ASSETS / "person_rgb.png")
    person_thermal = _embed(ASSETS / "person_thermal_trio.png")
    outdoor_thermal = _embed(ASSETS / "outdoor_thermal_trio.png")
    yolo = _embed(ASSETS / "yolo_chip.png")
    ff = _embed(ASSETS / "feature_fusion.png")
    rig = _embed(ASSETS / "rig_assembled.png")

    W, H = 1080, 438
    parts: list[str] = []

    parts.append(textwrap.dedent(f"""\
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
         width="{W}" height="{H}" viewBox="0 0 {W} {H}">
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" fill="#222"/>
        </marker>
        <style>
          .section-title {{
            font-family: Helvetica, Arial, sans-serif;
            font-size: 12px;
            font-weight: 700;
            fill: #111;
          }}
          .caption {{
            font-family: Helvetica, Arial, sans-serif;
            font-size: 8px;
            fill: #333;
          }}
        </style>
      </defs>
      <rect width="100%" height="100%" fill="#FFFFFF"/>

      <!-- Training group -->
      <rect x="30" y="18" width="1024" height="178" rx="10" ry="10" fill="#FBFBFB" stroke="#BBBBBB" stroke-width="1" stroke-dasharray="6 4"/>
      {_group_brace(48, 32, 150, "Training time")}

      <!-- Training stage headers -->
      {_stage_box(82, 28, 112, 40, "Data Collection", "RGB + thermal rig")}
      {_stage_box(232, 28, 112, 40, "Auto Annotation", "YOLO on RGB")}
      {_stage_box(382, 28, 112, 40, "Thermal Labeling", "cam2 · cam5 · cam4")}
      {_stage_box(532, 28, 124, 40, "Model Training", "Feature Fusion")}

      {_arrow(194, 48, 232, 48)}
      {_arrow(344, 48, 382, 48)}
      {_arrow(494, 48, 532, 48)}

      <!-- Training visuals -->
      <rect x="82" y="74" width="112" height="104" rx="6" ry="6" fill="#FFF7EC" stroke="#E08E00" stroke-width="1"/>
      <image x="84" y="76" width="108" height="100" preserveAspectRatio="xMidYMid meet" href="{person_rgb}"/>
      <text x="138" y="188" text-anchor="middle" class="caption">Synced RGB frame</text>

      <rect x="232" y="74" width="112" height="104" rx="6" ry="6" fill="#FFF7EC" stroke="#E08E00" stroke-width="1"/>
      <image x="236" y="82" width="104" height="64" preserveAspectRatio="xMidYMid meet" href="{yolo}"/>
      <text x="288" y="158" text-anchor="middle" class="caption">Project to panorama</text>
      <text x="288" y="170" text-anchor="middle" class="caption">footprints → labels</text>

      <rect x="382" y="74" width="112" height="104" rx="6" ry="6" fill="#EEF3FA" stroke="#2E5A8C" stroke-width="1"/>
      <image x="384" y="78" width="108" height="92" preserveAspectRatio="xMidYMid meet" href="{person_thermal}"/>
      <text x="438" y="188" text-anchor="middle" class="caption">Supervised thermal targets</text>

      <rect x="532" y="74" width="124" height="104" rx="6" ry="6" fill="#F3F3F3" stroke="#444" stroke-width="1"/>
      <image x="538" y="82" width="112" height="64" preserveAspectRatio="xMidYMid meet" href="{ff}"/>
      <text x="594" y="158" text-anchor="middle" class="caption">Shared backbone +</text>
      <text x="594" y="170" text-anchor="middle" class="caption">per-level 1×1 fusion</text>
      <text x="594" y="188" text-anchor="middle" class="caption">3-class CE (masked)</text>

      <!-- mini annotation flow -->
      <rect x="678" y="74" width="360" height="104" rx="6" ry="6" fill="#FFFFFF" stroke="#CCCCCC" stroke-width="1" stroke-dasharray="4 3"/>
      <text x="690" y="88" class="caption" font-weight="600">Label generation (training only)</text>
      <rect x="690" y="96" width="68" height="30" rx="4" fill="#FFF1DC" stroke="#E08E00"/>
      <text x="724" y="115" text-anchor="middle" class="caption">RGB frame</text>
      {_arrow(758, 111, 778, 111)}
      <rect x="778" y="96" width="88" height="30" rx="4" fill="#EFEFEF" stroke="#666"/>
      <text x="822" y="109" text-anchor="middle" class="caption">Pretrained</text>
      <text x="822" y="120" text-anchor="middle" class="caption">YOLOv8</text>
      {_arrow(866, 111, 886, 111)}
      <rect x="886" y="96" width="68" height="30" rx="4" fill="#FFF1DC" stroke="#E08E00"/>
      <text x="920" y="115" text-anchor="middle" class="caption">BBox labels</text>
      {_arrow(920, 126, 920, 138)}
      <rect x="856" y="138" width="128" height="28" rx="4" fill="#EEF3FA" stroke="#2E5A8C"/>
      <text x="920" y="156" text-anchor="middle" class="caption">Per-direction thermal labels</text>
      <image x="690" y="138" width="150" height="32" preserveAspectRatio="xMidYMid meet" href="{person_thermal}"/>

      <!-- Deployment group -->
      <rect x="30" y="218" width="1024" height="198" rx="10" ry="10" fill="#FBFBFB" stroke="#BBBBBB" stroke-width="1" stroke-dasharray="6 4"/>
      {_group_brace(48, 234, 168, "Deployment")}

      {_stage_box(82, 228, 112, 40, "Wearable Rig", "5 thermal · 3 active")}
      {_stage_box(232, 228, 112, 40, "Live Thermal Input", "cam2 · cam5 · cam4")}
      {_stage_box(382, 228, 112, 40, "Feature Fusion", "inference only")}
      {_stage_box(532, 228, 196, 40, "Per-Camera Output", "navigability state")}

      {_arrow(194, 248, 232, 248)}
      {_arrow(344, 248, 382, 248)}
      {_arrow(494, 248, 532, 248)}

      <rect x="82" y="274" width="112" height="124" rx="6" ry="6" fill="#FFFFFF" stroke="#666" stroke-width="1"/>
      <image x="86" y="280" width="104" height="110" preserveAspectRatio="xMidYMid meet" href="{rig}"/>
      <text x="138" y="408" text-anchor="middle" class="caption">Jetson + 3× MLX90640</text>

      <rect x="232" y="274" width="112" height="124" rx="6" ry="6" fill="#EEF3FA" stroke="#2E5A8C" stroke-width="1"/>
      <image x="234" y="280" width="108" height="110" preserveAspectRatio="xMidYMid meet" href="{outdoor_thermal}"/>
      <text x="288" y="408" text-anchor="middle" class="caption">32×24 tiles (no RGB)</text>

      <rect x="382" y="274" width="112" height="124" rx="6" ry="6" fill="#F3F3F3" stroke="#444" stroke-width="1"/>
      <image x="388" y="288" width="100" height="72" preserveAspectRatio="xMidYMid meet" href="{ff}"/>
      <text x="438" y="378" text-anchor="middle" class="caption">Trained Feature Fusion</text>
      <text x="438" y="392" text-anchor="middle" class="caption">+ optional camera-ID resort</text>
      <text x="438" y="408" text-anchor="middle" class="caption">&lt; 6 ms on Jetson-class CPU</text>

      <!-- per-camera outputs -->
      <rect x="532" y="274" width="196" height="124" rx="6" ry="6" fill="#FFFFFF" stroke="#666" stroke-width="1"/>
      {_output_pill(542, 292, 56, 18, "Free path", "#C8E6C9")}
      {_output_pill(542, 316, 56, 18, "Person", "#FFCDD2")}
      {_output_pill(542, 340, 56, 18, "Other obj.", "#FFE0B2")}
      <text x="570" y="368" text-anchor="middle" class="caption">cam2 (left)</text>
      {_output_pill(602, 292, 56, 18, "Free path", "#C8E6C9")}
      {_output_pill(602, 316, 56, 18, "Person", "#FFCDD2")}
      {_output_pill(602, 340, 56, 18, "Other obj.", "#FFE0B2")}
      <text x="630" y="368" text-anchor="middle" class="caption">cam5 (mid)</text>
      {_output_pill(662, 292, 56, 18, "Free path", "#C8E6C9")}
      {_output_pill(662, 316, 56, 18, "Person", "#FFCDD2")}
      {_output_pill(662, 340, 56, 18, "Other obj.", "#FFE0B2")}
      <text x="690" y="368" text-anchor="middle" class="caption">cam4 (right)</text>
      <text x="630" y="392" text-anchor="middle" class="caption">3-class navigability per direction</text>

      <!-- trained weights link -->
      <path d="M 594 178 C 594 200, 438 200, 438 228" fill="none" stroke="#444" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#arrow)"/>
      <text x="512" y="208" text-anchor="middle" class="caption" font-style="italic">trained weights</text>
    </svg>
    """))
    return "".join(parts)


def main() -> None:
    ensure_assets()
    svg = build_svg()
    OUT_SVG.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
