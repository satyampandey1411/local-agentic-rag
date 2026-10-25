"""Render the cover illustration for the progress report (PIL only).

An isometric scene that tells the pipeline in one picture: a floating stack of
PDF pages (text, a table, a scanned page) breaks into chunk cubes, which flow
into a glowing retrieval index; on the right, a cited answer card and language
tiles (A, अ, म) for English, Hindi and Marathi.

    python3 report/tools/make_cover.py report/figures/cover.png
"""

import math
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "cover.png"
SS = 2                           # supersampling factor
W, H = 2480 * SS, 1060 * SS      # 21 cm x 8.97 cm at 300 dpi
random.seed(20261025)

NAVY_DEEP = (6, 12, 30)
NAVY = (18, 36, 72)
CYAN = (96, 205, 240)
TEAL = (52, 170, 190)
AMBER = (245, 170, 80)
GLASS = (200, 228, 250)
FONT_DIR = "/usr/share/fonts/truetype/noto/"

UNIT = 52 * SS                   # one isometric unit in pixels
C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))
OX, OY = int(W * 0.465), int(H * 0.60)


def iso(x, y, z=0.0):
    return (OX + (x - y) * C30 * UNIT, OY + (x + y) * S30 * UNIT - z * UNIT)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def shade(col, f):
    return tuple(max(0, min(255, int(c * f))) for c in col[:3])


def layer():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def add(base, lay, glow=0, strength=1.0):
    if glow:
        g = lay.filter(ImageFilter.GaussianBlur(glow))
        if strength != 1.0:
            g.putalpha(g.getchannel("A").point(lambda v: min(255, int(v * strength))))
        base.alpha_composite(g)
    base.alpha_composite(lay)


def background():
    small = Image.new("RGB", (124, 53))
    px = small.load()
    for y in range(53):
        for x in range(124):
            fx, fy = x / 123, y / 52
            base = lerp(NAVY_DEEP, NAVY, 0.25 + 0.55 * fy)
            d1 = math.hypot((fx - 0.50) * 1.5, (fy - 0.45) * 1.2)
            d2 = math.hypot((fx - 0.82) * 1.8, (fy - 0.40) * 1.3)
            col = lerp(base, (24, 86, 120), max(0.0, 1 - d1) ** 2.0 * 0.9)
            col = lerp(col, (70, 52, 40), max(0.0, 1 - d2) ** 3.0 * 0.5)
            px[x, y] = col
    return small.resize((W, H), Image.BICUBIC).convert("RGBA")


def floor_grid():
    lay = layer()
    d = ImageDraw.Draw(lay)
    n = 26
    for i in range(-n, n + 1):
        for a, b in (((i, -n), (i, n)), ((-n, i), (n, i))):
            # fade segments by distance from the scene centre
            steps = 40
            for k in range(steps):
                t0, t1 = k / steps, (k + 1) / steps
                p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                dist = math.hypot(mx - 1.0, my - 1.0)
                alpha = int(70 * max(0.0, 1 - dist / 17) ** 1.4)
                if alpha > 2:
                    d.line([iso(*p0, -2.6), iso(*p1, -2.6)], fill=CYAN + (alpha,), width=SS)
    return lay


def shadow(cx, cy, rx, ry, z=-2.6, alpha=120):
    lay = layer()
    d = ImageDraw.Draw(lay)
    sx, sy = iso(cx, cy, z)
    d.ellipse([sx - rx * UNIT, sy - ry * UNIT, sx + rx * UNIT, sy + ry * UNIT], fill=(0, 0, 0, alpha))
    return lay.filter(ImageFilter.GaussianBlur(18 * SS))


def box(d, x0, y0, z0, dx, dy, dz, col, alpha=255, edge=None):
    """Isometric box; returns the three visible face polygons."""
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
    top = [iso(x0, y0, z1), iso(x1, y0, z1), iso(x1, y1, z1), iso(x0, y1, z1)]
    left = [iso(x0, y1, z1), iso(x1, y1, z1), iso(x1, y1, z0), iso(x0, y1, z0)]
    right = [iso(x1, y0, z1), iso(x1, y1, z1), iso(x1, y1, z0), iso(x1, y0, z0)]
    d.polygon(left, fill=shade(col, 0.62) + (alpha,))
    d.polygon(right, fill=shade(col, 0.80) + (alpha,))
    d.polygon(top, fill=shade(col, 1.0) + (alpha,))
    if edge:
        for poly in (top, left, right):
            d.line(poly + [poly[0]], fill=edge, width=2 * SS)
    return top, left, right


def on_plane(origin, ex, ey, u, v):
    return (origin[0] + ex[0] * u + ey[0] * v, origin[1] + ex[1] * u + ey[1] * v)


def page_stack():
    """Floating stack of PDF pages; the top page shows text lines and a table."""
    lay = layer()
    d = ImageDraw.Draw(lay)
    px, py, pw, pl = -10.4, -1.2, 4.6, 6.0
    for k in range(5):
        z = -0.4 + k * 0.42
        tone = lerp((120, 160, 210), GLASS, k / 4)
        box(d, px + k * 0.05, py - k * 0.05, z, pw, pl, 0.12, tone, alpha=150 + k * 20,
            edge=(220, 240, 255, 120))
    # content on the top page, drawn in the plane of its top face
    zt = -0.4 + 4 * 0.42 + 0.12
    x0, y0 = px + 0.2, py - 0.2

    def P(u, v):
        return iso(x0 + u, y0 + v, zt)

    ink = (40, 70, 115, 230)
    d.line([P(0.45, 0.55), P(0.45, 3.2)], fill=(30, 55, 95, 255), width=7 * SS)  # title line
    for i, ln in enumerate([3.9, 4.2, 3.6, 4.1, 2.8]):
        u = 1.05 + i * 0.36
        d.line([P(u, 0.55), P(u, 0.55 + ln)], fill=ink, width=3 * SS)
    # small ruled table
    tu, tv, tw, th = 3.0, 0.55, 1.15, 4.6
    for r in range(4):
        u = tu + r * tw / 3
        d.line([P(u, tv), P(u, tv + th)], fill=(60, 120, 170, 230), width=2 * SS)
    for c in range(5):
        v = tv + c * th / 4
        d.line([P(tu, v), P(tu + tw, v)], fill=(60, 120, 170, 230), width=2 * SS)
    d.polygon([P(tu, tv), P(tu, tv + th), P(tu + tw / 3, tv + th), P(tu + tw / 3, tv)],
              fill=(96, 205, 240, 90))
    # PDF tag
    d.polygon([P(0.25, 4.6), P(0.25, 5.6), P(0.85, 5.6), P(0.85, 4.6)], fill=(214, 72, 64, 235))
    return lay


def chunk_flow():
    """Chunk cubes along a curve from the page stack into the index."""
    lay = layer()
    d = ImageDraw.Draw(lay)
    pts = []
    for i in range(10):
        t = i / 9
        x = -5.2 + t * 5.4
        y = -0.6 + math.sin(t * math.pi) * -1.2
        z = 1.6 + math.sin(t * math.pi) * 1.0 - t * 0.4
        pts.append((x, y, z))
    trail = layer()
    td = ImageDraw.Draw(trail)
    td.line([iso(*p) for p in pts], fill=CYAN + (110,), width=3 * SS)
    lay.alpha_composite(trail.filter(ImageFilter.GaussianBlur(3 * SS)))
    for i, (x, y, z) in enumerate(pts[1:-1]):
        s = 0.55 - 0.02 * i
        col = AMBER if i in (2, 6) else (90, 190, 230)
        box(d, x - s / 2, y - s / 2, z - s / 2, s, s, s, col, alpha=235, edge=(255, 255, 255, 90))
    return lay


def index_tower():
    """Retrieval index: glowing rings around a light column."""
    glow = layer()
    gd = ImageDraw.Draw(glow)
    lay = layer()
    d = ImageDraw.Draw(lay)
    cx, cy = 1.2, 0.8
    sx, sy = iso(cx, cy, 0)
    for k, (z, r) in enumerate([(-1.6, 3.0), (0.0, 2.5), (1.5, 2.0), (2.8, 1.5)]):
        _, yz = iso(cx, cy, z)
        rx, ry = r * UNIT * 1.05, r * UNIT * 0.52
        col = CYAN if k % 2 == 0 else TEAL
        gd.ellipse([sx - rx, yz - ry, sx + rx, yz + ry], outline=col + (255,), width=9 * SS)
        d.ellipse([sx - rx, yz - ry, sx + rx, yz + ry], outline=(210, 245, 255, 235), width=3 * SS)
        # nodes on the ring
        for j in range(7):
            a = j / 7 * 2 * math.pi + k * 0.6
            nx, ny = sx + rx * math.cos(a), yz + ry * math.sin(a)
            rr = 7 * SS if j % 3 else 10 * SS
            fill = AMBER + (255,) if (j + k) % 5 == 0 else (230, 250, 255, 255)
            d.ellipse([nx - rr, ny - rr, nx + rr, ny + rr], fill=fill)
    # central light column
    _, ytop = iso(cx, cy, 3.6)
    _, ybot = iso(cx, cy, -1.9)
    col = layer()
    cd = ImageDraw.Draw(col)
    cd.rectangle([sx - 16 * SS, ytop, sx + 16 * SS, ybot], fill=(150, 230, 255, 150))
    col = col.filter(ImageFilter.GaussianBlur(14 * SS))
    return glow, lay, col


def text_on_face(base, text, font, face, color):
    """Draw text into an isometric face (4 screen points: tl, tr, br, bl)."""
    size = 256
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    bb = td.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    td.text(((size - tw) / 2 - bb[0], (size - th) / 2 - bb[1]), text, font=font, fill=color)
    tl, tr, _, bl = face
    # affine map: screen -> tile coordinates
    ax, ay = (tr[0] - tl[0]) / size, (tr[1] - tl[1]) / size
    bx, by = (bl[0] - tl[0]) / size, (bl[1] - tl[1]) / size
    det = ax * by - ay * bx
    ia, ib, id_, ie = by / det, -bx / det, -ay / det, ax / det
    coeffs = (ia, ib, -(ia * tl[0] + ib * tl[1]), id_, ie, -(id_ * tl[0] + ie * tl[1]))
    warped = tile.transform((W, H), Image.AFFINE, coeffs, resample=Image.BICUBIC)
    base.alpha_composite(warped)


def answer_card(base):
    """Upright answer card with citation chips, facing the viewer at an angle."""
    lay = layer()
    d = ImageDraw.Draw(lay)
    # card in the x-z plane at y = y0
    x0, y0, z0, cw, ch = 5.0, -4.6, -0.6, 6.4, 4.2

    def P(u, v):  # u along x, v downwards from the top edge
        return iso(x0 + u, y0, z0 + ch - v)

    card = [P(0, 0), P(cw, 0), P(cw, ch), P(0, ch)]
    edge = [P(0, 0), P(0, ch), iso(x0, y0 + 0.18, z0), iso(x0, y0 + 0.18, z0 + ch)]
    d.polygon(edge, fill=(120, 160, 200, 200))
    d.polygon(card, fill=(236, 246, 255, 238))
    d.line(card + [card[0]], fill=(255, 255, 255, 255), width=2 * SS)
    # answer lines
    for i, ln in enumerate([5.2, 4.6, 5.0, 3.4]):
        v = 0.75 + i * 0.5
        d.line([P(0.45, v), P(0.45 + ln, v)], fill=(60, 90, 130, 220), width=4 * SS)
    # header bar
    d.polygon([P(0, 0), P(cw, 0), P(cw, 0.35), P(0, 0.35)], fill=(28, 60, 110, 240))
    base.alpha_composite(lay)
    f = ImageFont.truetype(FONT_DIR + "NotoSans-Bold.ttf", 120)
    for k, label in enumerate(["p.18", "p.20"]):
        u0 = 0.45 + k * 2.2
        chip = [P(u0, 2.85), P(u0 + 1.9, 2.85), P(u0 + 1.9, 3.65), P(u0, 3.65)]
        cl = layer()
        ImageDraw.Draw(cl).polygon(chip, fill=AMBER + (255,) if k == 0 else (96, 205, 240, 255))
        base.alpha_composite(cl)
        text_on_face(base, label, f, chip, (20, 30, 50, 255))
    # verified badge
    bl = layer()
    bd = ImageDraw.Draw(bl)
    cx, cy = P(cw - 0.75, 3.25)
    r = 0.55 * UNIT
    bd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 190, 120, 255))
    bd.line([(cx - r * 0.45, cy), (cx - r * 0.1, cy + r * 0.38), (cx + r * 0.5, cy - r * 0.35)],
            fill=(255, 255, 255, 255), width=8 * SS, joint="curve")
    base.alpha_composite(bl)


def language_tiles(base):
    lay = layer()
    d = ImageDraw.Draw(lay)
    faces = []
    specs = [(8.0, -9.0, 3.3, (96, 205, 240)), (9.9, -10.6, 4.1, AMBER), (11.8, -12.2, 3.1, (150, 140, 235))]
    for x, y, z, col in specs:
        _, left, _ = box(d, x, y, z, 1.6, 1.6, 1.6, col, alpha=245, edge=(255, 255, 255, 120))
        faces.append(left)
    base.alpha_composite(lay)
    latin = ImageFont.truetype(FONT_DIR + "NotoSans-Bold.ttf", 170)
    deva = ImageFont.truetype(FONT_DIR + "NotoSansDevanagari-Bold.ttf", 160)
    for face, (txt, font) in zip(faces, [("A", latin), ("अ", deva), ("म", deva)], strict=True):
        tl, tr, br, bl = face
        text_on_face(base, txt, font, [tl, tr, br, bl], (12, 22, 44, 255))


def answer_beam():
    """Light path from the top of the index to the answer card."""
    lay = layer()
    d = ImageDraw.Draw(lay)
    a, b = (1.2, 0.8, 3.0), (5.2, -4.4, 2.2)
    pts = []
    for i in range(30):
        t = i / 29
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        z = a[2] + (b[2] - a[2]) * t + math.sin(t * math.pi) * 1.4
        pts.append(iso(x, y, z))
    d.line(pts, fill=AMBER + (200,), width=4 * SS)
    for i in range(3, 29, 5):
        x, y = pts[i]
        r = 6 * SS
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 225, 170, 255))
    return lay


def particles():
    lay = layer()
    d = ImageDraw.Draw(lay)
    for _ in range(170):
        x, y = random.uniform(0, W), random.uniform(0, H * 0.85)
        r = random.uniform(1.0, 3.2) * SS
        a = random.randint(50, 190)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(200, 235, 255, a))
    return lay


def main():
    img = background()
    add(img, floor_grid(), glow=2 * SS)
    img.alpha_composite(particles())
    img.alpha_composite(shadow(-8.0, 1.8, 3.4, 1.4))
    img.alpha_composite(shadow(1.2, 0.8, 3.0, 1.3, alpha=90))
    img.alpha_composite(shadow(8.0, -2.0, 3.6, 1.2, alpha=100))
    glow, rings, column = index_tower()
    img.alpha_composite(column)
    add(img, glow.filter(ImageFilter.GaussianBlur(10 * SS)))
    img.alpha_composite(rings)
    add(img, page_stack(), glow=10 * SS, strength=0.5)
    add(img, chunk_flow(), glow=8 * SS, strength=0.6)
    add(img, answer_beam(), glow=6 * SS, strength=0.8)
    answer_card(img)
    language_tiles(img)
    out = img.convert("RGB").resize((W // SS, H // SS), Image.LANCZOS)
    out.save(OUT, quality=95)


if __name__ == "__main__":
    main()
