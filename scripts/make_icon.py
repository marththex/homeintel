"""
scripts/make_icon.py — Generate the HomeIntel home-screen / PWA icons.

Renders a clean branded mark (indigo gradient + white house with a smart "lens")
at high resolution and writes the sizes iOS / PWA need into frontend/public/.

Run:
    python ../scripts/make_icon.py     (from backend/, homeintel env — needs Pillow)
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SS = 1024  # supersample render size
TOP = (0x81, 0x8C, 0xF8)  # indigo (accent-hover)
BOT = (0x4F, 0x46, 0xE5)  # deep indigo
WHITE = (255, 255, 255, 255)


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def build() -> Image.Image:
    # Background — vertical indigo gradient
    bg = Image.new("RGB", (SS, SS))
    dg = ImageDraw.Draw(bg)
    for y in range(SS):
        dg.line([(0, y), (SS, y)], fill=lerp(TOP, BOT, y / SS))

    # House mark on a transparent layer
    house = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
    dh = ImageDraw.Draw(house)
    cx = SS // 2

    roof = [(int(SS * 0.20), int(SS * 0.52)), (cx, int(SS * 0.27)), (int(SS * 0.80), int(SS * 0.52))]
    body = [int(SS * 0.28), int(SS * 0.49), int(SS * 0.72), int(SS * 0.76)]
    dh.polygon(roof, fill=WHITE)
    dh.rounded_rectangle(body, radius=int(SS * 0.05), fill=WHITE)

    # Smart "lens" window (indigo ring + white pupil) — home + intelligence
    lcx, lcy, lr = cx, int(SS * 0.625), int(SS * 0.085)
    dh.ellipse([lcx - lr, lcy - lr, lcx + lr, lcy + lr], fill=BOT + (255,))
    ir = int(lr * 0.42)
    dh.ellipse([lcx - ir, lcy - ir, lcx + ir, lcy + ir], fill=WHITE)

    # Soft drop shadow from the house silhouette
    alpha = house.split()[3]
    shadow = Image.composite(
        Image.new("RGBA", (SS, SS), (12, 12, 30, 150)),
        Image.new("RGBA", (SS, SS), (0, 0, 0, 0)),
        alpha,
    ).filter(ImageFilter.GaussianBlur(int(SS * 0.018)))
    shadow_off = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
    shadow_off.paste(shadow, (0, int(SS * 0.012)), shadow)

    out = bg.convert("RGBA")
    out = Image.alpha_composite(out, shadow_off)
    out = Image.alpha_composite(out, house)
    return out.convert("RGB")


def main() -> None:
    icon = build()
    pub = Path(__file__).parent.parent / "frontend" / "public"
    pub.mkdir(parents=True, exist_ok=True)
    targets = [
        (180, "apple-touch-icon.png"),  # iOS home screen
        (192, "icon-192.png"),          # PWA manifest
        (512, "icon-512.png"),          # PWA manifest
    ]
    for size, name in targets:
        icon.resize((size, size), Image.LANCZOS).save(pub / name)
        print(f"  wrote {name} ({size}x{size})")
    print(f"Icons written to {pub}")


if __name__ == "__main__":
    main()
