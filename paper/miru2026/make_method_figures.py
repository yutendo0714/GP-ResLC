from PIL import Image, ImageDraw, ImageFont


FIG_DIR = "paper/miru2026/figures"


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_HEAD = font(30, True)
F_BODY = font(23)
F_SMALL = font(20)


def rounded(draw, box, fill, outline, radius=18, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw, box, lines, fnt, fill=(25, 31, 40), gap=8):
    if isinstance(lines, str):
        lines = [lines]
    x0, y0, x1, y1 = box
    sizes = []
    for line in lines:
        b = draw.textbbox((0, 0), line, font=fnt)
        sizes.append((b[2] - b[0], b[3] - b[1]))
    total_h = sum(h for _, h in sizes) + gap * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, (w, h) in zip(lines, sizes):
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=fnt, fill=fill)
        y += h + gap


def arrow(draw, a, b, color=(55, 75, 100), width=5):
    draw.line([a, b], fill=color, width=width)
    x0, y0 = a
    x1, y1 = b
    if abs(x1 - x0) >= abs(y1 - y0):
        s = 1 if x1 >= x0 else -1
        pts = [(x1, y1), (x1 - 22 * s, y1 - 12), (x1 - 22 * s, y1 + 12)]
    else:
        s = 1 if y1 >= y0 else -1
        pts = [(x1, y1), (x1 - 12, y1 - 22 * s), (x1 + 12, y1 - 22 * s)]
    draw.polygon(pts, fill=color)


def label(draw, xy, text, fnt=F_SMALL, fill=(70, 80, 95)):
    draw.text(xy, text, font=fnt, fill=fill)


def make_glc_overview():
    w, h = 1800, 760
    img = Image.new("RGB", (w, h), (250, 252, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 36), "GLC natural image codec: generative latent transform coding", font=F_TITLE, fill=(20, 30, 42))

    boxes = {
        "x": (70, 210, 250, 330),
        "vqenc": (335, 180, 590, 360),
        "latent": (680, 185, 900, 355),
        "trans": (990, 175, 1260, 365),
        "entropy": (1350, 175, 1640, 365),
        "vqdec": (990, 500, 1260, 660),
        "out": (1390, 520, 1640, 640),
    }
    rounded(d, boxes["x"], (235, 244, 255), (79, 126, 180))
    text_center(d, boxes["x"], ["Input", "image x"], F_HEAD)
    rounded(d, boxes["vqenc"], (233, 248, 241), (67, 143, 112))
    text_center(d, boxes["vqenc"], ["VQGAN /", "VQ-VAE", "encoder"], F_HEAD)
    rounded(d, boxes["latent"], (243, 241, 255), (111, 101, 184))
    text_center(d, boxes["latent"], ["generative", "latent l"], F_HEAD)
    rounded(d, boxes["trans"], (255, 247, 231), (190, 121, 54))
    text_center(d, boxes["trans"], ["latent", "transform", "g_a, g_s"], F_HEAD)
    rounded(d, boxes["entropy"], (255, 247, 231), (190, 121, 54))
    text_center(d, boxes["entropy"], ["entropy model", "hyperprior +", "4-part context"], F_BODY)
    rounded(d, boxes["vqdec"], (233, 248, 241), (67, 143, 112))
    text_center(d, boxes["vqdec"], ["VQGAN /", "VQ-VAE", "decoder"], F_HEAD)
    rounded(d, boxes["out"], (235, 244, 255), (79, 126, 180))
    text_center(d, boxes["out"], ["reconstructed", "image x-hat"], F_BODY)

    arrow(d, (250, 270), (335, 270))
    arrow(d, (590, 270), (680, 270))
    arrow(d, (900, 270), (990, 270))
    arrow(d, (1260, 270), (1350, 270))
    arrow(d, (1125, 365), (1125, 500))
    arrow(d, (1260, 580), (1390, 580))
    label(d, (1020, 392), "decoded latent l-hat", F_SMALL)
    label(d, (1345, 392), "coded streams: z and y", F_SMALL, (145, 89, 37))

    note = (70, 520, 845, 660)
    rounded(d, note, (244, 247, 250), (145, 154, 166), radius=16, width=2)
    text_center(d, note, [
        "GLC's strength: transform coding is performed in a perceptual/generative latent space.",
        "GP-ResLC starts from this GLC codec and changes how bits are allocated in y-stream coding.",
    ], F_SMALL, fill=(45, 54, 66), gap=12)
    img.save(f"{FIG_DIR}/glc_model_overview_simple.png")


def make_difference():
    w, h = 1800, 900
    img = Image.new("RGB", (w, h), (250, 252, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 34), "What GP-ResLC changes relative to GLC", font=F_TITLE, fill=(20, 30, 42))

    # Left: GLC
    d.text((90, 115), "Original GLC coding path", font=F_HEAD, fill=(40, 52, 68))
    rounded(d, (90, 175, 365, 315), (243, 241, 255), (111, 101, 184))
    text_center(d, (90, 175, 365, 315), ["z-hat", "quality q"], F_HEAD)
    rounded(d, (475, 165, 815, 325), (255, 247, 231), (190, 121, 54))
    text_center(d, (475, 165, 815, 325), ["GLC prior", "Q, scales, means"], F_HEAD)
    rounded(d, (930, 165, 1250, 325), (255, 247, 231), (190, 121, 54))
    text_center(d, (930, 165, 1250, 325), ["arithmetic-code", "y around GLC prior"], F_BODY)
    rounded(d, (1370, 175, 1650, 315), (235, 244, 255), (79, 126, 180))
    text_center(d, (1370, 175, 1650, 315), ["bitstream", "z + y + header"], F_BODY)
    arrow(d, (365, 245), (475, 245))
    arrow(d, (815, 245), (930, 245))
    arrow(d, (1250, 245), (1370, 245))

    # Right/bottom: GP-ResLC
    d.text((90, 445), "GP-ResLC: decoder-consistent residual/precision control", font=F_HEAD, fill=(40, 52, 68))
    rounded(d, (90, 505, 365, 645), (243, 241, 255), (111, 101, 184))
    text_center(d, (90, 505, 365, 645), ["same z-hat", "same q"], F_HEAD)
    rounded(d, (475, 495, 815, 655), (235, 249, 244), (52, 149, 114))
    text_center(d, (475, 495, 815, 655), ["predictable mean", "Delta mu(z-hat, q)", "no side bits"], F_BODY)
    rounded(d, (475, 690, 815, 820), (235, 249, 244), (52, 149, 114))
    text_center(d, (475, 690, 815, 820), ["precision gate", "rho(z-hat, q) >= 1"], F_BODY)
    rounded(d, (930, 530, 1250, 730), (255, 247, 231), (190, 121, 54))
    text_center(d, (930, 530, 1250, 730), ["code residual", "r = y - (mu + Delta mu)", "Q_GP = Q_GLC * rho"], F_BODY)
    rounded(d, (1370, 560, 1650, 700), (235, 244, 255), (79, 126, 180))
    text_center(d, (1370, 560, 1650, 700), ["shorter", "y-stream"], F_HEAD)
    arrow(d, (365, 575), (475, 575), color=(52, 149, 114))
    arrow(d, (365, 610), (475, 755), color=(52, 149, 114))
    arrow(d, (815, 575), (930, 610), color=(52, 149, 114))
    arrow(d, (815, 755), (930, 660), color=(52, 149, 114))
    arrow(d, (1250, 630), (1370, 630))
    label(d, (930, 762), "decoder sees the same controls, so no encoder-only map is sent", F_SMALL, (52, 120, 92))

    img.save(f"{FIG_DIR}/gp_reslc_difference_from_glc.png")


if __name__ == "__main__":
    make_glc_overview()
    make_difference()
    print(f"{FIG_DIR}/glc_model_overview_simple.png")
    print(f"{FIG_DIR}/gp_reslc_difference_from_glc.png")
