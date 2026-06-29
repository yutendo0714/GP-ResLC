from PIL import Image, ImageDraw, ImageFont


OUT = "paper/miru2026/figures/gp_reslc_model_overview.png"
W, H = 1800, 980


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
F_HEAD = font(31, True)
F_BODY = font(25, False)
F_SMALL = font(21, False)
F_MONO = font(23, False)


def rounded(draw, xy, fill, outline, radius=22, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw, box, lines, fnt, fill=(25, 30, 38), line_gap=8):
    if isinstance(lines, str):
        lines = [lines]
    x0, y0, x1, y1 = box
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, tw, th in zip(lines, widths, heights):
        draw.text((x0 + (x1 - x0 - tw) / 2, y), line, font=fnt, fill=fill)
        y += th + line_gap


def arrow(draw, start, end, color=(52, 74, 102), width=5):
    draw.line([start, end], fill=color, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        sign = 1 if x1 >= x0 else -1
        pts = [(x1, y1), (x1 - 24 * sign, y1 - 13), (x1 - 24 * sign, y1 + 13)]
    else:
        sign = 1 if y1 >= y0 else -1
        pts = [(x1, y1), (x1 - 13, y1 - 24 * sign), (x1 + 13, y1 - 24 * sign)]
    draw.polygon(pts, fill=color)


def label(draw, xy, text, fnt=F_SMALL, fill=(70, 80, 95)):
    draw.text(xy, text, font=fnt, fill=fill)


img = Image.new("RGB", (W, H), (250, 252, 255))
draw = ImageDraw.Draw(img)

draw.text((60, 42), "GP-ResLC: decoder-consistent residual coding on pretrained GLC", font=F_TITLE, fill=(18, 28, 40))

# Input / frozen GLC path
boxes = {
    "x": (80, 190, 270, 300),
    "enc": (360, 170, 610, 320),
    "hyper": (710, 115, 980, 245),
    "prior": (1080, 115, 1390, 245),
    "arith": (1490, 235, 1720, 375),
    "dec": (1080, 660, 1390, 810),
    "out": (1500, 680, 1720, 790),
}

rounded(draw, boxes["x"], (235, 244, 255), (84, 132, 186))
centered_text(draw, boxes["x"], ["Input", "image x"], F_HEAD)

rounded(draw, boxes["enc"], (232, 246, 241), (73, 145, 120))
centered_text(draw, boxes["enc"], ["Frozen GLC", "encoder"], F_HEAD)

rounded(draw, boxes["hyper"], (243, 241, 255), (114, 101, 184))
centered_text(draw, boxes["hyper"], ["Hyper", "latent z-hat"], F_HEAD)

rounded(draw, boxes["prior"], (243, 241, 255), (114, 101, 184))
centered_text(draw, boxes["prior"], ["GLC prior", "Q, scales, means"], F_BODY)

rounded(draw, boxes["arith"], (255, 243, 228), (190, 121, 54))
centered_text(draw, boxes["arith"], ["Arithmetic", "coding"], F_HEAD)

rounded(draw, boxes["dec"], (232, 246, 241), (73, 145, 120))
centered_text(draw, boxes["dec"], ["Frozen GLC", "decoder"], F_HEAD)

rounded(draw, boxes["out"], (235, 244, 255), (84, 132, 186))
centered_text(draw, boxes["out"], ["Reconstructed", "image x-hat"], F_BODY)

arrow(draw, (270, 245), (360, 245))
label(draw, (295, 210), "x")
arrow(draw, (610, 215), (710, 180))
label(draw, (632, 160), "z")
arrow(draw, (980, 180), (1080, 180))
label(draw, (1007, 145), "z-hat, q")

# Main latent y path
draw.line([(610, 285), (760, 285), (760, 305)], fill=(52, 74, 102), width=5)
label(draw, (638, 292), "main latent y")

# GP-ResLC modules
mean_box = (800, 335, 1120, 480)
gate_box = (800, 515, 1120, 660)
rounded(draw, mean_box, (235, 249, 244), (52, 149, 114))
centered_text(draw, mean_box, ["GP-ResLC", "predictable mean", "Delta mu(z-hat, q)"], F_BODY)

rounded(draw, gate_box, (235, 249, 244), (52, 149, 114))
centered_text(draw, gate_box, ["GP-ResLC", "precision gate", "rho(z-hat, q) >= 1"], F_BODY)

arrow(draw, (855, 245), (900, 335), color=(52, 149, 114), width=4)
arrow(draw, (895, 245), (910, 515), color=(52, 149, 114), width=4)
label(draw, (840, 286), "decoder-computable, no side map", F_SMALL, (52, 120, 92))

# Residual coding box
res_box = (1210, 370, 1460, 600)
rounded(draw, res_box, (255, 248, 235), (190, 121, 54))
centered_text(draw, res_box, ["Code residual", "r = y - mu_GP", "Q_GP = Q_GLC * rho"], F_BODY)

arrow(draw, (760, 305), (1210, 425))
arrow(draw, (1120, 405), (1210, 430), color=(52, 149, 114), width=4)
arrow(draw, (1120, 585), (1210, 540), color=(52, 149, 114), width=4)
arrow(draw, (1460, 485), (1545, 375))
label(draw, (1490, 445), "shorter y-stream", F_SMALL, (146, 86, 36))

# Bitstream and decode
bit_box = (1485, 430, 1725, 570)
rounded(draw, bit_box, (255, 250, 240), (190, 121, 54))
centered_text(draw, bit_box, ["Serialized", "payload bytes", "z + y + header"], F_BODY)
arrow(draw, (1605, 570), (1240, 660))
arrow(draw, (1390, 735), (1500, 735))

# Frozen and contribution notes
note_box = (80, 700, 690, 860)
rounded(draw, note_box, (244, 247, 250), (145, 154, 166), radius=18, width=2)
centered_text(
    draw,
    note_box,
    [
        "Paper-facing evaluation",
        "bpp = serialized payload bytes / original pixels",
        "GLC modules are frozen; GP-ResLC adds decoder-side controls",
    ],
    F_SMALL,
    fill=(45, 54, 66),
    line_gap=12,
)

draw.text((80, 915), "Key idea: do not spend bits on components predictable from the generative latent prior; code only the perceptually useful residual.", font=F_MONO, fill=(45, 54, 66))

img.save(OUT)
print(OUT)
