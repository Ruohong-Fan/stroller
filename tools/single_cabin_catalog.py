"""Single-cabin layouts and the dedicated sling and adult seat modules."""

from __future__ import annotations

import math

from create_d02_double_seat import Geometry, port_position


# Body offsets preserve each named receiver while clearing the sloping rails.
SINGLE_BODY_OFFSETS = {
    ("S02", 380): (0.000, 0.090),
    ("S03", 550): (-0.045, 0.075),
    ("S04", 650): (-0.060, 0.090),
    ("S10", 650): (-0.045, 0.135),
    ("S11", 550): (-0.075, 0.073),
    ("S11", 650): (-0.075, 0.058),
    ("S12", 650): (-0.075, 0.058),
    ("S13", 550): (-0.060, 0.075),
    ("S14", 650): (-0.060, 0.090),
    ("S15", 380): (0.000, 0.090),
}


def sling_geometry():
    """One fabric surface runs from the backrest to the soft foot sling."""
    fabric, edging, reinforcement, harness = (Geometry() for _ in range(4))
    # The curved cross sections deepen through the hips and flatten at the feet.
    sections = [
        (-0.475, -0.073, 0.096, 0.012),
        (-0.442, -0.080, 0.115, 0.018),
        (-0.363, -0.018, 0.131, 0.025),
        (-0.293, 0.036, 0.143, 0.038),
        (-0.238, 0.032, 0.146, 0.047),
        (-0.157, 0.044, 0.149, 0.052),
        (-0.097, 0.111, 0.152, 0.041),
        (-0.052, 0.210, 0.154, 0.031),
        (-0.012, 0.305, 0.153, 0.019),
        (0.041, 0.390, 0.145, 0.010),
        (0.056, 0.407, 0.127, 0.009),
    ]
    rows = [
        [(x, edge_y - sag * (1 - (i / 5) ** 2), width * i / 5) for i in range(-5, 6)]
        for x, edge_y, width, sag in sections
    ]
    for front, back in zip(rows, rows[1:]):
        for i in range(len(front) - 1):
            fabric.quad(front[i], front[i + 1], back[i + 1], back[i])
    for edge in (0, -1):
        edging.tube([row[edge] for row in rows], 0.0055, (0, 0, 1), sides=8)
    for row in (rows[0], rows[2], rows[-1]):
        edging.tube(row, 0.004, (1, 0, 0), sides=8)

    for side in (-1, 1):
        # Soft reinforced flaps extend to the two station mounts at either side.
        reinforcement.side_cheek(
            [(-0.132, 0.119), (-0.099, 0.194), (-0.036, 0.232), (0.022, 0.163),
             (0.021, 0.117)], side * 0.146, 0.012)
        reinforcement.side_cheek(
            [(-0.286, -0.034), (-0.281, 0.036), (-0.204, 0.037), (-0.182, 0.006),
             (-0.208, -0.033)], side * 0.144, 0.013)
        harness.tube(
            [(-0.058, 0.185, side * 0.074), (-0.105, 0.094, side * 0.063),
             (-0.186, 0.021, side * 0.018)], 0.007, (0, 0, 1), sides=8)
        harness.beam((-0.166, 0.022, side * 0.125), (-0.196, 0.020, side * 0.014), 0.007, sides=8)
    harness.beam((-0.251, 0.016, 0), (-0.202, 0.022, 0), 0.008, sides=8)
    harness.box((-0.197, 0.023, 0), (0.025, 0.018, 0.032))
    return {"upholstery": fabric, "canopy_stay": edging, "canopy": reinforcement, "harness": harness}


def adult_geometry(width):
    frame, pad, trim, cheek = (Geometry() for _ in range(4))
    wide = width == 650
    seat_width = 0.514 if wide else 0.438
    arm = 0.248 if wide else 0.213
    front = -0.259 if wide else -0.228
    back_height = 0.665 if wide else 0.621
    back_depth = 0.235 if wide else 0.221
    mid = (front + 0.181) / 2
    depth = 0.181 - front

    frame.seat_pad((mid, 0.167, 0), (depth + 0.018, 0.038, seat_width + 0.022),
                   0.048, 0.009, 0.010)
    pad.seat_pad((mid, 0.202, 0), (depth, 0.052, seat_width), 0.053, 0.011, 0.012)
    # A moderately reclined, padded back shell with open side entry.
    back_tilt = math.radians(-10)
    back_mid = (0.205, (0.196 + back_height) / 2, 0)
    frame.box(back_mid, (0.035, back_height - 0.174, seat_width + 0.008), back_tilt)
    pad.box((back_mid[0] - 0.027, back_mid[1], 0),
            (0.044, back_height - 0.197, seat_width - 0.014), back_tilt)
    for side in (-1, 1):
        lateral = side * arm
        frame.tube([(front + 0.058, 0.166, lateral), (front + 0.037, 0.315, lateral),
                    (0.173, 0.324, lateral), (0.206, 0.190, lateral)],
                   0.014, (0, 0, 1), sides=12)
        frame.beam((0.144, 0.155, lateral), (back_depth, 0.364, lateral), 0.018)
        cheek.side_cheek(
            [(-0.087, 0.127), (-0.073, 0.178), (0.083, 0.178),
             (0.091, 0.149), (0.056, 0.130)], lateral, 0.027)
        cheek.side_cheek(
            [(0.161, 0.228), (0.195, 0.340), (back_depth + 0.029, 0.341),
             (back_depth + 0.019, 0.290), (0.190, 0.227)], lateral, 0.024)
        pad.seat_pad(((front + 0.207) / 2, 0.334, lateral),
                     (0.173 - front, 0.039, 0.047), 0.017, 0.008, 0.0)
        trim.beam((0.156, 0.240, side * (seat_width / 2 - 0.023)),
                  (0.223, back_height - 0.027, side * (seat_width / 2 - 0.023)), 0.004)
    return {"shell": frame, "upholstery": pad, "canopy_stay": trim, "mount_cheek": cheek}


def special_cabin_geometry(cabin_id, width, profile="standard"):
    if cabin_id == "S01":
        return sling_geometry()
    if cabin_id in ("S08", "S09"):
        return adult_geometry(width)
    raise ValueError(f"No dedicated single-cabin geometry for {cabin_id}")


def add_sling_mid_mount(doc, binary, name, centers, level, *, shift=(0, 0)):
    from catalog_modeling import add_aux_mount

    x, y = port_position(centers, "H")
    dx, dy = shift
    return add_aux_mount(doc, binary, name, centers, level, "M",
                         (x + dx - 0.240, y + dy - 0.030, 0.142))


def add_adult_footrest(doc, binary, name, width, centers, level):
    from catalog_modeling import add_aux_mount, attach

    x, y = port_position(centers, "L")
    deck_width = 0.479 if width == 650 else 0.407
    shell, grip = Geometry(), Geometry()
    shell.seat_pad((-0.123, 0.133, 0), (0.207, 0.031, deck_width), 0.034, 0.008, 0)
    for station in range(5):
        edge = deck_width / 2 - 0.025
        tread_x = -0.190 + 0.029 * station
        grip.beam((tread_x, 0.149, -edge), (tread_x, 0.149, edge), 0.004, sides=8)
    attach(doc, binary, name, {"legrest_shell": shell, "legrest_arm": grip}, (x, y), level)
    add_aux_mount(doc, binary, name + "_LowDocks", centers, level, "L",
                  (x - 0.095, y + 0.130, deck_width / 2 - 0.013))


def build_single(doc, binary, spec, width, centers, level):
    from catalog_modeling import add_aux_mount, add_static_cabin

    cabin_id = spec["id"]
    station = {"S01": "H", "S15": "L"}.get(cabin_id, "M")
    name = cabin_id + "_Cabin"
    shift = SINGLE_BODY_OFFSETS.get((cabin_id, width), (0, 0))
    profile = {"S05": "sitting", "S15": "paired"}.get(cabin_id, "standard")
    add_static_cabin(doc, binary, name, cabin_id, width, centers, level, station,
                     shift=shift, profile=profile)
    if cabin_id == "S01":
        add_sling_mid_mount(doc, binary, name + "_MidDocks", centers, level)
    elif cabin_id in ("S08", "S09"):
        x, y = port_position(centers, "M")
        add_aux_mount(doc, binary, name + "_HighDocks", centers, level, "H",
                      (x + 0.239, y + 0.318, 0.248 if width == 650 else 0.213))
        add_adult_footrest(doc, binary, name + "_Footrest", width, centers, level)
