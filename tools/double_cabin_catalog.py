"""Mount the catalogue's paired cabins on their designated frame receivers."""

from catalog_modeling import add_static_cabin
from single_cabin_catalog import add_sling_mid_mount


STATIONS = {"LOW": "L", "MID": "M", "HIGH": "H"}
PET_CABINS = ("S02", "S03", "S04")
CAMP_CABINS = ("S11", "S12")


def build_double(doc, binary, spec, width, centers, level):
    lower, upper = spec["lower"], spec["upper"]
    lower_port = STATIONS[spec["ports"][0]]
    upper_port = "H" if upper == "S01" else STATIONS[spec["ports"][1]]
    lower_name, upper_name = spec["id"] + "_Lower", spec["id"] + "_Upper"
    lower_profile = "standard"
    lower_shift = (0.0, 0.0)
    upper_shift = (0.0, 0.0)

    # Keep the lower loading opening accessible under overlapping modules.
    if lower in CAMP_CABINS:
        lower_profile = "front-row"
        lower_shift = {("S11", 550): (-0.095, 0.047),
                       ("S11", 650): (-0.075, 0.0),
                       ("S12", 650): (-0.110, 0.029)}[lower, width]
    if lower in PET_CABINS and upper not in PET_CABINS:
        lower_profile = "low"
    if lower_port == "M" and lower in ("S03", "S04", "S13", "S14"):
        # The tucked footrest still sweeps below the upper seat pan, so give
        # the rear rim of each larger lower module its own fore-aft clearance.
        inset = {"S04": -0.223, "S13": -0.189, "S14": -0.210}
        lower_shift = (inset.get(lower, -0.175) if upper == "S05" else -0.175, 0.012)
    if lower == "S15":
        lower_profile = "paired"
        lower_shift = (-0.020, 0.090)
    if lower == "S02":
        lower_shift = (-0.040, 0.079)
    if lower in ("S06", "S07"):
        lower_profile = "low"
    if lower == "S10":
        lower_shift = (0.0, 0.133)

    # Long upper shells sit ahead of the push grip. The wide two-child
    # deck sits slightly higher to clear the rising tubes on both sides.
    if upper == "S06":
        upper_shift = (-0.129, 0.026) if lower_port == "L" else (-0.071, 0.053)
    if upper == "S07":
        upper_shift = (-0.064, 0.013) if lower_port == "L" else (-0.017, 0.056)
    if upper == "S10":
        upper_shift = (-0.064, 0.137)
    if upper == "S02":
        upper_shift = (-0.038 if upper_port == "H" else 0.0, 0.079)
    if upper == "S01":
        upper_shift = (0.0, 0.009)
    upper_profile = "standard"
    if upper == "S05" and lower_port == "M":
        upper_shift = (0.047, 0.116)
        upper_profile = "tucked"

    add_static_cabin(doc, binary, lower_name, lower, width, centers, level, lower_port,
                     shift=lower_shift, profile=lower_profile)
    add_static_cabin(doc, binary, upper_name, upper, width, centers, level, upper_port,
                     shift=upper_shift, profile=upper_profile)
    if upper == "S01":
        add_sling_mid_mount(doc, binary, upper_name + "_MidMount", centers, level, shift=upper_shift)
