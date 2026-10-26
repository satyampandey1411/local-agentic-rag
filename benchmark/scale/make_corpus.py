#!/usr/bin/env python3
"""Deterministic generator for the scale-benchmark corpus.

Produces one long, coherent, entirely fictional publication -- the "Statistical and
Administrative Compendium" of the National Statistics and Development Agency (NSDA)
of the Republic of Veloria -- at an exact page count, plus a ground-truth question
set whose answers were planted at known pages.

    python benchmark/scale/make_corpus.py --pages 1000 --out DIR [--seed 7]
    python benchmark/scale/make_corpus.py --pages all --out DIR

writes, per tier, ``DIR/scale-N.pdf``, ``DIR/scale-N.questions.jsonl`` and
``DIR/scale-N.meta.json``.

Design notes
------------
* Pages are written as raw PDF content streams (Base-14 Times/Helvetica/Courier,
  WinAnsi, ASCII-only text) instead of ``page.insert_text`` -- about 1 ms per page,
  so 10,000 pages build in well under a minute.  Ruled tables are real stroked grid
  lines, so ``page.find_tables`` detects them.
* Scanned pages are laid out with the same code in a scratch document, rasterised to
  grey JPEG at 150 dpi with a slight skew and sensor noise, and inserted as image-only
  pages (no text layer at all).
* Every number, name and code comes from a ``random.Random`` seeded with
  ``(seed, pages)``; the PDF is saved with fixed metadata and ``no_new_id`` so the
  same arguments give a byte-identical file.
* Facts are recorded at the moment their block is placed on a page, so the gold
  page of every planted fact is known by construction.  Questions are then chosen
  from those facts at 40 evenly spaced document depths, using the same 40 question
  slots (type + template) for every tier.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pymupdf

GENERATOR_VERSION = "1.0"
TIERS = (100, 250, 1000, 2500, 5000, 10000)
N_QUESTIONS = 40

# --------------------------------------------------------------------------------------
# Page geometry (A4, points; y grows downwards in this module and is flipped on output)
# --------------------------------------------------------------------------------------
PAGE_W, PAGE_H = 595.0, 842.0
MARGIN_X = 62.0
BODY_W = PAGE_W - 2 * MARGIN_X
BODY_TOP = 76.0
BODY_BOTTOM = 784.0
COL_GAP = 18.0
HEADER_Y = 44.0
FOOTER_Y = 814.0
SCAN_DPI = 150
SCAN_JPEG_QUALITY = 55
#: Pages per in-memory part.  ``Document.new_page`` costs O(pages already in the document),
#: so a single 10,000-page document takes ~2 minutes to build; parts joined with
#: ``insert_pdf`` keep it to seconds.
PART_PAGES = 500

# --------------------------------------------------------------------------------------
# The fictional publication
# --------------------------------------------------------------------------------------
COUNTRY = "Republic of Veloria"
AGENCY = "National Statistics and Development Agency"
AGENCY_SHORT = "NSDA"
PUB_TITLE = "Statistical and Administrative Compendium"
FIRST_YEAR, LAST_YEAR = 2010, 2023
CENSUS_YEAR = 2020
PUB_SPAN = f"{FIRST_YEAR}-{LAST_YEAR}"
RUNNING_TITLE = f"{AGENCY_SHORT} {PUB_TITLE} {PUB_SPAN}"
CURRENCY = "VRD"
#: Years that never appear in the content; used for out-of-range unanswerable questions.
ABSENT_YEARS = (2026, 2027, 2028, 2029)

#: Scanned-paper tone curve: ink stays dark, white paper turns light grey.
_PAPER_LUT = np.clip(np.arange(256) * 0.92 + 14, 0, 255).astype(np.uint8)

FONTS = {  # resource name -> (BaseFont, PyMuPDF short name used for metrics)
    "TR": ("Times-Roman", "tiro"),
    "TB": ("Times-Bold", "tibo"),
    "TI": ("Times-Italic", "tiit"),
    "HR": ("Helvetica", "helv"),
    "HB": ("Helvetica-Bold", "hebo"),
    "HI": ("Helvetica-Oblique", "heit"),
    "CR": ("Courier", "cour"),
    "CB": ("Courier-Bold", "cobo"),
}
_WIDTHS = {
    key: [
        pymupdf.get_text_length(chr(c), fontname=short, fontsize=1.0) if 32 <= c < 127 else 0.5
        for c in range(128)
    ]
    for key, (_, short) in FONTS.items()
}

# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------
ONSETS = (
    "Ar", "Bel", "Bran", "Cal", "Dun", "Eld", "Fen", "Gar", "Hal", "Ith", "Kes", "Lor",
    "Mar", "Nor", "Orm", "Pell", "Quen", "Ros", "Sel", "Tor", "Ul", "Vel", "Wyn", "Yar",
    "Zel", "Ash", "Brin", "Cor", "Dal", "Ev", "Fal", "Gil", "Har", "Jun", "Kel", "Lam",
    "Mor", "Nesh", "Ost", "Pry", "Rav", "Sten", "Tal", "Var", "Wex", "Aud", "Bex", "Cran",
)
MIDS = ("", "a", "e", "i", "o", "en", "an", "or", "il", "ar", "es")
ENDS = (
    "mont", "dale", "ford", "wick", "stead", "more", "holt", "bury", "haven", "field",
    "brook", "ton", "ley", "mere", "gate", "wood", "crest", "fall", "marsh", "ridge",
    "port", "by", "worth", "combe", "thorpe", "minster", "hurst", "wold", "ness", "stow",
    "bridge", "cliffe", "vale", "hollow",
)
MODIFIERS = ("Upper", "Lower", "East", "West", "North", "South", "Old", "New", "Great", "Little")
DIRECTIONS = (
    "north", "north-east", "east", "south-east", "south", "south-west", "west",
    "north-west", "centre",
)
GIVEN = (
    "Amara", "Bastian", "Celia", "Dario", "Elin", "Farid", "Greta", "Hollis", "Ines",
    "Jonas", "Kaia", "Leon", "Mirela", "Nadim", "Orla", "Pavel", "Quilla", "Rafael",
    "Signe", "Tomas", "Una", "Viktor", "Wren", "Xenia", "Yusuf", "Zora", "Anouk", "Bram",
    "Cyra", "Dmitri", "Edda", "Fintan", "Gaia", "Henrik", "Ilse", "Joaquin", "Keira",
    "Lukas", "Maren", "Niko", "Odile", "Piet", "Rosalind", "Stellan", "Tamsin", "Ulrich",
    "Vesna", "Willem", "Ysolde", "Zeno", "Adaeze", "Bernard", "Chiara", "Dunja", "Emil",
    "Femke", "Goran", "Halima", "Ivo", "Juna",
)
FAMILY = (
    "Okonjo", "Varga", "Lindqvist", "Moreau", "Castellan", "Haddad", "Petrov", "Asante",
    "Kovac", "Brennan", "Sato", "Ferreira", "Novak", "Halloran", "Iyer", "Mendel",
    "Draganov", "Oyelaran", "Rasku", "Sorensen", "Tavares", "Ulven", "Wexler", "Yilmaz",
    "Zabala", "Achterberg", "Bergstrom", "Caldera", "Dunmore", "Eriksen", "Falk", "Gruber",
    "Hasselt", "Ivanova", "Jovanic", "Karam", "Lachance", "Marsh", "Nyberg", "Ostrander",
    "Pellegrini", "Quist", "Rahman", "Strand", "Thorne", "Urquhart", "Valdez", "Whitlock",
    "Yaro", "Zelenko", "Abara", "Brandt", "Corvin", "Delacroix", "Esposito", "Fenwick",
    "Gallo", "Hartmann", "Ilunga", "Jensen", "Kowalczyk", "Lund", "Mwangi", "Nakamura",
    "Ochieng", "Pereira", "Quaye", "Rossi", "Szabo", "Tanaka", "Uddin", "Vasquez", "Weiss",
    "Xavier", "Yamada", "Zimmer", "Alvarez", "Bakker", "Chowdhury", "Demir",
)
TITLES = ("Dr.", "Eng.", "Prof.")
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)

# (type, capacity unit, capacity range, remarks)
FACILITY_TYPES = (
    ("District Hospital", "beds", (60, 420), (
        "Its emergency department operates around the clock and receives referrals from the surrounding health centres.",
        "An oxygen generation plant was installed on the hospital grounds during the reporting period.",
    )),
    ("Health Centre", "beds", (8, 48), (
        "The centre runs weekly outreach clinics for settlements more than ten kilometres away.",
        "Maternity services at the centre were expanded after the addition of a second delivery room.",
    )),
    ("Water Treatment Works", "cubic metres per day", (900, 46000), (
        "Raw water is abstracted from the river intake and treated by rapid sand filtration.",
        "Treated water is distributed through a network of elevated storage tanks.",
    )),
    ("Vocational Training Institute", "trainee places", (120, 2400), (
        "Courses are offered in construction trades, electrical installation and agro-processing.",
        "The institute operates evening classes for adult learners in the district seat.",
    )),
    ("Grain Storage Depot", "tonnes of storage", (600, 32000), (
        "The depot is used by the national food reserve for strategic grain stocks.",
        "Fumigation and moisture testing are carried out on every consignment received.",
    )),
    ("Pumping Station", "litres per second", (25, 900), (
        "The station lifts water to the upland reservoir that supplies the eastern wards.",
        "Standby diesel generators allow operation during grid outages.",
    )),
    ("Agricultural Research Station", "hectares of trial plots", (15, 640), (
        "Field trials focus on drought-tolerant sorghum and improved bean varieties.",
        "The station also maintains a seed multiplication unit for smallholder cooperatives.",
    )),
    ("Public Library", "volumes", (4000, 150000), (
        "The library hosts the district archive and a public reading room.",
        "A mobile book service reaches schools in the outlying wards each term.",
    )),
    ("Solar Mini-Grid", "kilowatts", (60, 2400), (
        "The mini-grid supplies households and small enterprises that are not connected to the national grid.",
        "Battery storage allows the system to supply evening peak demand.",
    )),
    ("Veterinary Clinic", "animal treatment bays", (6, 60), (
        "The clinic coordinates vaccination campaigns against livestock diseases.",
        "Mobile teams from the clinic visit pastoral settlements during the dry season.",
    )),
)
FACILITY_TYPE_NAMES = tuple(t[0] for t in FACILITY_TYPES)

PROGRAMMES = (
    ("Rural Infrastructure Programme", "RIP"),
    ("Water and Sanitation Programme", "WSP"),
    ("Primary Health Strengthening Programme", "PHS"),
    ("Basic Education Access Programme", "BEA"),
    ("Agricultural Resilience Programme", "ARP"),
    ("Renewable Energy Access Programme", "REA"),
    ("Urban Services Programme", "USP"),
    ("Digital Registry Programme", "DRP"),
)
ACTIVITIES = (
    "financed the rehabilitation of",
    "funded an extension to",
    "paid for new equipment at",
    "supported the solar retrofit of",
    "financed flood-protection works at",
    "funded the digitisation of records at",
    "financed a new access road to",
    "supported staff housing construction at",
)

# (metric phrase, initial range, yearly drift range)
SERIES_METRICS = (
    ("household access to piped water", (35.0, 80.0), (0.5, 2.5)),
    ("measles immunisation coverage", (70.0, 90.0), (0.2, 1.2)),
    ("adult literacy rate", (55.0, 90.0), (0.2, 1.0)),
    ("registered unemployment rate", (4.0, 15.0), (-0.6, 0.4)),
    ("secondary school completion rate", (30.0, 70.0), (0.5, 2.0)),
    ("household electrification rate", (40.0, 85.0), (0.8, 2.5)),
    ("share of births attended by skilled personnel", (60.0, 92.0), (0.3, 1.5)),
)

# (caption title, question template, value range)
WARD_TABLES = (
    ("Outpatient visits", "how many outpatient visits were recorded in {row} in {col}", (900, 60000)),
    ("Pupils enrolled in primary schools", "how many pupils were enrolled in primary schools in {row} in {col}", (300, 18000)),
    ("Households connected to piped water", "how many households in {row} were connected to piped water in {col}", (150, 12000)),
    ("Grain procured by public depots (tonnes)", "how many tonnes of grain did public depots procure in {row} in {col}", (80, 9000)),
    ("Registered small enterprises", "how many small enterprises were registered in {row} in {col}", (40, 3800)),
    ("Electricity consumption (MWh)", "how many megawatt-hours of electricity were consumed in {row} in {col}", (200, 25000)),
    ("Births registered", "how many births were registered in {row} in {col}", (60, 5200)),
)
REGION_TABLE_HEADER = ("District", "Households", "Registered voters", "Revenue (VRD m)", "Paved roads (km)")
REGION_TABLE_Q = (
    None,
    "how many households were there in {row}",
    "how many registered voters did {row} have",
    "what local revenue, in VRD million, did {row} collect",
    "how many kilometres of paved road did {row} have",
)

COMMENTARY = (
    "Rainfall in the {y} growing season was {p}% below the long-term average, which delayed planting in several wards.",
    "Road connectivity improved after the resurfacing of {k} kilometres of feeder roads between {y} and {y2}.",
    "District officials report that seasonal migration continues to affect school attendance in the upland wards.",
    "A joint review with the regional office in {y} recommended consolidating the smaller service points.",
    "Local revenue collection rose by {p}% in {y}, largely owing to the revaluation of market stall licences.",
    "The district took part in the national household survey in {y}, with a response rate of {q}%.",
    "Flooding in {y} damaged {k} dwellings and temporarily closed two rural health posts.",
    "Mobile network coverage reached {q}% of settlements by {y}, according to the regulator's returns.",
    "The district council approved a revised land-use plan in {y}, reserving {k} hectares for light industry.",
    "Administrative data for the most remote settlements are collected each quarter by outreach teams.",
    "Staff vacancies remain concentrated in specialist health and secondary teaching posts.",
    "A district statistical unit was set up in {y} to coordinate routine administrative returns.",
    "Market prices for staple cereals were broadly stable in {y}, after a sharp rise in the previous season.",
    "The share of women among elected ward councillors rose to {q}% following the {y} local elections.",
    "Several wards report difficulties in recruiting extension workers for livestock services.",
    "A cadastral survey completed in {y} registered {k} previously undocumented land parcels.",
    "Bus services between the district seat and the regional capital were extended to weekends in {y}.",
    "Electricity supply interruptions averaged {k} hours per month in {y}, mostly during the rainy season.",
)
SERIES_OPEN = (
    "Survey estimates of {metric} in {d} District are available for {y0}-{yn}.",
    "The annual household survey has tracked {metric} in {d} District since {y0}.",
    "Between {y0} and {yn}, {metric} in {d} District was measured every year.",
)
SERIES_STEP = (
    "In {y}, the rate stood at {v}%.",
    "The {y} estimate was {v}%.",
    "By {y}, it had {moved} to {v}%.",
    "The survey for {y} recorded {v}%.",
    "For {y}, the figure was {v}%.",
)
SERIES_CLOSE = (
    "District planners attribute the trend mainly to changes in service coverage in the outlying wards.",
    "Year-to-year movements of less than one point are within the survey's margin of error.",
    "The estimates are weighted to the {c} census population.",
    "Comparable figures for neighbouring districts are presented in the same section of each profile.",
)
REGION_GENERIC = (
    "Agriculture remains the main source of employment in the region, although services and construction have grown steadily since {y}.",
    "The regional climate is marked by a single long rainy season; average annual rainfall varies from {k} to {k2} millimetres across the districts.",
    "The trunk road linking the regional capital to the national highway was upgraded between {y} and {y2}, cutting travel times for freight considerably.",
    "Population growth has been fastest in the districts closest to the regional capital, where the share of urban residents now exceeds {q}%.",
    "The regional directorate maintains a register of public facilities that is updated after each inspection round.",
    "Development partners finance a large share of capital investment in the region, mainly through national programmes administered by line ministries.",
    "Administrative boundaries in the region were last revised in {y}, when two former sub-districts were merged.",
    "The region's statistical returns are compiled from district administrative records and validated against survey data where available.",
)
NOTES_TOPICS = (
    "Sources of administrative data", "Census reference dates", "Survey design and weighting",
    "Treatment of provisional figures", "Facility registers", "Project financing data",
    "Rounding and totals", "Geographic classifications", "Revisions policy",
    "Comparability over time", "Confidentiality and disclosure control", "Quality assurance",
)
NOTES_SENTENCES = (
    "Administrative returns are submitted by district offices to the regional directorates and consolidated by the {a} head office.",
    "Figures marked as provisional are revised in the following edition once reconciliation with treasury records is complete.",
    "Totals may not equal the sum of their components because of rounding.",
    "Facility capacities refer to the design capacity recorded in the national facility register at the end of the reference year.",
    "Survey estimates are weighted to the {c} census population and are subject to sampling error.",
    "Monetary values are expressed in current {cur} and are not adjusted for inflation.",
    "Ward boundaries follow the delimitation in force at the time of the {c} census.",
    "Project budgets refer to the amount approved at appraisal and exclude later supplementary allocations.",
    "Where a district did not submit a return, the corresponding cells are left blank rather than imputed.",
    "Breaks in series are documented in the technical notes accompanying each regional chapter.",
    "The compendium is compiled under the Statistics Act and follows the agency's dissemination standards.",
    "Users are advised to consult the regional chapters for definitions specific to each indicator.",
)
GLOSSARY = (
    ("Administrative return", "A periodic report submitted by a public office, used as a source of official statistics."),
    ("Approved budget", "The amount authorised for a project at appraisal, before any supplementary allocation."),
    ("Census population", "The number of persons enumerated in the national census on the reference night."),
    ("Commissioning", "The date on which a facility was formally handed over and began regular operation."),
    ("Design capacity", "The maximum output or number of users a facility was built to serve."),
    ("District", "The second-level administrative unit, headed by a District Coordinator."),
    ("District Coordinator", "The senior civil servant responsible for the district administration."),
    ("District seat", "The town in which the district administration has its offices."),
    ("Electrification rate", "The share of households with a connection to the grid or a mini-grid."),
    ("Facility register", "The national list of public facilities maintained by the agency."),
    ("Feeder road", "A rural road linking settlements to the main road network."),
    ("Household", "A group of persons who share living quarters and common arrangements for food."),
    ("Immunisation coverage", "The share of children aged 12-23 months who have received the vaccine."),
    ("Inspection certificate", "A document issued after a facility inspection, recording the findings."),
    ("Literacy rate", "The share of persons aged 15 and over who can read and write a short statement."),
    ("Local revenue", "Taxes, fees and charges collected by a district administration."),
    ("Mini-grid", "An electricity network serving a local area, not connected to the national grid."),
    ("Outpatient visit", "A visit to a health facility that does not involve an overnight stay."),
    ("Paved road", "A road with a bituminous or concrete surface."),
    ("Programme", "A national framework that finances related projects across several districts."),
    ("Project code", "A unique identifier assigned to a project, made up of a programme prefix, a number and a suffix letter."),
    ("Provisional figure", "An estimate published before all source data have been received and reconciled."),
    ("Pupil-teacher ratio", "The number of pupils enrolled divided by the number of teachers."),
    ("Region", "The first-level administrative unit, grouping several districts."),
    ("Regional capital", "The town in which the regional assembly and directorates are located."),
    ("Regional Director", "The head of the agency's regional directorate."),
    ("Registered voter", "A person entered on the electoral roll for the most recent local election."),
    ("Response rate", "The share of sampled households that completed the survey interview."),
    ("Settlement", "A recognised village, town or other inhabited locality."),
    ("Skilled birth attendance", "Births attended by a doctor, nurse or midwife."),
    ("Small enterprise", "A business with fewer than ten employees registered with the district."),
    ("Unemployment rate", "The share of the labour force without work and seeking work."),
    ("Ward", "A subdivision of a district used for local elections and service planning."),
    ("Water treatment works", "A facility that treats raw water to drinking-water standards."),
)

HINDI_LINES = (
    "{d} जिले में वर्ष {y} के दौरान {n} आंगनवाड़ी केंद्र संचालित थे।",
    "जिला मुख्यालय {s} में {m} सामुदायिक स्वास्थ्य कार्यकर्ता तैनात हैं।",
    "यह सारांश राष्ट्रीय सांख्यिकी एवं विकास एजेंसी द्वारा क्षेत्रीय पाठकों के लिए तैयार किया गया है।",
    "विस्तृत आँकड़े और तालिकाएँ इस अध्याय के अंग्रेज़ी खंड में उपलब्ध हैं।",
    "जिले की प्रमुख सेवाओं में स्वास्थ्य, शिक्षा, पेयजल और ग्रामीण सड़कें शामिल हैं।",
)


# --------------------------------------------------------------------------------------
# Low-level PDF writing
# --------------------------------------------------------------------------------------
_WORD_CACHE: dict[tuple[str, str], float] = {}


def text_width(s: str, font: str, size: float) -> float:
    w = _WIDTHS[font]
    return size * sum(w[o] if (o := ord(ch)) < 128 else 0.5 for ch in s)


def _word_width(word: str, font: str) -> float:
    """Width of ``word`` at size 1, memoised (the vocabulary is small and repetitive)."""
    key = (word, font)
    width = _WORD_CACHE.get(key)
    if width is None:
        if len(_WORD_CACHE) > 200_000:
            _WORD_CACHE.clear()
        width = _WORD_CACHE[key] = text_width(word, font, 1.0)
    return width


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    """Greedy word wrap; never breaks inside a word."""
    lines: list[str] = []
    cur: list[str] = []
    cur_w = 0.0
    space = text_width(" ", font, size)
    for word in text.split():
        ww = _word_width(word, font) * size
        if cur and cur_w + space + ww > width:
            lines.append(" ".join(cur))
            cur, cur_w = [word], ww
        else:
            cur_w = ww if not cur else cur_w + space + ww
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Canvas:
    """Collects raw PDF content-stream operators for one page (top-left coordinates)."""

    __slots__ = ("ops",)

    def __init__(self) -> None:
        self.ops: list[str] = []

    def text(self, x: float, y: float, s: str, font: str = "TR", size: float = 10.0,
             gray: float = 0.0) -> None:
        """Draw ``s`` with its baseline at ``y``."""
        op = f"BT /{font} {size:g} Tf 1 0 0 1 {x:.2f} {PAGE_H - y:.2f} Tm ({_esc(s)}) Tj ET"
        self.ops.append(f"q {gray:.2f} g {op} Q" if gray else op)

    def text_right(self, x_right: float, y: float, s: str, font: str, size: float,
                   gray: float = 0.0) -> None:
        self.text(x_right - text_width(s, font, size), y, s, font, size, gray)

    def line(self, x0: float, y0: float, x1: float, y1: float, width: float = 0.5,
             gray: float = 0.0) -> None:
        self.ops.append(
            f"{width:g} w {gray:.2f} G {x0:.2f} {PAGE_H - y0:.2f} m {x1:.2f} {PAGE_H - y1:.2f} l S"
        )

    def fill_rect(self, x: float, y: float, w: float, h: float, gray: float) -> None:
        self.ops.append(f"q {gray:.2f} g {x:.2f} {PAGE_H - y - h:.2f} {w:.2f} {h:.2f} re f Q")

    def stroke_rect(self, x: float, y: float, w: float, h: float, width: float = 1.0) -> None:
        self.ops.append(f"{width:g} w 0 G {x:.2f} {PAGE_H - y - h:.2f} {w:.2f} {h:.2f} re S")


def install_fonts(doc: pymupdf.Document) -> int:
    """Create the Base-14 font objects and one shared /Resources dict; return its xref."""
    entries = []
    for key, (base, _) in FONTS.items():
        xref = doc.get_new_xref()
        doc.update_object(
            xref, f"<< /Type /Font /Subtype /Type1 /BaseFont /{base} /Encoding /WinAnsiEncoding >>"
        )
        entries.append(f"/{key} {xref} 0 R")
    res = doc.get_new_xref()
    doc.update_object(res, "<< /Font << " + " ".join(entries) + " >> >>")
    return res


def commit(doc: pymupdf.Document, page_xref: int, canvas: Canvas, res_xref: int) -> None:
    data = "\n".join(canvas.ops).encode("latin-1")
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(stream, data)
    doc.xref_set_key(page_xref, "Resources", f"{res_xref} 0 R")
    doc.xref_set_key(page_xref, "Contents", f"{stream} 0 R")


def draw_running(canvas: Canvas, right: str, page_no: int) -> None:
    """Running header (publication + chapter) and footer ("Page N")."""
    canvas.text(MARGIN_X, HEADER_Y, RUNNING_TITLE, "HR", 7.5, gray=0.35)
    if right:
        canvas.text_right(PAGE_W - MARGIN_X, HEADER_Y, right, "HI", 7.5, gray=0.35)
    canvas.line(MARGIN_X, HEADER_Y + 6, PAGE_W - MARGIN_X, HEADER_Y + 6, 0.4, 0.5)
    footer = f"Page {page_no}"
    canvas.text((PAGE_W - text_width(footer, "HR", 8)) / 2, FOOTER_Y, footer, "HR", 8, gray=0.25)


# --------------------------------------------------------------------------------------
# Layout engine
# --------------------------------------------------------------------------------------
class PhaseFull(Exception):
    """Raised when a new page would exceed the page budget of the current phase."""


class Writer:
    """Flows headings, paragraphs and tables onto pages and records planted facts."""

    def __init__(self, part_pages: int = PART_PAGES) -> None:
        self.part_pages = part_pages
        self.parts: list[bytes] = []
        self.doc: pymupdf.Document | None = None
        self.res = 0
        self.pno = -1
        self.limit = -1
        self.canvas: Canvas | None = None
        self.page_xref = 0
        self.y = BODY_TOP
        self.footnotes: list[tuple[int, list[str]]] = []
        self.fn_height = 0.0
        self.fn_counter = 0
        self.header = ""
        self.kinds: list[set[str]] = []
        self.deferred: dict[int, Canvas] = {}
        self.defer_next = False
        self.special: dict[int, object] = {}
        self.facts: list[dict] = []
        self.outline: list[list] = []

    # ---- pages -------------------------------------------------------------------
    def _new_page(self) -> pymupdf.Page:
        if self.doc is None or self.doc.page_count >= self.part_pages:
            self._flush_part()
            self.doc = pymupdf.open()
            self.res = install_fonts(self.doc)
        return self.doc.new_page(width=PAGE_W, height=PAGE_H)

    def _flush_part(self) -> None:
        if self.doc is not None:
            self.parts.append(self.doc.tobytes(garbage=1, deflate=True, no_new_id=True))
            self.doc.close()
            self.doc = None

    def assemble(self) -> pymupdf.Document:
        """Finish the last page and join all parts into one document."""
        self.finish_page()
        self._flush_part()
        final = pymupdf.open()
        for data in self.parts:
            part = pymupdf.open("pdf", data)
            final.insert_pdf(part)
            part.close()
        self.parts = []
        return final

    def start_page(self, kind: str = "text") -> None:
        self.finish_page()
        while True:
            nxt = self.pno + 1
            if nxt > self.limit:
                raise PhaseFull
            if nxt in self.special:
                self.pno = nxt
                page = self._new_page()
                self.kinds.append(set())
                self.special.pop(nxt)(self, page)
                continue
            break
        self.pno = nxt
        page = self._new_page()
        self.page_xref = page.xref
        self.canvas = Canvas()
        self.kinds.append({kind})
        self.y = BODY_TOP
        self.footnotes = []
        self.fn_height = 0.0

    def finish_page(self) -> None:
        c = self.canvas
        if c is None:
            return
        if self.footnotes:
            y = BODY_BOTTOM - self.fn_height + 6
            c.line(MARGIN_X, y, MARGIN_X + 120, y, 0.4)
            y += 10
            for number, lines in self.footnotes:
                c.text(MARGIN_X, y - 2, str(number), "TR", 5.5)
                for ln in lines:
                    c.text(MARGIN_X + 6, y, ln, "TR", 7.5)
                    y += 9.5
            self.kinds[self.pno].add("footnote")
        draw_running(c, self.header, self.pno + 1)
        if self.defer_next:
            self.deferred[self.pno] = c
            self.defer_next = False
        else:
            commit(self.doc, self.page_xref, c, self.res)
        self.canvas = None

    def avail(self) -> float:
        return BODY_BOTTOM - self.fn_height - self.y

    def ensure(self, h: float) -> None:
        if self.canvas is None or h > self.avail():
            self.start_page()

    def register(self, facts) -> None:
        for f in facts:
            f["page"] = self.pno + 1
            self.facts.append(f)

    # ---- blocks ------------------------------------------------------------------
    def heading(self, text: str, level: int, facts=(), outline: bool = False) -> None:
        font, size, before, after = ("HB", 13.0, 10.0, 6.0) if level == 2 else ("HB", 10.5, 6.0, 4.0)
        lines = wrap(text, font, size, BODY_W)
        h = len(lines) * size * 1.25
        if self.canvas is not None and self.y > BODY_TOP:
            self.y += before
        self.ensure(h + 55)  # keep with the next few lines
        for i, ln in enumerate(lines):
            self.canvas.text(MARGIN_X, self.y + size + i * size * 1.25, ln, font, size)
        self.y += h + after
        if outline:
            self.outline.append([level, text, self.pno + 1])
        self.register(facts)

    def paragraph(self, text: str, *, font: str = "TR", size: float = 10.0, indent: float = 0.0,
                  space_after: float = 5.0, facts=(), footnote: str | None = None) -> None:
        lead = size * 1.32
        lines = wrap(text, font, size, BODY_W - indent)
        h = len(lines) * lead
        fn_lines = wrap(footnote, "TR", 7.5, BODY_W - 6) if footnote else []

        def fn_extra() -> float:  # footnote lines, plus the separator rule for the first one
            return len(fn_lines) * 9.5 + (0.0 if self.footnotes else 16.0) if footnote else 0.0

        if self.canvas is None or h + fn_extra() > self.avail():
            self.start_page()
        extra = fn_extra()
        x = MARGIN_X + indent
        for i, ln in enumerate(lines):
            self.canvas.text(x, self.y + size + i * lead, ln, font, size)
        if footnote:
            self.fn_counter += 1
            last_base = self.y + size + (len(lines) - 1) * lead
            self.canvas.text(x + text_width(lines[-1], font, size) + 0.8, last_base - 3.5,
                             str(self.fn_counter), font, 6.0)
            self.fn_height += extra
            self.footnotes.append((self.fn_counter, fn_lines))
        self.y += h + space_after
        self.register(facts)

    def table(self, caption: str, header, rows, *, facts=(), note: str | None = None,
              first_col_w: float = 150.0, size: float = 8.0) -> None:
        row_h = 14.0
        cap_lines = wrap(caption, "HB", 9.0, BODY_W)
        h = len(cap_lines) * 11.5 + 4 + row_h * (len(rows) + 1) + (13 if note else 0) + 10
        self.ensure(h)
        c = self.canvas
        y = self.y
        for ln in cap_lines:
            c.text(MARGIN_X, y + 9, ln, "HB", 9.0)
            y += 11.5
        y += 4
        n_cols = len(header)
        widths = [first_col_w] + [(BODY_W - first_col_w) / (n_cols - 1)] * (n_cols - 1)
        xs = [MARGIN_X]
        for w in widths:
            xs.append(xs[-1] + w)
        n_rows = len(rows) + 1
        c.fill_rect(MARGIN_X, y, BODY_W, row_h, 0.88)
        for r in range(n_rows + 1):
            c.line(MARGIN_X, y + r * row_h, MARGIN_X + BODY_W, y + r * row_h, 0.5)
        for x in xs:
            c.line(x, y, x, y + n_rows * row_h, 0.5)
        for r, cells in enumerate([list(header)] + [list(row) for row in rows]):
            base = y + r * row_h + 10
            font = "HB" if r == 0 else "HR"
            for k, cell in enumerate(cells):
                s = size
                while s > 5.5 and text_width(cell, font, s) > widths[k] - 6:
                    s -= 0.25
                if k == 0:
                    c.text(xs[0] + 3, base, cell, font, s)
                else:
                    c.text_right(xs[k + 1] - 3, base, cell, font, s)
        y += n_rows * row_h
        if note:
            c.text(MARGIN_X, y + 10, note, "HI", 7.0, gray=0.2)
        self.y += h
        self.kinds[self.pno].add("table")
        self.register(facts)

    def two_column(self, items, *, multi_page: bool = False, size: float = 9.5) -> None:
        """Flow (text, facts, bold) items down the left then the right column."""
        colw = (BODY_W - COL_GAP) / 2
        lead = size * 1.3
        top, col, y = self.y, 0, self.y
        self.kinds[self.pno].add("two_column")
        laid = []
        for text, facts, bold in items:
            font = "TB" if bold else "TR"
            lines = wrap(text, font, size, colw)
            laid.append((lines, facts, font, len(lines) * lead + (1 if bold else 5)))
        # single-page blocks are balanced: switch column once half the material is placed
        half = sum(it[3] for it in laid) / 2 if not multi_page else float("inf")
        for lines, facts, font, h in laid:
            if col == 0 and y > top and y - top + h / 2 > half:
                col, y = 1, top
            while y + h > BODY_BOTTOM - self.fn_height:
                if col == 0:
                    col, y = 1, top
                    continue
                if not multi_page:
                    self.y = BODY_BOTTOM
                    return
                self.start_page("appendix")
                self.kinds[self.pno].add("two_column")
                top, col, y = self.y, 0, self.y
            x = MARGIN_X + col * (colw + COL_GAP)
            for i, ln in enumerate(lines):
                self.canvas.text(x, y + size + i * lead, ln, font, size)
            y += h
            self.register(facts)
        self.y = max(y, self.y) if col == 0 else BODY_BOTTOM


# --------------------------------------------------------------------------------------
# Content generation
# --------------------------------------------------------------------------------------
def _fmt_int(n: int) -> str:
    return f"{n:,}"


class Compendium:
    """Builds one tier of the corpus: the PDF plus its fact registry."""

    def __init__(self, n_pages: int, seed: int, font_path: Path | None = None) -> None:
        if n_pages < 20:
            raise ValueError("documents shorter than 20 pages are not supported")
        self.n = n_pages
        self.seed = seed
        self.rng = random.Random(f"scale-corpus:{seed}:{n_pages}")
        self.np_rng = np.random.default_rng([seed, n_pages])
        self.font_path = font_path
        names = sorted({o + m + e for o in ONSETS for m in MIDS for e in ENDS
                        if 5 <= len(o + m + e) <= 11})
        self.rng.shuffle(names)
        self.reserved_fakes = names[:40]          # never used anywhere in the document
        self.all_names = names[40:]               # sampled with replacement (towns, wards)
        self._name_iter = iter(self.all_names)    # consumed without replacement (districts)
        self.district_names: set[str] = set()
        self.region_names: set[str] = set()
        self.facility_names: set[str] = set()
        self.project_keys: set[tuple[str, int]] = set()
        self.cert_numbers: set[int] = set()
        self.recent_facilities: list[dict] = []
        self.scanned_facilities: set[str] = set()
        self.scan_pages: list[int] = []
        self.hindi_pages: list[int] = []
        self.counts = Counter()
        self.chapter = 0
        self.annex = 0
        self.last_district: dict | None = None
        self._noise: np.ndarray | None = None

    # ---- vocab helpers -------------------------------------------------------------
    def unique_name(self) -> str:
        name = next(self._name_iter, None)
        if name is None:
            raise RuntimeError("district/region name pool exhausted; extend ONSETS/MIDS/ENDS")
        return name

    def any_name(self) -> str:
        return self.rng.choice(self.all_names)

    def person(self) -> tuple[str, str]:
        return self.rng.choice(TITLES), f"{self.rng.choice(GIVEN)} {self.rng.choice(FAMILY)}"

    # ---- document --------------------------------------------------------------------
    def build(self, pdf_path: Path, part_pages: int = PART_PAGES) -> dict:
        n = self.n
        w = Writer(part_pages)
        toc_pages = 1 if n <= 250 else 2
        app_pages = 1 if n <= 60 else 2 if n <= 1000 else 3 if n <= 5000 else 4
        body_first, body_last = 1 + toc_pages, n - app_pages - 1
        body_len = body_last - body_first + 1

        n_scan = max(2, round(n * 0.015))
        for i in range(n_scan):
            p = body_first + 2 + int((i + 0.5) * (body_len - 3) / n_scan)
            p += self.rng.randint(-1, 1)
            p = min(max(p, body_first + 2), body_last)
            while p in w.special:
                p += 1
            w.special[p] = self._scan_page
        self.hindi_font = None
        if self.font_path is not None:
            n_hindi = max(1, n // 400)
            self.hindi_font = self.font_path
            for i in range(n_hindi):
                p = body_first + 4 + int((i + 0.37) * (body_len - 5) / n_hindi)
                while p in w.special:
                    p += 1
                if p <= body_last:
                    w.special[p] = self._hindi_page

        # title page
        w.limit = 0
        w.start_page("title")
        self._title_page(w)
        # reserved table-of-contents pages (filled once page numbers are known)
        w.limit = toc_pages
        for _ in range(toc_pages):
            w.start_page("toc")
            w.header = "Contents"
            w.defer_next = True
        w.outline.append([1, "Contents", 2])
        # body
        w.limit = body_last
        try:
            while True:
                self._chapter(w)
        except PhaseFull:
            pass
        # appendices
        w.limit = n - 1
        with contextlib.suppress(PhaseFull):
            self._appendix(w)
        w.finish_page()
        while w.pno + 1 < n:  # safety net: pad with deliberately blank pages
            w.limit = n - 1
            w.start_page("blank")
            w.canvas.text(MARGIN_X, 400, "This page is intentionally left blank.", "TI", 10)
            w.finish_page()
        doc = w.assemble()
        assert doc.page_count == n, (doc.page_count, n)

        self._fill_toc(w, doc)
        doc.set_toc([[1, "Title page", 1]] + w.outline)
        doc.set_metadata({
            "title": f"{PUB_TITLE} of the {COUNTRY}, {PUB_SPAN}",
            "author": AGENCY,
            "subject": "Synthetic long document for RAG scale benchmarking (fictional data)",
            "keywords": "synthetic, benchmark, fictional",
            "creator": f"make_corpus.py {GENERATOR_VERSION}",
            "producer": "PyMuPDF",
            "creationDate": "D:20240315090000Z",
            "modDate": "D:20240315090000Z",
        })
        doc.save(str(pdf_path), garbage=3, deflate=True, no_new_id=True)
        doc.close()
        self.kinds = w.kinds
        self.facts = w.facts
        return {"facts": w.facts, "kinds": w.kinds, "toc_entries": self.toc_entries}

    # ---- front matter ------------------------------------------------------------------
    def _title_page(self, w: Writer) -> None:
        c = w.canvas
        c.text(MARGIN_X, 150, COUNTRY.upper(), "HB", 13, gray=0.3)
        c.text(MARGIN_X, 172, AGENCY, "HR", 12)
        c.line(MARGIN_X, 190, PAGE_W - MARGIN_X, 190, 1.2)
        y = 240
        for ln in wrap(PUB_TITLE, "HB", 28, BODY_W):
            c.text(MARGIN_X, y, ln, "HB", 28)
            y += 34
        c.text(MARGIN_X, y + 6, f"Regions, districts, facilities and programmes, {PUB_SPAN}", "TI", 14)
        c.text(MARGIN_X, y + 40, f"Edition 2024  |  {self.n:,} pages  |  Reference series SC-{self.seed:03d}",
               "HR", 10, gray=0.2)
        blurb = (
            f"This compendium consolidates the administrative returns, facility registers and survey "
            f"estimates compiled by the {AGENCY} for every region and district of the {COUNTRY}. "
            f"Each regional chapter opens with an overview and a summary table, followed by district "
            f"profiles covering population, public facilities, service indicators and development "
            f"projects. Archival inspection records are reproduced as scanned images where the "
            f"original typed documents are the only available source."
        )
        y += 90
        for ln in wrap(blurb, "TR", 11, BODY_W):
            c.text(MARGIN_X, y, ln, "TR", 11)
            y += 15
        c.text(MARGIN_X, 700, f"Published by the {AGENCY_SHORT} Directorate of Statistical Coordination",
               "HR", 9)
        c.text(MARGIN_X, 714, "Synthetic document generated for software benchmarking. All names, places "
               "and figures are fictional.", "HI", 8, gray=0.3)

    def _fill_toc(self, w: Writer, doc: pymupdf.Document) -> None:
        """Print chapters and sections of the first chapters on the reserved pages."""
        entries = [e for e in w.outline if e[0] in (1, 2) and e[1] != "Contents"]
        self.toc_entries: dict[str, int] = {}  # entry text -> TOC page it is printed on
        it = iter(entries)
        res = install_fonts(doc)
        lead = 12.5
        done = False
        for idx, pno in enumerate(sorted(w.deferred)):
            c = w.deferred[pno]
            y = BODY_TOP + 10
            if idx == 0:
                c.text(MARGIN_X, y + 18, "Contents", "HB", 20)
                y += 40
            while not done and y + lead < BODY_BOTTOM - 30:
                e = next(it, None)
                if e is None:
                    done = True
                    break
                level, title, page = e
                font, x = ("HB", MARGIN_X) if level == 1 else ("HR", MARGIN_X + 16)
                if level == 1:
                    y += 4
                num = str(page)
                c.text(x, y + 9, title, font, 9)
                x_dots = x + text_width(title, font, 9) + 4
                x_num = PAGE_W - MARGIN_X - text_width(num, font, 9) - 4
                n_dots = int((x_num - x_dots) / text_width(" .", "HR", 9))
                if n_dots > 0:
                    c.text(x_dots, y + 9, " ." * n_dots, "HR", 9, gray=0.5)
                c.text_right(PAGE_W - MARGIN_X, y + 9, num, font, 9)
                self.toc_entries[title] = pno + 1
                y += lead
            if not done and idx == len(w.deferred) - 1:
                c.text(MARGIN_X, BODY_BOTTOM - 10,
                       "The complete list of sections is available in the document bookmarks.",
                       "TI", 9)
            commit(doc, doc[pno].xref, c, res)

    # ---- chapters ------------------------------------------------------------------------
    def _chapter(self, w: Writer) -> None:
        rng = self.rng
        self.chapter += 1
        ch = self.chapter
        self.table_no = 0
        region = self.unique_name()
        self.region_names.add(region)
        n_d = rng.randint(8, 18)
        districts = [self._new_district(region) for _ in range(n_d)]
        capital = self.any_name()
        title, director = self.person()
        dir_year = rng.randint(2011, 2021)
        area = sum(d["area"] for d in districts)
        w.start_page("chapter_opener")
        w.header = f"Chapter {ch}: {region} Region"
        w.outline.append([1, f"Chapter {ch}: {region} Region", w.pno + 1])
        c = w.canvas
        c.text(MARGIN_X, BODY_TOP + 30, f"CHAPTER {ch}", "HB", 11, gray=0.4)
        c.text(MARGIN_X, BODY_TOP + 62, f"{region} Region", "HB", 24)
        c.line(MARGIN_X, BODY_TOP + 76, PAGE_W - MARGIN_X, BODY_TOP + 76, 1.0)
        w.y = BODY_TOP + 88
        w.paragraph(
            f"This chapter presents administrative and statistical profiles for the {n_d} districts "
            f"of {region} Region, covering population, public facilities, service indicators and "
            f"development projects for {PUB_SPAN}.", font="TI", size=11, space_after=12)
        items = [
            (f"{region} Region lies in the {rng.choice(DIRECTIONS)} of the {COUNTRY} and comprises "
             f"{n_d} districts with a combined area of {_fmt_int(area)} square kilometres. The regional "
             f"capital is {capital}, which hosts the regional assembly and the offices of the "
             f"{AGENCY_SHORT} regional directorate.",
             [{"kind": "region", "region": region, "capital": capital, "chapter": ch}], False),
            (f"The regional directorate has been led by Regional Director {title} {director} since "
             f"{dir_year}. It coordinates the statistical returns of the district administrations and "
             f"publishes a quarterly bulletin of administrative indicators.", [], False),
        ]
        for tpl in rng.sample(REGION_GENERIC, 6):
            y0 = rng.randint(FIRST_YEAR, LAST_YEAR - 3)
            k = rng.randint(400, 900)
            items.append((tpl.format(y=y0, y2=y0 + rng.randint(1, 3), k=k, k2=k + rng.randint(200, 700),
                                     q=rng.randint(20, 60)), [], False))
        w.two_column(items)
        self.counts["chapters"] += 1

        # Regional overview with the region summary table
        w.start_page()
        w.heading(f"{ch}.1 Regional overview", 2, outline=True)
        w.paragraph(
            f"Table {ch}.1 summarises selected indicators for each district of {region} Region. "
            f"Household and voter counts refer to {CENSUS_YEAR}; revenue figures are for fiscal year "
            f"{LAST_YEAR}. District profiles follow in sections {ch}.2 to {ch}.{n_d + 1}.")
        rows = []
        for d in districts:
            rows.append([f"{d['name']} District", _fmt_int(d["households"]), _fmt_int(d["voters"]),
                         f"{d['revenue']:.1f}", _fmt_int(d["roads"])])
        self.table_no += 1
        tid = f"{ch}.{self.table_no}"
        caption = f"Table {tid}: Selected administrative indicators by district, {region} Region, {CENSUS_YEAR}"
        w.table(caption, REGION_TABLE_HEADER, rows, first_col_w=140,
                note=f"Source: {AGENCY_SHORT} district administrative returns.",
                facts=[{"kind": "table", "table": tid, "caption": caption, "header": list(REGION_TABLE_HEADER),
                        "rows": rows, "table_kind": "region", "qtpl": list(REGION_TABLE_Q)}])
        self.counts["tables"] += 1
        for s, d in enumerate(districts, start=2):
            self._district(w, ch, s, d)

    def _new_district(self, region: str) -> dict:
        rng = self.rng
        name = self.unique_name()
        self.district_names.add(name)
        pop = rng.randint(18000, 950000)
        return {
            "name": name, "region": region, "pop": pop,
            "area": rng.randint(120, 9800),
            "households": int(pop / rng.uniform(3.4, 5.6)),
            "voters": int(pop * rng.uniform(0.48, 0.66)),
            "revenue": round(rng.uniform(1.2, 95.0), 1),
            "roads": rng.randint(40, 2400),
        }

    def _new_facility(self, district: str) -> dict:
        rng = self.rng
        while True:
            locality = f"{rng.choice(MODIFIERS)} {self.any_name()}"
            ftype, unit, (lo, hi), remarks = rng.choice(FACILITY_TYPES)
            name = f"{locality} {ftype}"
            if name not in self.facility_names:
                break
        self.facility_names.add(name)
        title, manager = self.person()
        return {"facility": name, "locality": locality, "type": ftype, "unit": unit,
                "capacity": rng.randint(lo, hi),
                "year": rng.randint(2008 if ftype == "Solar Mini-Grid" else 1978, 2021),
                "title": title, "manager": manager, "district": district, "remarks": remarks}

    def _district(self, w: Writer, ch: int, s: int, d: dict) -> None:
        rng = self.rng
        name = d["name"]
        sec = f"{ch}.{s}"
        self.last_district = d
        w.heading(f"{sec} {name} District", 2, outline=True,
                  facts=[{"kind": "section_start", "district": name, "section": sec}])
        self.counts["districts"] += 1
        # -- overview
        seat = self.any_name()
        d["seat"] = seat
        title, coord = self.person()
        cy = rng.randint(2011, 2022)
        pop = _fmt_int(d["pop"])
        area = _fmt_int(d["area"])
        dens = f"{d['pop'] / d['area']:.1f}"
        n_set = rng.randint(14, 260)
        size_phrase = ("one of the larger districts" if d["pop"] > 500000 else
                       "a mid-sized district" if d["pop"] > 150000 else "one of the smaller districts")
        overview = rng.choice((
            f"{name} District covers {area} square kilometres in the {rng.choice(DIRECTIONS)} of "
            f"{d['region']} Region and had a population of {pop} at the {CENSUS_YEAR} census, giving a "
            f"density of {dens} persons per square kilometre. The district administration is seated in "
            f"{seat}, and {n_set} recognised settlements fall within its boundaries.",
            f"With {pop} residents recorded in the {CENSUS_YEAR} census, {name} District is "
            f"{size_phrase} of {d['region']} Region. Its {area} square kilometres are administered from "
            f"{seat}, which also hosts the district court and the land registry office.",
            f"The {CENSUS_YEAR} census enumerated {pop} people in {name} District, spread across {n_set} "
            f"settlements and an area of {area} square kilometres. Administrative functions are "
            f"concentrated in the district seat, {seat}.",
        ))
        coord_s = rng.choice((
            f"Since {cy}, the district administration has been headed by District Coordinator {title} {coord}.",
            f"{title} {coord} was appointed District Coordinator in {cy} and continues to lead the district executive.",
            f"The office of District Coordinator has been held by {title} {coord} since {cy}.",
        ))
        w.paragraph(overview + " " + coord_s, facts=[
            {"kind": "pop", "district": name, "value": d["pop"], "text": pop},
            {"kind": "coordinator", "district": name, "person": coord, "title": title, "year": cy},
        ])
        if rng.random() < 0.8:
            w.paragraph(self._commentary(rng.randint(2, 3)))
        # -- facilities
        w.heading(f"{sec}.1 Services and facilities", 3)
        facilities = [self._new_facility(name) for _ in range(rng.randint(2, 4))]
        for f in facilities:
            fy, fname, cap = f["year"], f["facility"], _fmt_int(f["capacity"])
            unit, t, m = f["unit"], f["title"], f["manager"]
            sent = rng.choice((
                f"The {fname} was commissioned in {fy} and has a capacity of {cap} {unit}. It is managed by {t} {m}.",
                f"Commissioned in {fy}, the {fname} provides {cap} {unit} and is run by {t} {m}, who reports to the district coordinator.",
                f"{t} {m} manages the {fname}, which entered service in {fy} with a capacity of {cap} {unit}.",
                f"Operating since {fy}, the {fname} offers {cap} {unit}; its management is the responsibility of {t} {m}.",
            ))
            w.paragraph(sent + " " + rng.choice(f["remarks"]), facts=[dict(f, kind="fac", remarks=None)])
            self.recent_facilities.append(f)
            self.counts["facilities"] += 1
        del self.recent_facilities[:-60]
        sy = rng.randint(2014, LAST_YEAR)
        prim, secd = rng.randint(12, 420), rng.randint(3, 90)
        ptr = f"{rng.uniform(18.0, 52.0):.1f}"
        school = rng.choice((
            f"In {sy}, {prim} primary schools and {secd} secondary schools were registered within the district, and the pupil-teacher ratio stood at {ptr} to 1.",
            f"School registration data for {sy} list {prim} primary and {secd} secondary schools; the average pupil-teacher ratio was {ptr} to 1.",
            f"The {sy} education return counted {prim} primary schools and {secd} secondary schools, with {ptr} pupils per teacher on average.",
        ))
        footnote = None
        if rng.random() < 0.18:
            school += f" Figures for {sy} are provisional administrative estimates."
            footnote = (f"Provisional figures are revised once the annual reconciliation with the "
                        f"{d['region']} regional treasury is complete; see Appendix B.")
            self.counts["footnotes"] += 1
        w.paragraph(school, footnote=footnote, facts=[
            {"kind": "schools", "district": name, "year": sy, "primary": prim, "secondary": secd, "ptr": ptr}])
        # -- indicator trend (near-duplicate facts: same metric across years and districts)
        w.heading(f"{sec}.2 Indicator trends", 3)
        metric, (lo, hi), (dlo, dhi) = rng.choice(SERIES_METRICS)
        n_years = rng.randint(5, 7)
        y0 = rng.randint(FIRST_YEAR, LAST_YEAR - n_years + 1)
        v = rng.uniform(lo, hi)
        values: dict[int, str] = {}
        prev = None
        for yr in range(y0, y0 + n_years):
            vs = f"{min(max(v, 0.5), 99.5):.1f}"
            while vs in values.values():
                v += 0.3
                vs = f"{min(max(v, 0.5), 99.5):.1f}"
            values[yr] = vs
            v += rng.uniform(dlo, dhi)
        parts = [rng.choice(SERIES_OPEN).format(metric=metric, d=name, y0=y0, yn=y0 + n_years - 1)]
        for yr, vs in values.items():
            tpl = rng.choice(SERIES_STEP)
            if "{moved}" in tpl and prev is None:
                tpl = SERIES_STEP[0]
            moved = "risen" if prev is not None and float(vs) > float(prev) else "fallen"
            parts.append(tpl.format(y=yr, v=vs, moved=moved))
            prev = vs
        parts.append(rng.choice(SERIES_CLOSE).format(c=CENSUS_YEAR))
        w.paragraph(" ".join(parts), facts=[
            {"kind": "series", "district": name, "metric": metric, "values": values}])
        r = rng.random()
        for _ in range(0 if r < 0.5 else 1 if r < 0.92 else 2):
            self._ward_table(w, ch, d)
        # -- projects
        w.heading(f"{sec}.3 Programmes and projects", 3)
        for _ in range(rng.randint(1, 3)):
            pool = self.recent_facilities
            fac = rng.choice(pool[:-len(facilities)] or facilities) if rng.random() < 0.55 else rng.choice(facilities)
            prog, prefix = rng.choice(PROGRAMMES)
            while True:
                num = rng.randint(3000, 9999)
                if (prefix, num) not in self.project_keys:
                    break
            self.project_keys.add((prefix, num))
            code = f"{prefix}-{num}-{rng.choice('ABCDEF')}"
            budget = f"{rng.uniform(0.4, 38.0):.1f}"
            py = rng.randint(2012, LAST_YEAR)
            act = rng.choice(ACTIVITIES)
            fname = fac["facility"]
            sent = rng.choice((
                f"Project {code}, financed under the {prog}, {act} the {fname}. The approved budget was {CURRENCY} {budget} million and the works were completed in {py}.",
                f"Under the {prog}, project {code} {act} the {fname}, with an approved budget of {CURRENCY} {budget} million; the works were completed in {py}.",
                f"With an approved budget of {CURRENCY} {budget} million, project {code} of the {prog} {act} the {fname}. Works were completed in {py}.",
            ))
            w.paragraph(sent, facts=[{"kind": "project", "code": code, "prefix": prefix, "num": num,
                                      "facility": fname, "budget": budget, "year": py,
                                      "district": name}])
            self.counts["projects"] += 1
        if rng.random() < 0.6:
            w.paragraph(self._commentary(rng.randint(2, 4)))

    def _commentary(self, n: int) -> str:
        rng = self.rng
        out = []
        for tpl in rng.sample(COMMENTARY, n):
            y = rng.randint(FIRST_YEAR, LAST_YEAR - 2)
            out.append(tpl.format(y=y, y2=y + rng.randint(1, 2), p=f"{rng.uniform(3, 28):.1f}",
                                  q=f"{rng.uniform(55, 98):.1f}", k=rng.randint(12, 480)))
        return " ".join(out)

    def _ward_table(self, w: Writer, ch: int, d: dict) -> None:
        rng = self.rng
        title, qtpl, (lo, hi) = rng.choice(WARD_TABLES)
        n_years = rng.randint(5, 6)
        y0 = rng.randint(FIRST_YEAR, LAST_YEAR - n_years + 1)
        years = [str(y) for y in range(y0, y0 + n_years)]
        wards: list[str] = []
        n_wards = rng.randint(4, 9)
        while len(wards) < n_wards:
            wn = f"{self.any_name()} Ward"
            if wn not in wards:
                wards.append(wn)
        rows, seen = [], set()
        for wn in wards:
            base = lo * (hi / lo) ** rng.random()
            row = [wn]
            for _ in years:
                val = int(base)
                while val in seen:
                    val += 1
                seen.add(val)
                row.append(_fmt_int(val))
                base *= rng.uniform(0.94, 1.12)
            rows.append(row)
        self.table_no += 1
        tid = f"{ch}.{self.table_no}"
        caption = f"Table {tid}: {title} by ward, {d['name']} District, {years[0]}-{years[-1]}"
        w.table(caption, ["Ward"] + years, rows,
                note=rng.choice((f"Source: {AGENCY_SHORT} district administrative returns.",
                                 "Source: district sector offices; figures for the latest year are provisional.",
                                 "Source: facility registers compiled by the regional directorate.")),
                facts=[{"kind": "table", "table": tid, "caption": caption, "header": ["Ward"] + years,
                        "rows": rows, "table_kind": "ward", "qtpl": qtpl, "district": d["name"]}])
        self.counts["tables"] += 1

    # ---- special pages ----------------------------------------------------------------------
    def _scan_page(self, w: Writer, page: pymupdf.Page) -> None:
        """An archival inspection record, rendered then rasterised: image only, no text layer."""
        rng = self.rng
        fac = next((f for f in reversed(self.recent_facilities)
                    if f["facility"] not in self.scanned_facilities), None)
        if fac is None:
            fac = self._new_facility(self.last_district["name"] if self.last_district else "Central")
            self.recent_facilities.append(fac)
        self.scanned_facilities.add(fac["facility"])
        while True:
            cert = rng.randint(10000, 99999)
            if cert not in self.cert_numbers:
                break
        self.cert_numbers.add(cert)
        day, month, year = rng.randint(1, 28), rng.choice(MONTHS), rng.randint(2012, LAST_YEAR)
        checked = rng.randint(30, 60)
        compliant = rng.randint(checked // 2, checked - 1)
        grade = rng.choice("ABC")
        title, inspector = self.person()
        self.annex += 1
        tmp = pymupdf.open()
        tres = install_fonts(tmp)
        tpage = tmp.new_page(width=PAGE_W, height=PAGE_H)
        c = Canvas()
        draw_running(c, w.header, w.pno + 1)
        y = BODY_TOP + 20
        c.text(MARGIN_X, y, f"ARCHIVAL ANNEX {self.chapter}-{self.annex}: REPRODUCED FROM SCANNED ORIGINAL",
               "CB", 10)
        y += 40
        c.text(MARGIN_X + 90, y, "FACILITY INSPECTION CERTIFICATE", "CB", 15)
        y += 36
        date_s = f"{day} {month} {year}"
        rows = (
            ("Certificate No.", f"IC-{cert}"),
            ("Facility", fac["facility"]),
            ("District", f"{fac['district']} District"),
            ("Date of inspection", date_s),
            ("Inspecting officer", f"{title} {inspector}"),
            ("Safety items checked", str(checked)),
            ("Items found compliant", str(compliant)),
            ("Overall grade", grade),
        )
        for label, value in rows:
            c.text(MARGIN_X, y, f"{label}:", "CR", 12)
            c.text(MARGIN_X + 190, y, value, "CB", 12)
            y += 24
        y += 16
        remarks = (
            f"Remarks: The inspection covered structural safety, fire protection, electrical "
            f"installations and record keeping. {checked - compliant} items require corrective action "
            f"within ninety days. A follow-up visit will be arranged by the district office. The "
            f"original certificate is held in the regional archive."
        )
        for ln in wrap(remarks, "CR", 11, BODY_W):
            c.text(MARGIN_X, y, ln, "CR", 11)
            y += 16
        y += 40
        c.stroke_rect(PAGE_W - MARGIN_X - 170, y, 160, 70, 1.5)
        c.text(PAGE_W - MARGIN_X - 150, y + 30, "INSPECTED", "CB", 16, gray=0.25)
        c.text(PAGE_W - MARGIN_X - 150, y + 52, f"Grade {grade} / {year}", "CR", 11, gray=0.25)
        c.line(MARGIN_X, y + 60, MARGIN_X + 180, y + 60, 0.8)
        c.text(MARGIN_X, y + 74, "Signature of inspecting officer", "CR", 9)
        commit(tmp, tpage.xref, c, tres)
        angle = rng.uniform(-0.8, 0.8)
        zoom = SCAN_DPI / 72
        pix = tpage.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom).prerotate(angle),
                               colorspace=pymupdf.csGRAY, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]
        page.insert_image(page.rect, stream=self._scan_jpeg(arr))
        tmp.close()
        w.kinds[w.pno].add("scanned")
        self.scan_pages.append(w.pno + 1)
        w.facts.append({"kind": "scan", "page": w.pno + 1, "facility": fac["facility"],
                        "district": fac["district"], "cert": f"IC-{cert}", "date": date_s,
                        "day": day, "month": month, "year": year, "checked": checked,
                        "compliant": compliant})

    def _scan_jpeg(self, arr: np.ndarray) -> bytes:
        """Grey paper tone + sensor noise, then JPEG (Pillow when available: ~10x faster)."""
        h, wd = arr.shape
        if self._noise is None or self._noise.shape[0] < h + 64 or self._noise.shape[1] < wd + 64:
            self._noise = self.np_rng.normal(0, 6, (h + 64, wd + 64)).astype(np.int16)
        dy, dx = (int(v) for v in self.np_rng.integers(0, 64, 2))
        tone = _PAPER_LUT[arr].astype(np.int16) + self._noise[dy:dy + h, dx:dx + wd]
        img = np.clip(tone, 0, 255).astype(np.uint8)
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - Pillow ships with sentence-transformers
            out = pymupdf.Pixmap(pymupdf.csGRAY, wd, h, np.ascontiguousarray(img).tobytes(), False)
            return out.tobytes("jpg", jpg_quality=SCAN_JPEG_QUALITY)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=SCAN_JPEG_QUALITY,
                                  dpi=(SCAN_DPI, SCAN_DPI))
        return buf.getvalue()

    def _hindi_page(self, w: Writer, page: pymupdf.Page) -> None:
        """Optional Hindi summary page (needs a Devanagari font; text is shaped by MuPDF)."""
        rng = self.rng
        d = self.last_district or {"name": self.any_name(), "seat": self.any_name()}
        lines = [tpl.format(d=d["name"], y=rng.randint(2015, LAST_YEAR), n=rng.randint(20, 400),
                            s=d.get("seat", d["name"]), m=rng.randint(10, 300)) for tpl in HINDI_LINES]
        html = (f"<h2>जिला सारांश: {d['name']} District</h2>"
                + "".join(f"<p>{ln}</p>" for ln in lines))
        css = (f"@font-face {{font-family: deva; src: url({self.hindi_font.name});}} "
               "* {font-family: deva; font-size: 12pt; line-height: 1.5;}")
        page.insert_htmlbox(pymupdf.Rect(MARGIN_X, BODY_TOP + 10, PAGE_W - MARGIN_X, BODY_BOTTOM),
                            html, css=css, archive=pymupdf.Archive(str(self.hindi_font.parent)))
        page.insert_text((MARGIN_X, HEADER_Y), RUNNING_TITLE, fontname="helv", fontsize=7.5,
                         color=(0.35, 0.35, 0.35))
        if w.header:
            x = PAGE_W - MARGIN_X - pymupdf.get_text_length(w.header, "heit", 7.5)
            page.insert_text((x, HEADER_Y), w.header, fontname="heit", fontsize=7.5,
                             color=(0.35, 0.35, 0.35))
        footer = f"Page {w.pno + 1}"
        page.insert_text(((PAGE_W - pymupdf.get_text_length(footer, "helv", 8)) / 2, FOOTER_Y),
                         footer, fontname="helv", fontsize=8, color=(0.25, 0.25, 0.25))
        w.kinds[w.pno].add("hindi")
        self.hindi_pages.append(w.pno + 1)

    # ---- appendices ---------------------------------------------------------------------------
    def _appendix(self, w: Writer) -> None:
        rng = self.rng
        w.start_page("appendix")
        w.header = "Appendix A: Glossary of terms"
        w.outline.append([1, "Appendix A: Glossary of terms", w.pno + 1])
        w.heading("Appendix A: Glossary of terms", 2)
        items = []
        for term, definition in GLOSSARY:
            items.append((term, [], True))
            items.append((definition, [], False))
        w.two_column(items, multi_page=True, size=8.8)
        w.start_page("appendix")
        w.header = "Appendix B: Methodological notes"
        w.outline.append([1, "Appendix B: Methodological notes", w.pno + 1])
        w.heading("Appendix B: Methodological notes", 2)
        k = 0
        while True:  # fills the remaining appendix pages; stops at PhaseFull
            k += 1
            topic = NOTES_TOPICS[(k - 1) % len(NOTES_TOPICS)]
            w.heading(f"B.{k} {topic}", 3)
            sents = rng.sample(NOTES_SENTENCES, rng.randint(3, 5))
            w.paragraph(" ".join(s.format(a=AGENCY_SHORT, c=CENSUS_YEAR, cur=CURRENCY) for s in sents))
            w.kinds[w.pno].add("appendix")


# --------------------------------------------------------------------------------------
# Question set
# --------------------------------------------------------------------------------------
#: 40 question slots (type, kind); the same slots and templates are used for every tier.
SLOTS = (
    [("lookup", k) for k in ("pop", "coordinator", "fac_commission", "fac_manager",
                             "project_budget", "region_capital")]
    + [("table_cell", "ward")] * 4 + [("table_cell", "region")] * 2
    + [("numeric", "series_change")] * 2 + [("numeric", "pop_sum")] * 2
    + [("numeric", "capacity_diff")]
    + [("multi_hop", "project_manager")] * 2 + [("multi_hop", "project_commission")] * 2
    + [("section_scoped", "schools")] * 2 + [("section_scoped", "ptr")] * 2
    + [("page_ref", "table_on_page")] * 2 + [("page_ref", "section_on_page")] * 2
    + [("ocr", "cert"), ("ocr", "date"), ("ocr", "items")]
    + [("unanswerable", k) for k in ("fake_district", "absent_year", "fake_project", "fake_facility")]
    + [("distractor", "series_value")] * 4
)
assert len(SLOTS) == N_QUESTIONS
#: Filling order: scarce candidate pools first.
_PRIORITY = ("ocr", "multi_hop", "page_ref", "numeric", "unanswerable", "section_scoped",
             "table_cell", "distractor", "lookup")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


class QuestionBuilder:
    def __init__(self, comp: Compendium) -> None:
        self.c = comp
        self.n = comp.n
        self.rng = random.Random(f"scale-questions:{comp.seed}:{comp.n}")
        self.by_kind: dict[str, list[dict]] = {}
        for f in comp.facts:
            self.by_kind.setdefault(f["kind"], []).append(f)
        self.fac_by_name = {f["facility"]: f for f in self.by_kind.get("fac", [])}
        per_page = Counter((f["kind"], f["page"]) for f in comp.facts if f["kind"] in ("table", "section_start"))
        self.tables_on = {p: n for (k, p), n in per_page.items() if k == "table"}
        self.sections_on = {p: n for (k, p), n in per_page.items() if k == "section_start"}

    def depth(self, pages) -> float:
        return round(sum((p - 1) / (self.n - 1) for p in pages) / len(pages), 4)

    # A candidate is (gold pages, entity ids, payload).  ``payload["question"]`` may be a pair
    # of phrasings; the slot index picks one, so a slot words its question the same way in
    # every tier.
    def candidates(self, kind: str) -> list[tuple[list[int], set[str], dict]]:
        K = self.by_kind
        rng = self.rng
        out: list[tuple[list[int], set[str], dict]] = []

        def add(pages, ents, question, reference_answer, answer_keys=(), numeric_answer=None,
                **extra) -> None:
            payload = {"question": question, "reference_answer": reference_answer,
                       "answer_keys": list(answer_keys), "numeric_answer": numeric_answer}
            payload.update(extra)
            out.append((list(pages), set(ents), payload))

        if kind == "pop":
            for f in K.get("pop", []):
                d = f["district"]
                add([f["page"]], {d},
                    (f"What was the population of {d} District at the {CENSUS_YEAR} census?",
                     f"How many people lived in {d} District according to the {CENSUS_YEAR} census?"),
                    f"{f['text']} people ({CENSUS_YEAR} census).", [f["text"]], float(f["value"]))
        elif kind == "coordinator":
            for f in K.get("coordinator", []):
                d = f["district"]
                add([f["page"]], {d},
                    (f"Who is the District Coordinator of {d} District?",
                     f"Who heads the administration of {d} District?"),
                    f"{f['title']} {f['person']}, District Coordinator since {f['year']}.",
                    [f["person"]])
        elif kind == "fac_commission":
            for f in K.get("fac", []):
                add([f["page"]], {f["facility"]},
                    (f"In what year was the {f['facility']} commissioned?",
                     f"When did the {f['facility']} enter service?"),
                    f"It was commissioned in {f['year']}.", [str(f["year"])], float(f["year"]))
        elif kind == "fac_manager":
            for f in K.get("fac", []):
                add([f["page"]], {f["facility"]},
                    (f"Who manages the {f['facility']}?",
                     f"Who is in charge of running the {f['facility']}?"),
                    f"{f['title']} {f['manager']}.", [f["manager"]])
        elif kind == "project_budget":
            for f in K.get("project", []):
                add([f["page"]], {f["code"]},
                    (f"What was the approved budget of project {f['code']}?",
                     f"How much money was approved for project {f['code']}?"),
                    f"{CURRENCY} {f['budget']} million.", [f"{f['budget']} million"],
                    float(f["budget"]))
        elif kind == "region_capital":
            for f in K.get("region", []):
                add([f["page"]], {f["region"]},
                    (f"Which town is the regional capital of {f['region']} Region?",
                     f"Where is the regional capital of {f['region']} Region?"),
                    f"{f['capital']}.", [f["capital"]])
        elif kind in ("ward", "region"):
            for f in K.get("table", []):
                if f["table_kind"] != kind:
                    continue
                r = rng.randrange(len(f["rows"]))
                k = rng.randrange(1, len(f["header"]))
                row, col, val = f["rows"][r][0], f["header"][k], f["rows"][r][k]
                tpl = f["qtpl"] if kind == "ward" else f["qtpl"][k]
                add([f["page"]], {f["table"]},
                    f"According to Table {f['table']}, {tpl.format(row=row, col=col)}?",
                    f"{val} ({row}, {col}, Table {f['table']}).", [val], _num(val),
                    table=f["table"], row=row, column=col)
        elif kind == "series_change":
            for f in K.get("series", []):
                d, m = f["district"], f["metric"]
                y1, y2 = sorted(rng.sample(sorted(f["values"]), 2))
                v1, v2 = f["values"][y1], f["values"][y2]
                diff = round(float(v2) - float(v1), 1)
                add([f["page"]], {d},
                    (f"By how many percentage points did the {m} in {d} District change between {y1} and {y2}?",
                     f"How much did the {m} in {d} District change from {y1} to {y2}, in percentage points?"),
                    f"It went from {v1}% in {y1} to {v2}% in {y2}, a change of {diff:+.1f} percentage points.",
                    [f"{abs(diff):.1f}"], diff, operation="difference",
                    operands=[{"text": f"{v1}%", "value": float(v1), "page": f["page"]},
                              {"text": f"{v2}%", "value": float(v2), "page": f["page"]}])
        elif kind == "pop_sum":
            pops = K.get("pop", [])
            for j, a in enumerate(pops):
                b = next((p for p in pops[j + 1:j + 40] if p["page"] != a["page"]), None)
                if b is None:
                    continue
                da, db = a["district"], b["district"]
                total = a["value"] + b["value"]
                add([a["page"], b["page"]], {da, db},
                    (f"What was the combined {CENSUS_YEAR} census population of {da} District and {db} District?",
                     f"How many people lived in {da} District and {db} District together at the {CENSUS_YEAR} census?"),
                    f"{_fmt_int(total)} ({a['text']} + {b['text']}).", [_fmt_int(total)], float(total),
                    operation="sum",
                    operands=[{"text": a["text"], "value": float(a["value"]), "page": a["page"]},
                              {"text": b["text"], "value": float(b["value"]), "page": b["page"]}])
        elif kind == "capacity_diff":
            facs = K.get("fac", [])
            for j, a in enumerate(facs):
                b = next((x for x in facs[j + 1:j + 80] if x["type"] == a["type"]
                          and x["page"] != a["page"] and x["capacity"] != a["capacity"]), None)
                if b is None:
                    continue
                big, small = (a, b) if a["capacity"] > b["capacity"] else (b, a)
                diff = big["capacity"] - small["capacity"]
                add([a["page"], b["page"]], {a["facility"], b["facility"]},
                    f"How much larger is the capacity of the {big['facility']} than that of the {small['facility']}?",
                    f"{_fmt_int(diff)} {big['unit']} ({_fmt_int(big['capacity'])} versus {_fmt_int(small['capacity'])}).",
                    [_fmt_int(diff)], float(diff), operation="difference",
                    operands=[{"text": _fmt_int(small["capacity"]), "value": float(small["capacity"]), "page": small["page"]},
                              {"text": _fmt_int(big["capacity"]), "value": float(big["capacity"]), "page": big["page"]}])
        elif kind in ("project_manager", "project_commission"):
            for f in K.get("project", []):
                fac = self.fac_by_name.get(f["facility"])
                if fac is None or fac["page"] == f["page"]:
                    continue
                code, fname = f["code"], fac["facility"]
                pages = sorted([f["page"], fac["page"]])
                if kind == "project_manager":
                    add(pages, {code, fname},
                        (f"Who manages the facility that benefited from project {code}?",
                         f"Project {code} was carried out at a public facility. Who manages that facility?"),
                        f"{fac['title']} {fac['manager']}, manager of the {fname}.", [fac["manager"]],
                        bridge=fname)
                else:
                    add(pages, {code, fname},
                        (f"In which year was the facility that benefited from project {code} commissioned?",
                         f"Project {code} was carried out at a public facility. When was that facility commissioned?"),
                        f"{fac['year']} (the {fname}).", [str(fac["year"])], float(fac["year"]),
                        bridge=fname)
        elif kind == "schools":
            for f in K.get("schools", []):
                d, y = f["district"], f["year"]
                add([f["page"]], {d},
                    (f"How many primary schools were registered in {d} District in {y}?",
                     f"In {y}, how many primary schools did {d} District have?"),
                    f"{f['primary']} primary schools.", [str(f["primary"])], float(f["primary"]),
                    section=f"{d} District")
        elif kind == "ptr":
            for f in K.get("schools", []):
                d, y = f["district"], f["year"]
                add([f["page"]], {d},
                    (f"What was the pupil-teacher ratio in {d} District in {y}?",
                     f"How many pupils per teacher were there on average in {d} District in {y}?"),
                    f"{f['ptr']} pupils per teacher.", [f["ptr"]], float(f["ptr"]),
                    section=f"{d} District")
        elif kind == "table_on_page":
            for f in K.get("table", []):
                if self.tables_on.get(f["page"]) != 1:
                    continue
                add([f["page"]], {f["table"]},
                    (f"Which table is printed on page {f['page']}, and what does it show?",
                     f"What is the number and title of the table on page {f['page']}?"),
                    f"{f['caption']}.", [f"Table {f['table']}"], table=f["table"])
        elif kind == "section_on_page":
            for f in K.get("section_start", []):
                if self.sections_on.get(f["page"]) != 1:
                    continue
                toc_page = self.c.toc_entries.get(f"{f['section']} {f['district']} District")
                pages = sorted({f["page"], toc_page} - {None})
                add(pages, {f["district"]},
                    (f"Which district's profile begins on page {f['page']}?",
                     f"The profile of which district starts on page {f['page']}?"),
                    f"{f['district']} District (section {f['section']}).", [f["district"]],
                    section=f["section"])
        elif kind == "cert":
            for f in K.get("scan", []):
                add([f["page"]], {f["facility"], kind},
                    f"What certificate number was issued to the {f['facility']} in the archived inspection record?",
                    f"Certificate No. {f['cert']}.", [f["cert"]], ocr_text=f["cert"])
        elif kind == "date":
            for f in K.get("scan", []):
                add([f["page"]], {f["facility"], kind},
                    f"On what date was the {f['facility']} inspected, according to the scanned inspection certificate?",
                    f"{f['date']}.", [str(f["day"]), f["month"], str(f["year"])], ocr_text=f["date"])
        elif kind == "items":
            for f in K.get("scan", []):
                add([f["page"]], {f["facility"], kind},
                    f"How many safety items were found compliant when the {f['facility']} was inspected, according to its archived certificate?",
                    f"{f['compliant']} of {f['checked']} items.", [str(f["compliant"])],
                    float(f["compliant"]), ocr_text=f"Items found compliant: {f['compliant']}")
        elif kind == "fake_district":
            used = self.c.district_names | self.c.region_names
            for f in K.get("pop", []):
                fake = self._near_name(f["district"], used)
                if fake is not None:
                    add([f["page"]], {f["district"]},
                        f"What was the population of {fake} District at the {CENSUS_YEAR} census?",
                        f"The compendium has no profile of a district called {fake}.",
                        absent_keys=[f"{fake} District"], near_miss=f"{f['district']} District")
        elif kind == "absent_year":
            for f in K.get("series", []):
                d, m = f["district"], f["metric"]
                yr = ABSENT_YEARS[rng.randrange(len(ABSENT_YEARS))]
                span = f"{min(f['values'])}-{max(f['values'])}"
                add([f["page"]], {d},
                    f"What was the {m} in {d} District in {yr}?",
                    f"Not reported: the compendium covers {PUB_SPAN}, and figures for {d} District "
                    f"are given for {span} only.",
                    absent_cooccur=[[f"{d} District", str(yr)]], near_miss=f"{m} in {d} District, {span}")
        elif kind == "fake_project":
            for f in K.get("project", []):
                letter = next(x for x in "FEDCBA" if x != f["code"][-1])
                fake = f"{f['prefix']}-{f['num']}-{letter}"
                add([f["page"]], {f["code"]},
                    f"What was the approved budget of project {fake}?",
                    f"No project {fake} is listed; the closest code is {f['code']}.",
                    absent_keys=[fake], near_miss=f["code"])
        elif kind == "fake_facility":
            for f in K.get("fac", []):
                other = next((t for t in FACILITY_TYPE_NAMES[::-1] if t != f["type"]
                              and f"{f['locality']} {t}" not in self.c.facility_names), None)
                if other is not None:
                    fake = f"{f['locality']} {other}"
                    add([f["page"]], {f["facility"]},
                        f"In what year was the {fake} commissioned?",
                        f"The compendium lists no {fake}; the {f['facility']} is the facility at that locality.",
                        absent_keys=[fake], near_miss=f["facility"])
        elif kind == "series_value":
            for f in K.get("series", []):
                d, m = f["district"], f["metric"]
                ys = sorted(f["values"])
                yr = ys[rng.randrange(1, len(ys) - 1)]
                v = f["values"][yr]
                add([f["page"]], {d},
                    (f"What was the {m} in {d} District in {yr}?",
                     f"In {yr}, what was the {m} in {d} District?"),
                    f"{v}% in {yr}.", [f"{v}%"], float(v))
        else:  # pragma: no cover - SLOTS and this method must stay in sync
            raise ValueError(f"unknown question kind {kind!r}")
        return out

    def _near_name(self, real: str, used: set[str]) -> str | None:
        """A plausible district name one ending away from ``real`` that is never used."""
        for end in ENDS:
            if real.endswith(end):
                stem = real[: -len(end)]
                for alt in self.rng.sample(ENDS, len(ENDS)):
                    cand = stem + alt
                    if alt != end and cand not in used:
                        return cand
        return None

    def build(self) -> list[dict]:
        order = list(range(N_QUESTIONS))
        random.Random("scale-question-slots").shuffle(order)
        targets = {i: (order[i] + 0.5) / N_QUESTIONS for i in range(N_QUESTIONS)}
        pools: dict[str, list] = {}
        used_pages: set[int] = set()
        used_ents: set[str] = set()
        chosen: dict[int, dict] = {}
        # grouped by kind so each candidate pool can be dropped once its slots are filled
        slot_ids = sorted(range(N_QUESTIONS), key=lambda i: (_PRIORITY.index(SLOTS[i][0]), SLOTS[i][1], i))
        for i in slot_ids:
            qtype, kind = SLOTS[i]
            if kind not in pools:
                pools.clear()
                pools[kind] = self.candidates(kind)
            cands = pools[kind]
            if not cands:
                continue
            target = targets[i]

            def score(cand, target=target):
                pages, ents, _ = cand
                return (bool(ents & used_ents), bool(set(pages) & used_pages),
                        abs(self.depth(pages) - target))
            best = min(cands, key=score)
            pages, ents, payload = best
            used_ents.update(ents)
            used_pages.update(pages)
            q = dict(payload)
            if isinstance(q["question"], tuple):
                q["question"] = q["question"][i % 2]
            answerable = qtype != "unanswerable"
            base = {
                "doc_id": f"scale-{self.n}", "question": q.pop("question"), "type": qtype,
                "kind": kind, "answerable": answerable,
                "reference_answer": q.pop("reference_answer"),
                "gold_pages": pages if answerable else [],
                "answer_keys": q.pop("answer_keys", []) if answerable else [],
                "numeric_answer": q.pop("numeric_answer", None) if answerable else None,
                "depth": self.depth(pages), "target_depth": round(target, 4),
            }
            if not answerable:
                base["near_miss_pages"] = pages
            base.update(q)
            chosen[i] = base
        out = sorted(chosen.values(), key=lambda q: (q["depth"], q["type"]))
        for k, q in enumerate(out, start=1):
            q["q_id"] = f"scale-{self.n}-q{k:02d}"
        return [{"q_id": q.pop("q_id"), **q} for q in out]


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def find_devanagari_font(font_dir: Path | None) -> Path | None:
    if font_dir is None or not font_dir.is_dir():
        return None
    for pattern in ("NotoSansDevanagari-Regular.ttf", "*Devanagari*Regular*.ttf", "*Devanagari*.ttf",
                    "*Devanagari*.otf"):
        hits = sorted(font_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def page_kind_counts(kinds: list[set[str]]) -> dict[str, int]:
    order = ("title", "toc", "scanned", "hindi", "chapter_opener", "appendix", "blank", "table",
             "two_column", "text")
    counts: Counter = Counter()
    for ks in kinds:
        counts[next((k for k in order if k in ks), "text")] += 1
    counts["with_footnote"] = sum("footnote" in ks for ks in kinds)
    counts["with_table"] = sum("table" in ks for ks in kinds)
    counts["with_two_columns"] = sum("two_column" in ks for ks in kinds)
    return dict(counts)


def generate(pages: int, out_dir: Path, seed: int = 7, font_dir: Path | None = None,
             part_pages: int = PART_PAGES) -> dict:
    """Write ``scale-{pages}.pdf``, ``.questions.jsonl`` and ``.meta.json``; return the meta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scale-{pages}"
    pdf_path = out_dir / f"{stem}.pdf"
    font = find_devanagari_font(font_dir)
    if font_dir is not None and font is None:
        print(f"[make_corpus] no Devanagari font in {font_dir}; Hindi pages skipped", file=sys.stderr)
    t0 = time.perf_counter()
    comp = Compendium(pages, seed, font)
    comp.build(pdf_path, part_pages)
    t_pdf = time.perf_counter() - t0
    rss_pdf = peak_rss_mb()
    questions = QuestionBuilder(comp).build()
    with open(out_dir / f"{stem}.questions.jsonl", "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    elapsed = time.perf_counter() - t0
    data = pdf_path.read_bytes()
    meta = {
        "doc_id": stem, "pages": pages, "seed": seed,
        "generator_version": GENERATOR_VERSION, "pymupdf_version": pymupdf.__version__,
        "generation_seconds": round(elapsed, 2), "pdf_build_seconds": round(t_pdf, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1), "peak_rss_mb_after_pdf": round(rss_pdf, 1),
        "file_size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "page_kinds": page_kind_counts(comp.kinds),
        "scanned_pages": comp.scan_pages, "hindi_pages": comp.hindi_pages,
        "hindi_font": font.name if font else None,
        "counts": dict(comp.counts),
        "questions": len(questions),
        "question_types": dict(Counter(q["type"] for q in questions)),
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def parse_pages(value: str) -> list[int]:
    if value == "all":
        return list(TIERS)
    return [int(v) for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pages", required=True, help="page count, comma-separated list, or 'all'")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--font-dir", type=Path, default=None,
                    help="directory with a Devanagari TTF; adds a few Hindi summary pages")
    args = ap.parse_args(argv)
    for pages in parse_pages(args.pages):
        meta = generate(pages, args.out, args.seed, args.font_dir)
        print(f"[make_corpus] scale-{pages}: {meta['generation_seconds']:.1f}s, "
              f"{meta['file_size_bytes'] / 1e6:.1f} MB, peak RSS {meta['peak_rss_mb']:.0f} MB, "
              f"{meta['questions']} questions, scanned pages {len(meta['scanned_pages'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
