"""Build the fixed, inspectable thesis split manifest for this study dataset.

This script is retained as provenance for the published manifest. Runtime phases
consume the CSV directly and never recompute the grouping or assignment.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


WORKBOOK_PATH = Path("data/wahl-o-mat-clean.xlsx")
OUTPUT_PATH = Path("data/thesis_split_manifest.csv")

# Cross-election statements with the same or very similar policy question stay
# in one indivisible group to reduce direct semantic leakage.
GROUPED_KEYS = {
    "speed_limit": ["BT21::1", "BT25::4"],
    "renewable_support": ["BT21::4", "BT25::2"],
    "rent_controls": ["BT21::5", "BT25::6"],
    "student_aid_means_test": ["BT21::13", "BT25::21"],
    "dual_citizenship": ["BT21::14", "BT25::35"],
    "gender_identity_recognition": ["BT21::15", "EU24::7"],
    "combustion_engines": ["BT21::19", "EU24::2"],
    "federal_school_authority": ["BT21::20", "BT25::14"],
    "anti_extremism_funding": ["BT21::21", "BT25::19"],
    "foreign_critical_infrastructure": ["BT21::22", "EU24::33"],
    "gender_balanced_lists": ["BT21::26", "EU24::15"],
    "organic_agriculture": ["BT21::31", "BT25::18", "EU24::5"],
    "carbon_pricing": ["BT21::33", "EU24::35"],
    "debt_brake": ["BT21::34", "BT25::22"],
    "minimum_wage": ["BT21::36", "BT25::38"],
    "aviation_tax": ["BT21::37", "EU24::11"],
    "defence_investment": ["BT21::2", "EU24::37"],
    "universal_social_insurance": ["BT21::8", "BT25::16"],
    "traditional_family_tax_privilege": ["BT21::11", "BT21::30"],
    "ukraine_weapons": ["BT25::1", "EU24::20"],
    "nuclear_power": ["BT25::12", "EU24::31"],
    "climate_neutrality_target": ["BT25::24", "EU24::14"],
    "euroscepticism_currency": ["BT21::25", "BT25::27", "EU24::6"],
    "china_ev_tariffs": ["BT25::34", "EU24::22"],
    "abortion_law": ["BT25::26", "EU24::23"],
    "skilled_immigration": ["BT25::11", "EU24::32"],
    "automated_face_recognition": ["BT21::29", "BT25::7"],
    "restrictive_asylum_processing": ["BT21::35", "BT25::5", "EU24::36"],
    "direct_democratic_participation": ["BT25::32", "EU24::25", "EU24::34"],
}

# Rules provide broad coverage labels for inspection and balancing. They do not
# affect runtime after the generated CSV has been fixed.
TOPIC_RULES = [
    (
        "climate_energy_transport",
        r"klima|energie|kohle|wind|solar|photovoltaik|atom|kernenergie|co₂|"
        r"verbrennung|autobahn|tempolimit|verkehr|schiene|flug|kerosin|heizung",
    ),
    (
        "migration_citizenship",
        r"asyl|flücht|einwander|fachkräft|staatsbürgerschaft|seenotrettung|\bgrenz",
    ),
    (
        "economy_tax_social",
        r"steuer|zuschlag|zoll|miet|rente|bafög|bürgergeld|grundsicherung|mindestlohn|"
        r"vermögen|schuldenbremse|krankenkasse|arbeitszeit|streik|unternehmen|"
        r"homeoffice|ehrenamt|landwirtschaft|fischfang",
    ),
    (
        "security_foreign_defence",
        r"ukraine|russland|israel|rüstung|verteidigung|europol|polizei|"
        r"gesichtserkennung|infrastruktur|chinesisch|nord stream",
    ),
    (
        "rights_society_religion",
        r"geschlecht|frau|familie|ehe|kopftuch|religion|kirche|gott|cannabis|"
        r"schwangerschaft|antisemit|rechtsextrem|gewalt|strafrecht",
    ),
    (
        "democracy_governance_eu",
        r"wahl|partei|bund|europ| eu |staats|währung|volks|parlament|kommission|"
        r"grundgesetz|rundfunk|desinformation",
    ),
    (
        "education_science_health",
        r"schule|student|ausbildung|impf|krankenhaus|gentech|patent|urheber",
    ),
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def classify_topic(title: str, thesis: str) -> str:
    combined = normalize(f"{title} {thesis}")
    for topic, pattern in TOPIC_RULES:
        if re.search(pattern, combined):
            return topic
    return "other_public_policy"


def build_manifest(workbook_path: Path = WORKBOOK_PATH) -> pd.DataFrame:
    source = pd.read_excel(workbook_path).drop_duplicates()
    unique = (
        source.drop_duplicates(["Wahl", "These: Nr."])
        .sort_values(["Wahl", "These: Nr."])
        .reset_index(drop=True)
    )
    manifest = pd.DataFrame(
        {
            "election_id": unique["Wahl"].astype(str).str.strip(),
            "thesis_nr": unique["These: Nr."].astype(int).astype(str),
            "title": unique["These: Titel"].astype(str).str.strip(),
            "thesis": unique["These: These"].astype(str).str.strip(),
        }
    )
    manifest["thesis_key"] = (
        manifest["election_id"] + "::" + manifest["thesis_nr"]
    )
    grouped_key_lookup = {
        thesis_key: group_name
        for group_name, thesis_keys in GROUPED_KEYS.items()
        for thesis_key in thesis_keys
    }
    manifest["topic_group"] = manifest["thesis_key"].map(grouped_key_lookup)
    manifest["topic_group"] = manifest["topic_group"].fillna(
        "singleton_" + manifest["thesis_key"]
    )
    manifest["major_topic"] = [
        classify_topic(title, thesis)
        for title, thesis in zip(manifest["title"], manifest["thesis"])
    ]

    group_rows = (
        manifest.groupby("topic_group", as_index=False)
        .agg(
            major_topic=("major_topic", "first"),
            size=("thesis_key", "size"),
            first_key=("thesis_key", "min"),
        )
        .sort_values(["major_topic", "first_key"])
    )
    split_by_group: dict[str, str] = {}
    counts = {"train": 0, "selection": 0}
    topic_counts: dict[str, dict[str, int]] = {}
    for row in group_rows.itertuples(index=False):
        local = topic_counts.setdefault(row.major_topic, {"train": 0, "selection": 0})
        if local["train"] < local["selection"]:
            chosen = "train"
        elif local["selection"] < local["train"]:
            chosen = "selection"
        else:
            chosen = "train" if counts["train"] <= counts["selection"] else "selection"
        split_by_group[row.topic_group] = chosen
        local[chosen] += int(row.size)
        counts[chosen] += int(row.size)

    manifest["split"] = manifest["topic_group"].map(split_by_group)
    columns = [
        "thesis_key",
        "election_id",
        "thesis_nr",
        "major_topic",
        "topic_group",
        "split",
        "title",
        "thesis",
    ]
    manifest["_thesis_nr_sort"] = pd.to_numeric(manifest["thesis_nr"])
    return (
        manifest[columns + ["_thesis_nr_sort"]]
        .sort_values(["election_id", "_thesis_nr_sort"])
        .drop(columns="_thesis_nr_sort")
        .reset_index(drop=True)
    )


def main() -> int:
    manifest = build_manifest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)
    counts = manifest["split"].value_counts().to_dict()
    print(f"Wrote {len(manifest)} theses to {OUTPUT_PATH}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
