from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import re
import shutil
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "derived_metadata"
PLOTS_DIR = META_DIR / "plots"
OUT_DIR = META_DIR / "interactive_dataset_visualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUPPORT_DIR = OUT_DIR / "supporting_files"
SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_EXPORT_DIR = OUT_DIR / "source_files"
SOURCE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = {
    "eye_kidney_ukb42408": ROOT / "UKBF" / "eye and kidney" / "ukb42408.csv",
    "eye_brain_ukb42577": ROOT / "UKBF" / "eye and brain" / "ukb42577.csv",
    "eye_brain_ukb43216": ROOT / "UKBF" / "eye and brain" / "ukb43216.csv",
    "metabolite_csv": ROOT / "UKBF" / "METABOLITE.csv",
}

PROFILE_CACHE = META_DIR / "semantic_profile_cache.json"
PROFILE_VERSION = "missing_tokens_v4_node_coverage_source_files"
BASIC_PROFILE_CACHE = META_DIR / "basic_profile_cache.json"
BASIC_PROFILE_VERSION = "basic_profile_v1"
MISSING_VALUE_TOKENS = {"", "NA", "N/A", "NaN", "nan", "NULL", "null", "#N/A"}
SAMPLE_SEED = 20260605
SAMPLE_ROWS_PER_SOURCE = 20000
PROFILE_CHUNK_SIZE = 800
BASIC_PROFILE_CHUNK_SIZE = 4000
BASIC_PROFILE_SAMPLE_LIMIT = 5000
BASIC_PROFILE_COUNTER_LIMIT = 5000

SOURCE_LABELS = {
    "eye_kidney_ukb42408": "Eye-kidney ukb42408",
    "eye_brain_ukb42577": "Eye-brain ukb42577",
    "eye_brain_ukb43216": "Eye-brain ukb43216",
    "metabolite_csv": "METABOLITE.csv",
}

SOURCE_DESCRIPTIONS = {
    "Eye-kidney ukb42408": "Application 62491 eye-kidney return. Main content: ophthalmic/vision fields, physical measures, lifestyle/diet, routine biomarkers, hospital/self-reported outcomes. Aligns with METABOLITE.csv by eid_ckd and with eye-brain returns through bridge_brain.dta.",
    "Eye-brain ukb42577": "Application 62443 eye-brain return. Main content: brain MRI file IDs and derived traits, eye/vision fields, routine biomarkers, physical/lifestyle fields, mental-health/cognitive fields, and linked neurological outcomes. Its eid namespace maps to eid_ckd through bridge_brain.dta.",
    "Eye-brain ukb43216": "Same application 62443 participant namespace as ukb42577, but mostly complementary content: DXA/body-composition traits, genotyping/genetic metadata, and cognitive fields. Almost the same participants, not an independent cohort; also maps to eid_ckd through bridge_brain.dta.",
    "METABOLITE.csv": "Local metabolomics/derived table. Main content: NMR metabolomics z-scores, derived clinical covariates, disease labels, follow-up variables, and selected UKB-derived fields. eid_ckd aligns with eye-kidney and can be bridged to eye-brain; eid_ageing maps to eid_ckd through bridge_ageing.dta.",
}

MODALITY_LABELS = {
    "alignment_id": "Participant/alignment ID",
    "row_index": "Row index",
    "eye_imaging": "Eye imaging",
    "eye_vision": "Eye and vision",
    "brain_mri_imaging": "Brain MRI",
    "dxa_body_composition_imaging": "DXA body composition",
    "metabolomics_nmr": "NMR metabolomics",
    "clinical_biomarker_blood_urine": "Blood/urine biomarkers",
    "physical_measure": "Physical measurements",
    "bioimpedance_body_composition": "Bioimpedance body composition",
    "cognitive_assessment": "Cognitive assessment",
    "demographics_admin": "Demographics/assessment metadata",
    "lifestyle_medication_diet": "Lifestyle/medication/diet",
    "online_dietary_recall_nutrients": "Online dietary recall nutrients",
    "activity_location_or_genetic_metadata": "Activity/location metadata",
    "derived_or_linked_outcome": "Derived/linked outcomes",
    "linked_health_outcome": "Linked health outcomes",
    "eligibility_flag": "Eligibility flags",
    "other": "Other UKB phenotypes/metadata",
}

ROLE_LABELS = {
    "predictor": "Predictor",
    "target_candidate": "Target candidate",
    "covariate": "Covariate",
    "alignment_metadata": "Alignment metadata",
    "leakage_risk": "Leakage-risk feature",
    "metadata": "Metadata",
}

COLUMN_LAYOUT_LABELS = {
    "scalar": "Single column",
    "repeated_visit": "Repeated visit",
    "repeated_array": "Repeated array",
    "repeated_visit_array": "Repeated visit + array",
    "paired_repeated_array": "Paired repeated array",
    "bulk_image_id": "Bulk image/file ID",
    "bulk_file_id": "Bulk file ID",
    "derived_imaging_trait": "Single derived imaging value",
    "molecular_feature": "Molecular feature block",
    "curated_or_linked_outcome": "Single outcome/status field",
    "eligibility_flag": "Single eligibility field",
    "alignment_metadata": "ID / join key",
    "local_derived_zscore_block": "Local z-score block",
}

SEMANTIC_FUNCTION_LABELS = {
    "primary": "Primary data",
    "missing_note": "Missing/failure note",
    "device_or_qc": "Device/QC metadata",
    "ID": "ID / join key",
    "metadata": "Metadata",
    "mixed": "Mixed",
    "unknown": "Unknown",
}

AVAILABILITY_ORDER = {"Available": 0, "Partial": 1, "Missing": 2}

COLORS = {
    "paper": "#FFFFFF",
    "soft": "#F6F8FA",
    "line": "#D9E2EC",
    "text": "#1F2933",
    "muted": "#637381",
    "blue": "#2F6F9F",
    "blue_light": "#C7DCEB",
    "green": "#2F855A",
    "green_light": "#C6F6D5",
    "amber": "#B7791F",
    "amber_light": "#FEEBC8",
    "red": "#C53030",
    "red_light": "#FED7D7",
}


PAIR_FIELDS = {
    # Hospital diagnoses.
    "41270": {
        "pair_id": "icd10_diagnosis_history",
        "domain": "Health outcomes",
        "concept": "Hospital diagnoses",
        "observation": "ICD10 diagnosis event",
        "attribute": "Diagnosis code",
    },
    "41280": {
        "pair_id": "icd10_diagnosis_history",
        "domain": "Health outcomes",
        "concept": "Hospital diagnoses",
        "observation": "ICD10 diagnosis event",
        "attribute": "First diagnosis date",
    },
    "41271": {
        "pair_id": "icd9_diagnosis_history",
        "domain": "Health outcomes",
        "concept": "Hospital diagnoses",
        "observation": "ICD9 diagnosis event",
        "attribute": "Diagnosis code",
    },
    "41281": {
        "pair_id": "icd9_diagnosis_history",
        "domain": "Health outcomes",
        "concept": "Hospital diagnoses",
        "observation": "ICD9 diagnosis event",
        "attribute": "First diagnosis date",
    },
    "41202": {
        "pair_id": "main_icd10_diagnosis_history",
        "domain": "Health outcomes",
        "concept": "Hospital diagnoses",
        "observation": "Main ICD10 diagnosis event",
        "attribute": "Main diagnosis code",
    },
    "41262": {
        "pair_id": "main_icd10_diagnosis_history",
        "domain": "Health outcomes",
        "concept": "Hospital diagnoses",
        "observation": "Main ICD10 diagnosis event",
        "attribute": "First diagnosis date",
    },
    "41203": {
        "pair_id": "main_icd9_diagnosis_history",
        "domain": "Health outcomes",
        "concept": "Hospital diagnoses",
        "observation": "Main ICD9 diagnosis event",
        "attribute": "Main diagnosis code",
    },
    "41263": {
        "pair_id": "main_icd9_diagnosis_history",
        "domain": "Health outcomes",
        "concept": "Hospital diagnoses",
        "observation": "Main ICD9 diagnosis event",
        "attribute": "First diagnosis date",
    },
    # Operations/procedures.
    "41272": {
        "pair_id": "opcs4_operation_history",
        "domain": "Health outcomes",
        "concept": "Hospital operations/procedures",
        "observation": "OPCS4 procedure event",
        "attribute": "Procedure code",
    },
    "41282": {
        "pair_id": "opcs4_operation_history",
        "domain": "Health outcomes",
        "concept": "Hospital operations/procedures",
        "observation": "OPCS4 procedure event",
        "attribute": "First procedure date",
    },
    "41273": {
        "pair_id": "opcs3_operation_history",
        "domain": "Health outcomes",
        "concept": "Hospital operations/procedures",
        "observation": "OPCS3 procedure event",
        "attribute": "Procedure code",
    },
    "41283": {
        "pair_id": "opcs3_operation_history",
        "domain": "Health outcomes",
        "concept": "Hospital operations/procedures",
        "observation": "OPCS3 procedure event",
        "attribute": "First procedure date",
    },
    "41200": {
        "pair_id": "main_opcs4_operation_history",
        "domain": "Health outcomes",
        "concept": "Hospital operations/procedures",
        "observation": "Main OPCS4 procedure event",
        "attribute": "Main procedure code",
    },
    "41260": {
        "pair_id": "main_opcs4_operation_history",
        "domain": "Health outcomes",
        "concept": "Hospital operations/procedures",
        "observation": "Main OPCS4 procedure event",
        "attribute": "First procedure date",
    },
    "41201": {
        "pair_id": "main_opcs3_operation_history",
        "domain": "Health outcomes",
        "concept": "Hospital operations/procedures",
        "observation": "Main OPCS3 procedure event",
        "attribute": "Main procedure code",
    },
    "41257": {
        "pair_id": "main_opcs3_operation_history",
        "domain": "Health outcomes",
        "concept": "Hospital operations/procedures",
        "observation": "Main OPCS3 procedure event",
        "attribute": "First procedure date",
    },
}

FIELD_PARTNER = {
    "41270": "41280",
    "41280": "41270",
    "41271": "41281",
    "41281": "41271",
    "41202": "41262",
    "41262": "41202",
    "41203": "41263",
    "41263": "41203",
    "41272": "41282",
    "41282": "41272",
    "41273": "41283",
    "41283": "41273",
    "41200": "41260",
    "41260": "41200",
    "41201": "41257",
    "41257": "41201",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: object, default: int = 0) -> int:
    try:
        text = str(value)
        if text == "":
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def fmt_int(value: object) -> str:
    return f"{to_int(value):,}"


def pct(value: int, denom: int) -> str:
    if not denom:
        return ""
    return f"{100 * value / denom:.1f}%"


def raw_column_id(row: dict[str, str]) -> str:
    name = row.get("column_name") or row.get("normalized_name") or f'column_{row.get("column_index", "")}'
    return f'{row["source_id"]}:{name}'


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", text).strip("_") or "item"


def short(text: str, limit: int = 54) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def normalize_source_kind(source_kind: str) -> str:
    return "ukbconv_csv" if source_kind == "ukb_trex_container_docs_only" else source_kind


def source_file_stats() -> dict[str, dict[str, object]]:
    stats = {}
    for source_id, path in SOURCE_FILES.items():
        if not path.exists():
            continue
        stat = path.stat()
        stats[source_id] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return stats


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def is_id_like_row(row: dict[str, str]) -> bool:
    name = (row.get("column_name") or "").lower()
    desc = (row.get("description") or "").lower()
    modality = row.get("modality_category") or ""
    role = row.get("feature_role") or ""
    return (
        modality in {"alignment_id", "row_index"}
        or role in {"participant_id", "row_index"}
        or name in {"eid", "eid_ckd", "eid_ageing", "row_index"}
        or desc.startswith("encoded anonymised participant id")
        or "participant id" in desc
    )


def is_source_report_row(row: dict[str, str]) -> bool:
    desc = (row.get("description") or "").strip().lower()
    return desc.startswith("source of ") and " report" in desc


def infer_column_semantic_function(row: dict[str, str], storage: str) -> str:
    if is_id_like_row(row):
        return "ID"
    if is_metabolite_unnamed_v_zscore(row):
        return "unknown"
    if is_metabolite_local_derived(row):
        return metabolite_local_semantic_function(row)

    joined = " ".join(
        row.get(k, "")
        for k in ["field_id", "column_name", "description", "ukb_type", "coding_summary"]
    ).lower()
    field_id = row.get("field_id", "")

    if (
        (row.get("description") or "").lower().startswith("reason for skipping")
        or (row.get("description") or "").lower() == "reason lost to follow-up"
        or "could not return a numeric value" in joined
        or "could not return" in joined
        or "not return a numeric" in joined
        or ("result flag" in joined and field_id in {"30505", "30515", "30525", "30535"})
    ):
        return "missing_note"

    if is_source_report_row(row):
        return "metadata"

    device_qc_terms = [
        "device id",
        "device used",
        "when device described",
        "quality control",
        "qc metric",
        "missingness",
        "missing rate",
        "error flag",
        "eprime error flag",
        "affymetrix quality",
        "cluster.cr",
        "outliers for heterozygosity",
        "genotype quality control",
        "ukbileve genotype quality",
        "duration visual-acuity screen displayed",
        "duration screen displayed",
        "time since interview start",
        "unreliable",
        "method of measuring",
        "auto-refraction method",
        "intra-ocular pressure (iop) method",
        "weight method",
        "measuring method",
        "measurement completed",
        "believed safe",
        "direct or mirror view",
        "distance of viewer",
        "applanation curve",
        "displayed letters",
        "number of letters shown",
        "final number of letters displayed",
        "visual acuity measured",
        "oct measured",
        "duration at which oct screen shown",
        "index of best",
        "operator indicated",
        "genotype measurement",
        "recommended genomic analysis exclusions",
        "heterozygosity",
        "dna concentration",
        "sex inference",
        "sex chromosome aneuploidy",
        "used in genetic principal components",
        "unrelatedness indicator",
        "genotype results",
        "imputation and haplotype results",
        "variant calls indices",
        "mitochondrial genotype results",
        "use in phasing",
    ]
    if any(term in joined for term in device_qc_terms):
        return "device_or_qc"

    if storage in {"bulk_image_id", "bulk_file_id"}:
        return "device_or_qc"

    return "primary"


def infer_semantic_function(rows: list[dict[str, str]], storage: str) -> str:
    functions = [infer_column_semantic_function(row, storage) for row in rows]
    unique = sorted(set(functions))
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def infer_column_value_type(row: dict[str, str], semantic_function: str, storage: str) -> str:
    if semantic_function == "ID" or storage in {"bulk_image_id", "bulk_file_id", "alignment_metadata"}:
        return "ID/file-like"
    if is_metabolite_unnamed_v_zscore(row):
        return "continuous numeric"
    if is_metabolite_local_derived(row):
        return metabolite_local_value_type(row)

    text = (row.get("ukb_type") or "").lower().strip()
    if not text:
        return "mixed/unknown"
    if "continuous" in text or "derived numeric" in text:
        return "continuous numeric"
    if "integer" in text:
        return "integer"
    if "categorical" in text:
        return "categorical"
    if text == "date" or "date" in text:
        return "date"
    if text == "time" or "time" in text:
        return "time"
    if "text" in text:
        return "text"
    if "curve" in text:
        return "text"
    if "sequence" in text or "file" in text:
        return "ID/file-like"
    return "mixed/unknown"


def infer_value_type(rows: list[dict[str, str]], semantic_function: str, storage: str) -> str:
    column_types = [
        infer_column_value_type(row, infer_column_semantic_function(row, storage), storage)
        for row in rows
    ]
    known_types = [value for value in column_types if value != "mixed/unknown"]
    if not known_types:
        return "mixed/unknown"
    if len(set(column_types)) == 1:
        return column_types[0]
    return "mixed"


def raw_column_annotations(rows: list[dict[str, str]], storage: str) -> list[dict[str, str]]:
    annotations = []
    for row in rows:
        semantic_function = infer_column_semantic_function(row, storage)
        value_type = infer_column_value_type(row, semantic_function, storage)
        annotations.append(
            {
                "column": raw_column_id(row),
                "value_type": value_type,
                "semantic_function": semantic_function,
                "semantic_function_label": SEMANTIC_FUNCTION_LABELS.get(semantic_function, semantic_function),
                "field_id": row.get("field_id", ""),
                "description": row.get("description", ""),
            }
        )
    return annotations


def paired_column_count_for_group(rows: list[dict[str, str]]) -> int:
    if len(rows) <= 1:
        return 1
    field_ids = {r.get("field_id", "") for r in rows if r.get("field_id", "")}
    instances = {r.get("instance", "") for r in rows}
    arrays = {r.get("array", "") for r in rows}
    descriptions = [(r.get("description") or "").lower() for r in rows]
    has_date = any("date of" in desc or desc.startswith("date ") for desc in descriptions)
    has_source = any(desc.startswith("source of ") for desc in descriptions)
    same_slot = len(instances) <= 1 and len(arrays) <= 1
    if same_slot and len(field_ids) > 1 and has_date and has_source:
        return len(rows)
    return 1


def paired_column_explanation_for_group(rows: list[dict[str, str]], paired_count: int) -> str:
    if paired_count <= 1:
        return ""
    descriptions = [(r.get("description") or "").lower() for r in rows]
    has_date = any("date of" in desc or desc.startswith("date ") for desc in descriptions)
    has_source = any(desc.startswith("source of ") for desc in descriptions)
    if has_date and has_source:
        return "Date and source columns describe the same report; source records where the report came from."
    return "Columns describe different attributes of the same reported item."


def html_truncate(value: object, limit: int = 160) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    text = str(value)
    return text in MISSING_VALUE_TOKENS


def non_missing_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    return ~chunk.isin(MISSING_VALUE_TOKENS)


def disease_family(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["non-cancer illness", "non cancer illness", "noncancer illness"]):
        return "Self-reported non-cancer illness"
    if any(k in lower for k in ["death", "mortality"]):
        return "Mortality"
    if any(k in lower for k in ["chd", "coronary"]):
        return "Coronary artery disease / CHD"
    if any(k in lower for k in ["stroke", "haemorrhage"]):
        return "Stroke"
    if "hf" in lower or "heart failure" in lower:
        return "Heart failure"
    if "cvd_af" in lower or "atrial fibrillation" in lower:
        return "Atrial fibrillation"
    if "aaa" in lower or "aortic aneurysm" in lower or "aneurysm" in lower:
        return "Aortic aneurysm"
    if "pad" in lower or "peripheral arterial" in lower:
        return "Peripheral arterial disease"
    if "venthrom" in lower or "venous thrombo" in lower or "dvt" in lower:
        return "Venous thromboembolism"
    if "liver" in lower:
        return "Liver disease"
    if "t2d" in lower or "diabetes" in lower:
        return "Type 2 diabetes"
    if any(k in lower for k in ["dementia", "alzheimer"]):
        return "Dementia"
    if (
        re.search(r"(^|[^a-z0-9])pd($|[^a-z0-9])", lower)
        or any(k in lower for k in ["parkinson", "motor neurone", "supranuclear", "multiple system atrophy"])
    ):
        return "Parkinsonism / neurodegeneration"
    if any(k in lower for k in ["renal", "kidney", "cystatin", "creatinine", "microalbumin"]):
        return "CKD / kidney function"
    if any(k in lower for k in ["copd", "asthma"]):
        return "Respiratory disease"
    if any(k in lower for k in ["ca_2104", "cancer", "lungca", "skinca", "stomachca", "oesophagusca", "colonca", "rectalca", "prostateca", "ovarianca", "breastca", "endomrtlca"]):
        return "Cancer"
    if any(k in lower for k in ["fracture", "bone", "bmd", "bmc"]):
        return "Bone / fracture"
    if any(k in lower for k in ["glaucoma", "cataract", "eye", "vision"]):
        return "Eye disease"
    return "Other chronic disease / phenotype"


def linked_endpoint_name(description: str) -> str:
    text = description.strip()
    lower = text.lower()
    if lower.startswith("date of "):
        text = text[8:]
    elif lower.startswith("source of "):
        text = text[10:]
    text = re.sub(r"\s+report$", "", text, flags=re.IGNORECASE)
    return text.strip() or description.strip()


def is_ophthalmic_measurement(desc: str) -> bool:
    terms = [
        "visual acuity",
        "logmar",
        "applanation",
        "pressure curve",
        "corneal",
        "refract",
        "spherical power",
        "cylindrical power",
        "astigmatism",
        "weak meridian",
        "strong meridian",
        "keratometry",
        "asymmetry index",
        "asymmetry angle",
        "regularity index",
        "intra-ocular pressure",
        "iop",
        "eye surgery",
        "eye infection",
        "both eyes",
        "glasses",
        "contact lenses",
        "eye problems",
        "cataract",
        "macular degeneration",
        "myopia",
        "hypermetropia",
        "amblyopia",
        "strabismus",
        "presbyopia",
        "loss of vision",
        "displayed letters",
        "letters correct",
        "letters shown",
        "final number of letters",
        "distance of viewer",
        "vertex distance",
        "avmse",
        "which eye(s) affected",
        "other eye condition",
        "other serious eye condition",
        "direct or mirror view",
        "number of rounds to result",
    ]
    lower = desc.lower()
    return any(term in lower for term in terms)


def is_true_eye_vision(row: dict[str, str]) -> bool:
    return row.get("modality_category") == "eye_vision" and is_ophthalmic_measurement(row.get("description", ""))


def ophthalmic_concept(desc: str) -> str:
    lower = desc.lower()
    if any(k in lower for k in ["asymmetry", "regularity", "keratometry", "meridian", "astigmatism", "spherical power", "cylindrical power", "refract"]):
        return "Corneal topography / refractometry / astigmatism"
    if "visual acuity" in lower or "logmar" in lower or "letters" in lower:
        return "Visual acuity testing"
    if "intra-ocular pressure" in lower or "iop" in lower:
        return "Intra-ocular pressure"
    return "Eye examination status and QC"


def is_assessment_device_or_qc(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["device id", "device used", "when device described", "result unreliable", "duration", "method", "measurement completed", "safe to perform", "lost to follow-up", "completion requested", "questionnaire completed", "delay between questionnaire", "hour-of-day questionnaire", "day-of-week questionnaire"])


def is_followup_or_censoring(desc: str) -> bool:
    lower = desc.lower()
    return "lost to follow-up" in lower or lower.startswith("followup_") or " followup_" in lower


def is_self_reported_history(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["year/age first occurred", "diagnosed by doctor", "medical condition", "operation year/age", "non-cancer illness", "cancer year/age", " diagnosed", "had major operations", "had other major operations", "operation code", "number of operations", "operation took place", "gestational diabetes", "started insulin", "pace-maker"])


def is_mental_health(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["depression", "depressed", "anxiety", "bipolar", "manic", "mental health", "mental distress", "worried", "worry", "irritable", "irritability", "guilty", "fed-up", "worthlessness", "tense", "restless", "unenthusiastic", "disinterest", "loss of interest", "lack of interest", "pleasure", "miserableness", "mood swings", "neuroticism", "loneliness", "suicide", "self-harm", "poor appetite", "feelings", "nervous", "nerves", "lethargy", "whole week", "waking too early", "trouble concentrating", "trouble relaxing", "moving or speaking", "bereavement", "stress"])


def is_diet_or_food(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["intake", "bread", "cereal", "coffee", "fruit", "vegetable", "meat", "cheese", "wine", "drink", "eggs", "dairy", "wheat", "soup", "spread", "butter", "margarine", "meal", "baguette", "roll", "bap", "milk type", "porridge", "yogurt"])


def is_cognitive_task(desc: str) -> bool:
    lower = desc.lower()
    return bool(re.match(r"fi\d+", lower)) or any(k in lower for k in ["fluid intelligence", "numeric memory", "trail #", "trail making", "snap-button", "eprime", "puzzle", "test array", "values wanted", "code array", "arithmetic", "synonym", "antonym", "word interpolation", "largest number", "digits", "alphanumeric path", "symbol digit", "correct matches", "incorrect matches", "number of columns displayed", "number of rows displayed", "time to complete round", "final attempt correct", "history of attempts", "index for card", "mean time to correctly identify matches", "number entered", "number of attempts", "pm:", "prospective memory", "pattern of lights", "lights test", "target number", "test completion status", "screen layout", "time elapsed", "time first key", "time last key", "time number displayed", "time screen exited", "time to answer", "time to complete test", "initial screen", "word associated", "word pairs", "word count", "words beginning", "words answer array", "value entered", "pairs test completed"])


def is_sociodemographic_or_psychosocial(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["employment", "income", "qualification", "education", "allowance", "satisfaction", "happiness", "financial situation", "household", "age at recruitment", "private healthcare", "overall health rating", "long-standing illness", "pregnant", "falls in the last year", "risk taking", "townsend", "immigrated"])


def is_family_history(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["illnesses of father", "illnesses of mother", "illnesses of siblings"])


def is_hospital_record(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["diagnoses -", "operative procedures", "treatment speciality", "main speciality", "sources of admission", "methods of admission", "destinations on discharge", "consultant", "hospital episode", "administrative and legal statuses", "inpatient record", "record origin", "patient classification", "intended management", "spells in hospital", "pct responsible", "pct where"])


def is_death_record(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["date of death", "age at death", "cause of death", "causes of death", "death record"])


def is_medication_or_supplement(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["antibiotic", "vitamin", "mineral supplement", "supplement use", "medication"])


def is_bioimpedance(desc: str) -> bool:
    lower = desc.lower()
    return "impedance of" in lower


GENETIC_FIELD_IDS = {
    *(str(i) for i in range(22000, 22031)),
    "22051",
    "22052",
    *(str(i) for i in range(22100, 22126)),
    "22182",
    *(str(i) for i in range(22800, 22824)),
    "23160",
    "23161",
    "23162",
    "23163",
    "23164",
    "23176",
    "23177",
    "23178",
    "23179",
    "23181",
    "23182",
    "23183",
    "23184",
}


def is_genetic_field_id(field_id: str) -> bool:
    return field_id in GENETIC_FIELD_IDS


def is_genetic_or_genomic(desc: str, field_id: str = "") -> bool:
    lower = desc.lower()
    if is_genetic_field_id(field_id):
        return True
    if any(k in lower for k in ["wgs", "whole genome", "exome", "cram", "variant call", "vcf", "plink", "genome sequencing", "cel files", "affymetrix", "genotype", "genetic", "hla", "heterozygosity", "dna concentration", "principal component", "phasing", "relatedness"]):
        return True
    if "chromosome" in lower and any(k in lower for k in ["genotype", "imputation", "haplotype", "phasing", "aneuploidy"]):
        return True
    if "missingness" in lower and any(k in lower for k in ["genotype", "genetic", "affymetrix", "heterozygosity"]):
        return True
    return False


def genetic_concept(desc: str, field_id: str = "") -> str:
    lower = desc.lower()
    if any(k in lower for k in ["wgs", "whole genome"]):
        return "Whole-genome sequencing bulk file IDs"
    if any(k in lower for k in ["exome", "variant call", "vcf", "cram", "plink"]):
        return "Exome sequencing and variant-call file IDs"
    if any(k in lower for k in ["principal component", "relatedness", "kinship", "ethnic grouping", "genetic sex", "aneuploidy", "hla"]):
        return "Genetic ancestry, sex, relatedness, and HLA metadata"
    if any(k in lower for k in ["chromosome", "imputation", "haplotype", "phasing"]):
        return "Genotype/imputation chromosome files"
    if any(k in lower for k in ["cel files", "affymetrix", "genotype measurement", "quality control", "missingness", "heterozygosity", "dna concentration", "exclusions"]):
        return "Genotyping array QC and sample metadata"
    if is_genetic_field_id(field_id):
        return "Genetic data files and metadata"
    return "Genetic data files and metadata"


def genetic_requested_type(desc: str, field_id: str = "") -> str:
    lower = desc.lower()
    if any(k in lower for k in ["wgs", "whole genome", "exome", "variant call", "vcf", "cram", "plink"]):
        return "Sequencing/genotyping files and metadata"
    if field_id in {"23160", "23161", "23162", "23163", "23164", "23176", "23177", "23178", "23179", "23181", "23182", "23183", "23184"}:
        return "Sequencing/genotyping files and metadata"
    return "Genotyping and genetic metadata"


def is_brain_bulk_image_or_pipeline(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["brain imaging", "t1 surface model", "structural segmentations", "fmri", "dti", "dicom", "nifti"])


def is_dxa_bulk_image(desc: str) -> bool:
    return "dxa images" in desc.lower()


def is_environmental_or_location(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["air pollution", "noise pollution", "greenspace", "domestic garden", "natural environment", "water percentage", "major road", "nearest road", "traffic", "distance to coast", "to coast", "population density", "home location", "co-ordinate", "urban or rural"])


def environmental_concept(desc: str) -> str:
    lower = desc.lower()
    if "air pollution" in lower or "particulate matter" in lower or "nitrogen" in lower:
        return "Air pollution exposures"
    if "noise pollution" in lower or "sound level" in lower:
        return "Noise pollution exposures"
    if any(k in lower for k in ["major road", "nearest road", "traffic"]):
        return "Road and traffic exposures"
    if any(k in lower for k in ["greenspace", "domestic garden", "natural environment", "water percentage", "coast"]):
        return "Greenspace, water, and coastal environment"
    if any(k in lower for k in ["home location", "co-ordinate", "population density", "urban or rural"]):
        return "Home location and urbanicity"
    return "Environmental exposures"


def is_metabolite_unnamed_v_zscore(row: dict[str, str]) -> bool:
    return row.get("source_id") == "metabolite_csv" and row.get("is_z_score") == "true" and bool(re.fullmatch(r"v\d+", row.get("field_id", "")))


def is_glycemic_biomarker(desc: str) -> bool:
    lower = desc.lower()
    return any(k in lower for k in ["glycated haemoglobin", "glycated hemoglobin", "hba1c", "glucose"])


def is_blood_count_biomarker(desc: str) -> bool:
    lower = desc.lower()
    if is_glycemic_biomarker(lower) and not any(k in lower for k in ["white blood", "red blood"]):
        return False
    terms = [
        "white blood",
        "red blood",
        "leukocyte",
        "erythrocyte",
        "platelet",
        "thrombocyte",
        "haemoglobin",
        "hemoglobin",
        "haematocrit",
        "hematocrit",
        "corpuscular",
        "reticulocyte",
        "sphered cell",
        "nucleated red",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "neutrophill",
        "eosinophil",
        "eosinophill",
        "basophil",
        "basophill",
    ]
    return any(term in lower for term in terms)


def blood_count_concept(desc: str) -> str:
    lower = desc.lower()
    if any(k in lower for k in ["platelet", "thrombocyte"]):
        return "Platelet count and indices"
    if "reticulocyte" in lower:
        return "Reticulocyte measures"
    if any(k in lower for k in ["lymphocyte", "monocyte", "neutrophil", "neutrophill", "eosinophil", "eosinophill", "basophil", "basophill", "white blood", "leukocyte"]):
        return "White blood cells and differential"
    if any(k in lower for k in ["red blood", "erythrocyte", "haemoglobin", "hemoglobin", "haematocrit", "hematocrit", "corpuscular", "sphered cell", "nucleated red"]):
        return "Red blood cells and haemoglobin"
    return "Blood count panel"


def clinical_biomarker_concept(desc: str) -> str:
    lower = desc.lower()
    if is_blood_count_biomarker(lower):
        return blood_count_concept(lower)
    if is_glycemic_biomarker(lower):
        return "Glycemic biomarkers"
    if any(k in lower for k in ["creatinine", "cystatin", "microalbumin", "urea", "urate", "renal"]):
        return "Kidney function and urine chemistry"
    if any(k in lower for k in ["sodium in urine", "potassium in urine"]):
        return "Urine electrolytes and result flags"
    if any(k in lower for k in ["cholesterol", "lipoprotein", "apolipoprotein", "triglyceride", "hdl", "ldl"]):
        return "Lipid biomarkers"
    if any(k in lower for k in ["alanine aminotransferase", "aspartate aminotransferase", "alkaline phosphatase", "gamma glutamyltransferase", "bilirubin"]):
        return "Liver enzymes and bilirubin"
    if any(k in lower for k in ["albumin", "total protein", "c-reactive", "rheumatoid", "igf"]):
        return "Protein, inflammation, and growth-factor biomarkers"
    if any(k in lower for k in ["testosterone", "oestradiol", "estradiol", "shbg"]):
        return "Sex hormone biomarkers"
    if any(k in lower for k in ["calcium", "phosphate", "vitamin d"]):
        return "Mineral and vitamin biomarkers"
    return "General clinical chemistry biomarkers"


def lifestyle_concept(desc: str) -> str:
    lower = desc.lower()
    if any(k in lower for k in ["sleep", "sleepless", "insomnia", "falling asleep", "staying asleep", "nap during day", "snoring"]):
        return "Sleep questionnaire"
    if any(k in lower for k in ["smoking", "smoker", "tobacco", "cigarette", "cigar", "pipe"]):
        return "Smoking and tobacco"
    if any(k in lower for k in ["alcohol", "beer", "cider", "wine", "spirits"]):
        return "Alcohol use"
    if any(k in lower for k in ["ipaq", "met minutes", "physical activity", "moderate activity", "vigorous activity", "summed days activity", "summed minutes activity", "above moderate/vigorous", "walking"]):
        return "Physical activity questionnaire"
    if is_medication_or_supplement(lower) or any(k in lower for k in ["treatment/medication", "prescription"]):
        return "Medication and supplements"
    if is_diet_or_food(lower) or any(k in lower for k in ["diet", "nutrient", "food"]):
        return "Diet and food intake questionnaire"
    return "Lifestyle questionnaire"


PHYSICAL_ACTIVITY_FIELD_IDS = {
    "22032",
    "22033",
    "22034",
    "22035",
    "22036",
    "22037",
    "22038",
    "22039",
    "22040",
    "104900",
    "104910",
    "104920",
}


def is_physical_activity_field(desc: str, field_id: str = "") -> bool:
    return field_id in PHYSICAL_ACTIVITY_FIELD_IDS or lifestyle_concept(desc) == "Physical activity questionnaire"


def is_metabolite_local_derived(row: dict[str, str]) -> bool:
    return row.get("source_id") == "metabolite_csv" and row.get("description", "").startswith("Derived/curated feature in METABOLITE.csv")


LOCAL_METABOLITE_BINARY_FLAGS = {
    "dmstatus",
    "dyslipidemia",
    "hypertension",
    "prevliverdisease",
    "prevckd",
    "prevchd",
    "prevstroke",
    "prevvenousthrom",
    "prevperiart",
    "prevasthma",
    "prevcopd",
    "obesity",
    "abobesity",
    "eversmoker",
    "currentsmoker",
    "everdrinker",
    "currentdrinker",
    "above2pa",
    "hypercholesterolemia",
}

LOCAL_METABOLITE_NUMERIC_BIOMARKERS = {
    "screa_umol",
    "egfr",
    "whr",
    "chol_mmol",
    "ldl",
    "triglycerides",
    "apolipob",
    "hdl",
    "apolipoa",
    "diabp",
    "sysbp",
    "glucose",
    "alt",
    "ast",
    "ggt",
    "albumin",
    "alp",
    "crp",
    "urate",
    "wbc",
    "hemoglobin",
    "plt",
}


def metabolite_local_name(row: dict[str, str]) -> str:
    return (row.get("column_name") or "").lower()


def is_metabolite_local_eligibility_name(name: str) -> bool:
    return name.endswith("_eligebility") or name.endswith("_eligibility")


def is_metabolite_local_outcome_name(name: str, row: dict[str, str]) -> bool:
    role = row.get("feature_role", "")
    return (
        "outcome" in role
        or re.match(r"^(incident|prior)_", name) is not None
        or name.endswith("_date_2104")
        or name.endswith("_2104")
    )


def metabolite_local_semantic_function(row: dict[str, str]) -> str:
    name = metabolite_local_name(row)
    if not is_metabolite_local_derived(row):
        return ""
    if name == "_merge" or is_metabolite_local_eligibility_name(name):
        return "metadata"
    if (
        name.startswith("ts_")
        or name.startswith("followup_")
        or is_metabolite_local_outcome_name(name, row)
        or name in {"gender", "baselineage"}
        or name in LOCAL_METABOLITE_BINARY_FLAGS
        or name in LOCAL_METABOLITE_NUMERIC_BIOMARKERS
    ):
        return "primary"
    return "unknown"


def metabolite_local_value_type(row: dict[str, str]) -> str:
    name = metabolite_local_name(row)
    if name.startswith("ts_") or name.endswith("_date_2104"):
        return "date"
    if name == "baselineage":
        return "integer"
    if (
        name == "_merge"
        or name == "gender"
        or is_metabolite_local_eligibility_name(name)
        or re.match(r"^(incident|prior)_", name)
        or (name.endswith("_2104") and not name.endswith("_date_2104"))
        or name in LOCAL_METABOLITE_BINARY_FLAGS
    ):
        return "categorical"
    if name.startswith("followup_") or name in LOCAL_METABOLITE_NUMERIC_BIOMARKERS:
        return "continuous numeric"
    return "mixed/unknown"


def metabolite_local_derivation_explanation(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    first = rows[0]
    if first.get("modality_category") == "metabolomics_nmr":
        return "Verified UKB NMR metabolomics fields exported as local z-score columns."
    if is_metabolite_unnamed_v_zscore(first):
        return "Numeric local z-score columns, but no local metadata here resolves v170-v250 to biological feature names."
    if not is_metabolite_local_derived(first):
        return ""
    names = {metabolite_local_name(row) for row in rows}
    if all(name.startswith("ts_") for name in names):
        return "Local date-formatted copies/transforms of UKB date fields, including assessment, death, hospital diagnosis/procedure, and ESRD report dates."
    if any(name == "_merge" or is_metabolite_local_eligibility_name(name) for name in names):
        return "Mixed local block: verified covariates/status fields plus merge and eligibility metadata. Column annotations show the per-column type/function."
    if all(name.startswith("followup_") for name in names):
        return "Local follow-up or censoring times in years for curated disease outcomes."
    if any(name in LOCAL_METABOLITE_NUMERIC_BIOMARKERS for name in names):
        return "Locally prepared clinical biomarker/risk-factor variables; names and sampled values support continuous numeric interpretation."
    if names == {"baselineage"}:
        return "Locally prepared baseline age covariate; sampled values support integer interpretation."
    if any(name == "gender" or name in LOCAL_METABOLITE_BINARY_FLAGS for name in names):
        return "Locally prepared categorical covariate/status flag. The field meaning is clear from the name and values, but the exact derivation rule is not in the available metadata."
    if any(is_metabolite_local_outcome_name(metabolite_local_name(row), row) for row in rows):
        return "Locally curated disease status/date variables derived for outcome modeling."
    return "Local derived METABOLITE fields whose exact construction was not recoverable from the available metadata."


def metabolite_local_category(row: dict[str, str]) -> tuple[str, str, str]:
    name = row["column_name"]
    role = row["feature_role"]
    lower = name.lower()
    if "outcome" in role or re.match(r"^(incident|prior)_", lower) or lower.endswith("_date_2104"):
        return "Health records data", "Curated disease labels", f"{disease_family(lower)} curated outcome variables"
    if lower.startswith("followup_"):
        return "Health records data", "Follow-up / censoring time", f"{disease_family(lower)} follow-up/censoring time"
    if lower.startswith("ts_"):
        return "Health records data", "Transformed UKB date fields", "Timestamp-transformed UKB fields"
    if any(k in lower for k in ["screa", "egfr", "chol", "ldl", "hdl", "triglycer", "apolipo", "diabp", "sysbp", "glucose", "alt", "ast", "ggt", "albumin", "alp", "crp", "urate", "wbc", "hemoglobin", "plt"]):
        return "Biomarker data", "Derived clinical biomarkers", "Derived clinical biomarkers and risk factors"
    if any(k in lower for k in ["obesity", "whr", "above2pa", "bmi", "weight", "height"]):
        return "Physical measurements data", "Derived anthropometry/risk factors", "Derived anthropometry and activity risk factors"
    if any(k in lower for k in ["drinker", "smok", "alcohol"]):
        return "Demographic and lifestyle data", "Lifestyle / medication", "Derived lifestyle risk factors"
    if lower.startswith("prev"):
        return "Health records data", "Prevalent disease covariates", f"{disease_family(lower)} prevalent disease variables"
    return "Demographic and lifestyle data", "Local covariates and eligibility metadata", "METABOLITE local covariates and eligibility"


def humanize_metabolite_local_name(row: dict[str, str]) -> str:
    name = row["column_name"]
    lower = name.lower()
    if lower.startswith("followup_"):
        return f"{disease_family(lower)} follow-up/censoring time ({name})"
    if lower.startswith("ts_"):
        return f"Timestamp-transformed UKB field ({name})"
    if "outcome" in row["feature_role"] or re.match(r"^(incident|prior)_", lower) or lower.endswith("_date_2104"):
        return f"{disease_family(lower)} curated outcome variable ({name})"
    label = name.replace("_", " ")
    replacements = {
        "baselineage": "Baseline age",
        "currentdrinker": "Current drinker",
        "currentsmoker": "Current smoker",
        "everdrinker": "Ever drinker",
        "eversmoker": "Ever smoker",
        "dmstatus": "Diabetes status",
        "dyslipidemia": "Dyslipidemia status",
        "hypertension": "Hypertension status",
        "egfr": "eGFR",
        "ldl": "LDL",
        "hdl": "HDL",
        "crp": "CRP",
        "wbc": "white blood cell count",
        "plt": "platelets",
        "sysbp": "systolic blood pressure",
        "diabp": "diastolic blood pressure",
        "whr": "waist-to-hip ratio",
    }
    return replacements.get(lower, label)


def scientific_domain(row: dict[str, str]) -> str:
    modality = row["modality_category"]
    desc = row["description"].lower()
    col = row["column_name"].lower()
    field_id = row["field_id"]

    if is_metabolite_local_derived(row):
        area, req_type, concept = metabolite_local_category(row)
        if area == "Health records data":
            return "Health outcomes and temporality"
        if area == "Biomarker data":
            return "Clinical biomarkers"
        if area == "Physical measurements data":
            return "Conventional risk factors"
        if area == "Demographic and lifestyle data":
            return "Lifestyle and demographic covariates"
        return "METABOLITE local derived metadata"
    if field_id in PAIR_FIELDS and row.get("_paired_ready") == "true":
        return PAIR_FIELDS[field_id]["domain"]
    if field_id in PAIR_FIELDS:
        return "Health outcomes"
    if modality == "metabolomics_nmr":
        return "Molecular profiling"
    if is_metabolite_unnamed_v_zscore(row):
        return "Unverified local-derived variables"
    if is_genetic_or_genomic(desc, field_id):
        return "Genetic data and metadata"
    if is_physical_activity_field(desc, field_id):
        return "Conventional risk factors"
    if modality == "clinical_biomarker_blood_urine":
        concept = clinical_biomarker_concept(desc)
        if concept in {
            "White blood cells and differential",
            "Red blood cells and haemoglobin",
            "Platelet count and indices",
            "Reticulocyte measures",
            "Blood count panel",
        }:
            return "Clinical biomarkers - haematology"
        if concept in {"Kidney function and urine chemistry", "Urine electrolytes and result flags"}:
            return "Clinical biomarkers - kidney function"
        if concept == "Lipid biomarkers":
            return "Clinical biomarkers - lipids"
        if concept == "Glycemic biomarkers":
            return "Clinical biomarkers - glycemia"
        if concept == "Liver enzymes and bilirubin":
            return "Clinical biomarkers - liver function"
        if concept == "Protein, inflammation, and growth-factor biomarkers":
            return "Clinical biomarkers - proteins/inflammation"
        if concept == "Sex hormone biomarkers":
            return "Clinical biomarkers - hormones"
        if concept == "Mineral and vitamin biomarkers":
            return "Clinical biomarkers - minerals/vitamins"
        return "Clinical biomarkers"
    if modality in {"linked_health_outcome", "derived_or_linked_outcome", "eligibility_flag"}:
        if any(k in desc + " " + col for k in ["cancer", "stroke", "dementia", "parkinson", "death", "renal", "copd", "asthma", "diabetes", "t2d", "cvd", "heart", "glaucoma", "fracture"]):
            return "Health outcomes"
        return "Health outcomes and eligibility"
    if modality in {"brain_mri_imaging", "dxa_body_composition_imaging", "eye_imaging"}:
        return "Imaging"
    if modality == "eye_vision" and is_true_eye_vision(row):
        return "Eye and vision phenotypes"
    if modality == "eye_vision":
        if is_mental_health(desc):
            return "Mental health questionnaire phenotypes"
        if is_self_reported_history(desc):
            return "Self-reported medical history"
        return "Unclassified / Other UKB phenotypes"
    if modality == "cognitive_assessment":
        return "Cognitive and neuropsychological phenotypes"
    if modality in {"demographics_admin", "lifestyle_medication_diet", "online_dietary_recall_nutrients", "physical_measure", "bioimpedance_body_composition"}:
        return "Conventional risk factors"
    if modality == "activity_location_or_genetic_metadata":
        if is_genetic_or_genomic(desc, field_id):
            return "Genetic data and metadata"
        if is_environmental_or_location(desc):
            return "Environmental and location exposures"
        return "Activity and location metadata"
    if modality == "alignment_id":
        return "Data alignment"
    if modality == "row_index":
        return "Data alignment"
    if modality == "other":
        if is_genetic_or_genomic(desc, field_id):
            return "Genetic data and metadata"
        if is_brain_bulk_image_or_pipeline(desc) or is_dxa_bulk_image(desc):
            return "Imaging"
        if is_environmental_or_location(desc):
            return "Environmental and location exposures"
        if is_ophthalmic_measurement(desc):
            return "Eye and vision phenotypes"
        if is_death_record(desc) or is_hospital_record(desc):
            return "Health records"
        if is_mental_health(desc):
            return "Mental health questionnaire phenotypes"
        if is_cognitive_task(desc):
            return "Cognitive and neuropsychological phenotypes"
        if lifestyle_concept(desc) != "Lifestyle questionnaire" or is_diet_or_food(desc) or is_medication_or_supplement(desc):
            return "Lifestyle, medication, and diet"
        if is_family_history(desc) or is_sociodemographic_or_psychosocial(desc):
            return "Demographics, family history, and psychosocial"
        if is_bioimpedance(desc):
            return "Conventional risk factors"
        if is_self_reported_history(desc):
            return "Self-reported medical history"
        if is_followup_or_censoring(desc + " " + col):
            return "Follow-up and censoring metadata"
        if is_assessment_device_or_qc(desc):
            return "Assessment device and QC metadata"
    return "Unclassified / Other UKB phenotypes"


def concept_observation_attribute(row: dict[str, str]) -> tuple[str, str, str]:
    field_id = row["field_id"]
    desc = row["description"] or row["column_name"]
    modality = row["modality_category"]
    col = row["column_name"]

    if is_metabolite_local_derived(row):
        _area, _req_type, concept = metabolite_local_category(row)
        return concept, "Local derived METABOLITE attribute", humanize_metabolite_local_name(row)
    if field_id in PAIR_FIELDS and row.get("_paired_ready") == "true":
        info = PAIR_FIELDS[field_id]
        return info["concept"], info["observation"], info["attribute"]
    if field_id in PAIR_FIELDS:
        desc = row["description"] or row["column_name"]
        return PAIR_FIELDS[field_id]["concept"], "Unpaired repeated field", desc

    if modality == "metabolomics_nmr":
        return "NMR metabolomics", "Metabolite or QC feature", desc
    if is_metabolite_unnamed_v_zscore(row):
        return "Unnamed local derived z-score variables", "Local METABOLITE z-score block", desc
    if is_genetic_or_genomic(desc, field_id):
        return genetic_concept(desc, field_id), "Genetic file or metadata attribute", desc
    if is_physical_activity_field(desc, field_id):
        return "Lifestyle, medication, and diet questionnaires", "Physical activity questionnaire", desc
    if modality == "brain_mri_imaging":
        if any(k in desc.lower() for k in ["dicom", "nifti", "images"]):
            return "Brain MRI bulk image files", "Image file identifier", desc
        return "Brain MRI derived phenotypes", "Derived MRI scalar feature", desc
    if modality == "dxa_body_composition_imaging":
        return "DXA/body-composition imaging", "Derived DXA scalar feature", desc
    if modality == "eye_imaging":
        if any(k in desc.lower() for k in ["image", "file", "oct"]):
            return "Eye imaging files and acquisition", "Eye imaging file/acquisition attribute", desc
        return "Eye imaging acquisition metadata", "Acquisition or QC attribute", desc
    if modality == "clinical_biomarker_blood_urine":
        return clinical_biomarker_concept(desc), "Laboratory measurement", desc
    if modality == "linked_health_outcome":
        return "Algorithmic linked endpoints", "Linked endpoint attribute", desc
    if modality == "derived_or_linked_outcome":
        return "Disease outcomes and histories", "Outcome/status/date attribute", desc
    if modality == "eligibility_flag":
        return "Analysis eligibility flags", "Eligibility attribute", desc
    if modality == "demographics_admin":
        return "Demographics and assessment visit metadata", "Participant/visit attribute", desc
    if modality == "activity_location_or_genetic_metadata":
        if is_genetic_or_genomic(desc, field_id):
            return genetic_concept(desc, field_id), "Genetic file or metadata attribute", desc
        if is_environmental_or_location(desc):
            return environmental_concept(desc), "Environmental or location exposure", desc
        return "Activity and location metadata", "Activity/location metadata attribute", desc
    if modality == "lifestyle_medication_diet":
        return "Lifestyle, medication, and diet questionnaires", "Questionnaire or medication attribute", desc
    if modality == "online_dietary_recall_nutrients":
        return "Online dietary recall nutrients", "Dietary recall nutrient attribute", desc
    if modality == "physical_measure":
        return "Physical measurements", "Assessment measurement", desc
    if modality == "bioimpedance_body_composition":
        return "Bioimpedance body composition", "Body-composition measurement", desc
    if modality == "cognitive_assessment":
        return "Cognitive assessment", "Cognitive task attribute", desc
    if modality == "eye_vision":
        if is_true_eye_vision(row):
            return "Ophthalmic measurements", ophthalmic_concept(desc), desc
        if is_mental_health(desc):
            return "Mental health questionnaire", "Mental health symptom/status attribute", desc
        if is_self_reported_history(desc):
            return "Self-reported medical history", "Repeated self-reported event/history", desc
        return "Unclassified UKB field groups", "UKB field", desc
    if modality == "alignment_id":
        return "Participant identifier", "Dataset alignment key", col
    if modality == "row_index":
        return "File row order metadata", "Sequential row index", desc
    if modality == "other":
        if is_genetic_or_genomic(desc, field_id):
            return genetic_concept(desc, field_id), "Genetic file or metadata attribute", desc
        if is_brain_bulk_image_or_pipeline(desc):
            return "Brain MRI bulk pipeline files", "Bulk image/pipeline file identifier", desc
        if is_dxa_bulk_image(desc):
            return "DXA bulk image files", "Bulk image file identifier", desc
        if is_environmental_or_location(desc):
            return environmental_concept(desc), "Environmental or location exposure", desc
        if is_ophthalmic_measurement(desc):
            return "Ophthalmic measurements", ophthalmic_concept(desc), desc
        if is_death_record(desc):
            return "Death records", "Death date/cause attribute", desc
        if is_hospital_record(desc):
            return "Hospital records", "Hospital diagnosis/procedure/admin attribute", desc
        if is_mental_health(desc):
            return "Mental health questionnaire", "Mental health symptom/status attribute", desc
        if is_cognitive_task(desc):
            return "Cognitive assessment", "Cognitive task attribute", desc
        if lifestyle_concept(desc) != "Lifestyle questionnaire":
            return "Lifestyle questionnaire", lifestyle_concept(desc), desc
        if is_diet_or_food(desc):
            return "Diet and food questionnaire", "Dietary self-report attribute", desc
        if is_medication_or_supplement(desc):
            return "Medication and supplement self-report", "Medication/supplement attribute", desc
        if is_family_history(desc):
            return "Family history", "Family illness attribute", desc
        if is_sociodemographic_or_psychosocial(desc):
            return "Sociodemographics and psychosocial", "Socioeconomic/psychosocial attribute", desc
        if is_bioimpedance(desc):
            return "Bioimpedance body composition", "Manual bioimpedance attribute", desc
        if is_self_reported_history(desc):
            return "Self-reported medical history", "Repeated self-reported event/history", desc
        if is_followup_or_censoring(desc + " " + col):
            return "Follow-up / censoring time", "Follow-up/censoring attribute", desc
        if is_assessment_device_or_qc(desc):
            return "Assessment device and QC metadata", "Device/method/QC attribute", desc
    return "Unclassified UKB field groups", "UKB field", desc


def requested_area_type(row: dict[str, str]) -> tuple[str, str]:
    modality = row["modality_category"]
    desc = (row["description"] + " " + row["column_name"]).lower()
    field_id = row["field_id"]

    if is_metabolite_local_derived(row):
        area, req_type, _concept = metabolite_local_category(row)
        return area, req_type
    if modality == "metabolomics_nmr":
        return "Biomarker data", "Metabolomic biomarkers"
    if modality == "clinical_biomarker_blood_urine":
        if is_blood_count_biomarker(desc):
            return "Biomarker data", "Blood count"
        return "Biomarker data", "Biochemistry biomarkers"
    if modality == "brain_mri_imaging":
        return "Imaging data", "Brain MRI"
    if modality == "dxa_body_composition_imaging":
        return "Imaging data", "DXA/DEXA"
    if modality == "eye_imaging":
        return "Imaging data", "Retina/OCT"
    if modality == "eye_vision":
        if is_true_eye_vision(row):
            return "Physical measurements data", "Vision / ophthalmic measurements"
        if is_mental_health(desc):
            return "Questionnaire data", "Mental health / wellbeing"
        if is_self_reported_history(desc):
            return "Questionnaire data", "Self-reported medical history"
        return "Other retrieved data", "Other UKB phenotypes"
    if modality == "cognitive_assessment":
        return "Questionnaire data", "Cognitive function"
    if modality == "physical_measure" and is_genetic_or_genomic(desc, field_id):
        return "Genetic data", genetic_requested_type(desc, field_id)
    if modality in {"physical_measure", "bioimpedance_body_composition"}:
        return "Physical measurements data", "General physical measures"
    if modality in {"lifestyle_medication_diet", "online_dietary_recall_nutrients"}:
        life_concept = lifestyle_concept(desc)
        if life_concept == "Physical activity questionnaire":
            return "Demographic and lifestyle data", "Lifestyle / medication"
        if modality == "online_dietary_recall_nutrients" or life_concept in {"Diet and food intake questionnaire", "Alcohol use"}:
            return "Questionnaire data", "Diet / food preferences"
        return "Demographic and lifestyle data", "Lifestyle / medication"
    if modality == "demographics_admin":
        if is_genetic_or_genomic(desc, field_id):
            return "Genetic data", genetic_requested_type(desc, field_id)
        return "Demographic and lifestyle data", "Sociodemographics / assessment metadata"
    if modality == "activity_location_or_genetic_metadata":
        if is_genetic_or_genomic(desc, field_id):
            return "Genetic data", genetic_requested_type(desc, field_id)
        if is_physical_activity_field(desc, field_id):
            return "Demographic and lifestyle data", "Lifestyle / medication"
        if is_environmental_or_location(desc):
            return "Environmental data", "Environmental exposures"
        return "Demographic and lifestyle data", "Activity/location metadata"
    if modality in {"linked_health_outcome", "derived_or_linked_outcome", "eligibility_flag"} or field_id in PAIR_FIELDS:
        if "death" in desc:
            return "Health records data", "Death data"
        if any(k in desc for k in ["non-cancer illness", "non cancer illness", "noncancer illness"]):
            return "Health records data", "Self-reported non-cancer illness"
        if "cancer" in desc:
            return "Health records data", "Cancer data"
        if field_id in PAIR_FIELDS or any(k in desc for k in ["icd", "opcs", "in-patient", "hospital", "diagnosis", "operation"]):
            return "Health records data", "Hospital inpatient data"
        if any(k in desc for k in ["420", "source of", "date of", "report"]):
            return "Health records data", "Linked disease reports"
        return "Health records data", "Derived disease labels"
    if modality == "alignment_id":
        return "Data management", "Participant ID / alignment"
    if modality == "row_index":
        return "Data management", "Row index / file order"
    if modality == "other":
        if is_metabolite_unnamed_v_zscore(row):
            return "Other retrieved data", "Unverified local z-score variables"
        if is_genetic_or_genomic(desc, field_id):
            return "Genetic data", genetic_requested_type(desc, field_id)
        if is_brain_bulk_image_or_pipeline(desc):
            return "Imaging data", "Brain MRI"
        if is_dxa_bulk_image(desc):
            return "Imaging data", "DXA/DEXA"
        if is_environmental_or_location(desc):
            return "Environmental data", "Environmental/location exposures"
        if is_ophthalmic_measurement(desc):
            return "Physical measurements data", "Vision / ophthalmic measurements"
        if is_death_record(desc):
            return "Health records data", "Death data"
        if is_hospital_record(desc):
            return "Health records data", "Hospital inpatient data"
        if is_mental_health(desc):
            return "Questionnaire data", "Mental health / wellbeing"
        if is_cognitive_task(desc):
            return "Questionnaire data", "Cognitive function"
        life_concept = lifestyle_concept(desc)
        if life_concept != "Lifestyle questionnaire":
            if life_concept in {"Diet and food intake questionnaire", "Alcohol use"}:
                return "Questionnaire data", "Diet / food preferences"
            return "Demographic and lifestyle data", "Lifestyle / medication"
        if is_diet_or_food(desc):
            return "Questionnaire data", "Diet / food preferences"
        if is_medication_or_supplement(desc):
            return "Demographic and lifestyle data", "Lifestyle / medication"
        if is_family_history(desc):
            return "Demographic and lifestyle data", "Family history"
        if is_sociodemographic_or_psychosocial(desc):
            return "Demographic and lifestyle data", "Sociodemographics / psychosocial"
        if is_bioimpedance(desc):
            return "Physical measurements data", "Bioimpedance body composition"
        if is_self_reported_history(desc):
            return "Questionnaire data", "Self-reported medical history"
        if is_followup_or_censoring(desc):
            return "Health records data", "Follow-up / censoring time"
        if is_assessment_device_or_qc(desc):
            return "Data management", "Assessment device/QC metadata"
    return "Other retrieved data", "Other UKB phenotypes"


def display_concept(row: dict[str, str], concept: str, observation: str, attribute: str) -> str:
    modality = row["modality_category"]
    desc = (row["description"] or row["column_name"]).lower()
    field_id = row["field_id"]
    if is_metabolite_local_derived(row):
        _area, _req_type, concept_label = metabolite_local_category(row)
        return concept_label
    if field_id in PAIR_FIELDS and row.get("_paired_ready") == "true":
        return observation
    if field_id in PAIR_FIELDS:
        return f"Unpaired {attribute.lower()}"
    if modality == "metabolomics_nmr":
        return "NMR metabolite panel"
    if is_genetic_or_genomic(desc, field_id):
        return genetic_concept(desc, field_id)
    if is_physical_activity_field(desc, field_id):
        return "Physical activity questionnaire"
    if modality == "brain_mri_imaging":
        if any(k in desc for k in ["dicom", "nifti", "images"]):
            return "Brain MRI image files"
        if "volume" in desc:
            return "Brain regional volumes"
        if any(k in desc for k in ["weighted-mean", "tract", "fa", "md", "isovf", "od"]):
            return "Diffusion MRI tract phenotypes"
        return "Brain MRI derived phenotypes"
    if modality == "dxa_body_composition_imaging":
        if any(k in desc for k in ["bone", "bmd", "bmc"]):
            return "Bone density and mineral content"
        if any(k in desc for k in ["fat", "lean", "mass"]):
            return "Body composition"
        return "DXA-derived measurements"
    if modality == "eye_imaging":
        if "fundus" in desc:
            return "Fundus retinal images"
        if "oct" in desc:
            return "OCT imaging"
        return "Eye imaging files and acquisition"
    if modality == "clinical_biomarker_blood_urine":
        return clinical_biomarker_concept(desc)
    if modality == "physical_measure":
        if any(k in desc for k in ["height", "weight", "bmi", "waist", "hip"]):
            return "Body size and anthropometry"
        if "blood pressure" in desc or "pulse" in desc:
            return "Blood pressure and pulse"
        return "Assessment physical measurements"
    if modality == "bioimpedance_body_composition":
        return "Bioimpedance body-composition traits"
    if modality in {"linked_health_outcome", "derived_or_linked_outcome", "eligibility_flag"}:
        if modality == "linked_health_outcome":
            endpoint = linked_endpoint_name(row["description"] or row["column_name"])
            return f"{disease_family(endpoint)} linked disease reports"
        if any(k in desc for k in ["incident", "prior", "prevalent", "elig"]):
            return f"{disease_family(desc + ' ' + row['column_name'])} curated labels"
        if "date" in desc or "source" in desc:
            return f"{disease_family(desc + ' ' + row['column_name'])} date/source fields"
        return f"{disease_family(desc + ' ' + row['column_name'])} outcome fields"
    if modality == "cognitive_assessment":
        return "Cognitive task results"
    if modality == "eye_vision":
        if is_true_eye_vision(row):
            return ophthalmic_concept(desc)
        if is_mental_health(desc):
            return "Mental health questionnaire"
        if is_self_reported_history(desc):
            return "Self-reported diagnosis/procedure history"
        return "Unclassified UKB field groups"
    if modality in {"lifestyle_medication_diet", "online_dietary_recall_nutrients"}:
        return lifestyle_concept(desc)
    if modality == "demographics_admin":
        return "Participant and visit metadata"
    if modality == "row_index":
        return "File row order metadata"
    if modality == "activity_location_or_genetic_metadata":
        if is_genetic_or_genomic(desc, field_id):
            return genetic_concept(desc, field_id)
        if is_environmental_or_location(desc):
            return environmental_concept(desc)
        return "Activity/location metadata"
    if modality == "other":
        if is_metabolite_unnamed_v_zscore(row):
            return "Unnamed local derived z-score variables"
        if is_genetic_or_genomic(desc, field_id):
            return genetic_concept(desc, field_id)
        if is_brain_bulk_image_or_pipeline(desc):
            return "Brain MRI bulk pipeline files"
        if is_dxa_bulk_image(desc):
            return "DXA bulk image files"
        if is_environmental_or_location(desc):
            return environmental_concept(desc)
        if is_ophthalmic_measurement(desc):
            return ophthalmic_concept(desc)
        if is_death_record(desc):
            return "Death records"
        if is_hospital_record(desc):
            return "Hospital diagnosis/procedure records"
        if is_mental_health(desc):
            return "Mental health questionnaire"
        if is_cognitive_task(desc):
            return "Cognitive task results"
        if lifestyle_concept(desc) != "Lifestyle questionnaire":
            return lifestyle_concept(desc)
        if is_diet_or_food(desc):
            return "Diet and food self-report"
        if is_medication_or_supplement(desc):
            return "Medication and supplement self-report"
        if is_family_history(desc):
            return "Family history of illness"
        if is_sociodemographic_or_psychosocial(desc):
            return "Sociodemographic and psychosocial context"
        if is_bioimpedance(desc):
            return "Bioimpedance body-composition traits"
        if is_self_reported_history(desc):
            return "Self-reported diagnosis/procedure history"
        if is_followup_or_censoring(desc):
            return "Follow-up / censoring metadata"
        if is_assessment_device_or_qc(desc):
            return "Assessment device/QC metadata"
    return concept


def infer_storage_pattern(rows: list[dict[str, str]]) -> str:
    first = rows[0]
    field_id = first["field_id"]
    modality = first["modality_category"]
    desc = first["description"].lower()
    instances = {r["instance"] for r in rows if r["instance"] != ""}
    arrays = {r["array"] for r in rows if r["array"] != ""}

    if modality == "alignment_id":
        return "alignment_metadata"
    if field_id in PAIR_FIELDS and first.get("_paired_ready") == "true":
        return "paired_repeated_array"
    if modality == "metabolomics_nmr":
        return "molecular_feature"
    if is_metabolite_unnamed_v_zscore(first):
        return "local_derived_zscore_block"
    if modality == "eligibility_flag":
        return "eligibility_flag"
    if modality in {"linked_health_outcome", "derived_or_linked_outcome"}:
        return "curated_or_linked_outcome"
    if modality == "eye_imaging" and any(k in desc for k in ["image", "data file", "oct image", "file"]):
        return "bulk_image_id"
    if modality == "brain_mri_imaging" and any(k in desc for k in ["dicom", "nifti", "images"]):
        return "bulk_image_id"
    if is_brain_bulk_image_or_pipeline(desc) or is_dxa_bulk_image(desc):
        return "bulk_image_id"
    if is_genetic_or_genomic(desc, field_id) and any(k in desc for k in ["file", "files", "cram", "vcf", "plink", "cel"]):
        return "bulk_file_id"
    if modality in {"brain_mri_imaging", "dxa_body_composition_imaging"}:
        return "derived_imaging_trait"
    if len(instances) > 1 and len(arrays) > 1:
        return "repeated_visit_array"
    if len(instances) > 1:
        return "repeated_visit"
    if len(arrays) > 1:
        return "repeated_array"
    return "scalar"


def storage_pattern_label(storage: str) -> str:
    return COLUMN_LAYOUT_LABELS.get(storage, storage.replace("_", " "))


def storage_pattern_detail(storage: str) -> str:
    details = {
        "scalar": "The semantic group has one retrieved raw column.",
        "repeated_visit": "The semantic group has multiple UKB instance columns for the same field, usually assessment instances 0/1/2/3. Each raw column still stores one value per participant.",
        "repeated_array": "The semantic group has multiple repeated-entry columns for the same visit or record family. Each entry stores one value per participant.",
        "repeated_visit_array": "The semantic group has both UKB assessment instances and repeated-entry positions.",
        "paired_repeated_array": "Two repeated-entry field families are paired by instance and entry number, such as diagnosis code and first diagnosis date.",
        "bulk_image_id": "The retrieved column stores an image or imaging-pipeline file identifier, not the image pixels themselves.",
        "bulk_file_id": "The retrieved column stores a file identifier for bulk data such as CRAM/VCF/PLINK/CEL assets.",
        "derived_imaging_trait": "The retrieved column stores a derived scalar imaging trait.",
        "molecular_feature": "The semantic group represents a molecular panel or molecular feature block.",
        "curated_or_linked_outcome": "The retrieved column stores a curated or linked health outcome/status/date/source field.",
        "eligibility_flag": "The retrieved column stores an analysis eligibility flag.",
        "alignment_metadata": "The retrieved column stores an ID or namespace used for joining sources.",
        "local_derived_zscore_block": "These columns are local derived z-score variables in METABOLITE.csv, not official UKB field IDs.",
    }
    return details.get(storage, "Column layout inferred from UKB instances, repeated-entry positions, and source metadata.")


def value_representation(rows: list[dict[str, str]]) -> str:
    types = Counter(r["ukb_type"] or "Unknown" for r in rows)
    type_text = "; ".join(f"{k}: {v}" for k, v in sorted(types.items()))
    joined = " ".join(types).lower()
    if all("continuous" in (r["ukb_type"] or "").lower() for r in rows):
        return "One numeric value per participant per raw column."
    if all("categorical" in (r["ukb_type"] or "").lower() for r in rows):
        return "One coded categorical value per participant per raw column."
    if all("integer" in (r["ukb_type"] or "").lower() for r in rows):
        return "One integer value per participant per raw column."
    if "file" in joined or "sequence" in joined:
        return "One file/ID-like value per participant per raw column."
    return f"One value per participant per raw column; UKB types: {type_text}."


def infer_prediction_role(rows: list[dict[str, str]]) -> str:
    joined = " ".join(
        [
            " ".join(r.get(k, "") for k in ["column_name", "feature_role", "modality_category", "description"])
            for r in rows
        ]
    ).lower()
    modality = rows[0]["modality_category"]

    if modality in {"alignment_id", "row_index"}:
        return "alignment_metadata"
    if "eligibility" in joined:
        return "leakage_risk"
    if "followup_" in joined or "outcome_date" in joined or "ts_" in joined:
        return "leakage_risk"
    if "outcome_flag" in joined or "outcome_flag_or_status" in joined:
        return "target_candidate"
    if modality in {"linked_health_outcome", "derived_or_linked_outcome"}:
        return "target_candidate"
    if any(k in joined for k in ["incident_", "prior_", "followup_", "date of death", "date of first", "diagnosis date", "source of", "outcome_date"]):
        return "leakage_risk"
    if modality in {"demographics_admin", "lifestyle_medication_diet", "online_dietary_recall_nutrients", "physical_measure"}:
        return "covariate"
    if modality in {"metabolomics_nmr", "clinical_biomarker_blood_urine", "brain_mri_imaging", "dxa_body_composition_imaging", "eye_imaging", "eye_vision", "bioimpedance_body_composition", "cognitive_assessment"}:
        return "predictor"
    return "metadata"


def leakage_warning(rows: list[dict[str, str]], role: str) -> str:
    text = " ".join(" ".join(r.get(k, "") for k in ["column_name", "feature_role", "description"]) for r in rows).lower()
    if role == "target_candidate":
        return "Candidate target/outcome field; exclude from predictor set unless explicitly modeling labels."
    if role == "leakage_risk":
        return "Potential leakage: encodes disease status, event date, eligibility, source, follow-up, or target-definition logic."
    if any(k in text for k in ["date of first", "source of", "diagnosis", "death", "incident_", "prior_", "followup_"]):
        return "Check temporality before using as a predictor."
    return ""


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    field_id = row["field_id"]
    source_id = row["source_id"]
    modality = row["modality_category"]
    if modality == "alignment_id":
        return (source_id, "alignment_id_namespace")
    if modality == "metabolomics_nmr":
        return (source_id, "nmr_metabolomics_panel")
    if modality == "linked_health_outcome":
        return (source_id, "linked_disease_report", slug(linked_endpoint_name(row["description"] or row["column_name"])))
    if is_metabolite_unnamed_v_zscore(row):
        return (source_id, "metabolite_unnamed_v_zscore_block")
    if is_metabolite_local_derived(row):
        _area, req_type, concept = metabolite_local_category(row)
        if req_type in {
            "Transformed date/time fields",
            "Transformed UKB date fields",
            "METABOLITE local covariates and eligibility",
            "Local covariates and eligibility metadata",
        }:
            return (source_id, "metabolite_local_derived_aggregate", req_type)
        if req_type in {"Follow-up and temporality variables", "Follow-up / censoring time"}:
            return (source_id, "metabolite_followup", concept)
        return (source_id, "metabolite_local_derived", row["column_name"])
    if field_id in PAIR_FIELDS and row.get("_paired_ready") == "true":
        pair = PAIR_FIELDS[field_id]
        return (
            source_id,
            "paired",
            pair["pair_id"],
            pair["attribute"],
            row["value_prefix"],
            row["is_z_score"],
        )
    if field_id:
        return (
            source_id,
            "field",
            field_id,
            modality,
            row["value_prefix"],
            row["is_z_score"],
            row["description"],
        )
    return (source_id, "derived_column", row["column_name"], modality, row["description"])


REPORT_DOMAIN_ORDER = [
    "Metabolomics and molecular biomarkers",
    "Clinical laboratory biomarkers",
    "Brain and neuroimaging",
    "Eye and vision",
    "Body composition and bone",
    "Cardiometabolic physical measures",
    "Clinical outcomes and disease history",
    "Lifestyle and environmental exposures",
    "Cognition and mental health",
    "Demographics and assessment context",
    "Genetics and genomics",
    "Data linkage and source administration",
]

ACQUISITION_TYPES = {
    "Molecular assay",
    "Clinical lab measurement",
    "Assessment measurement",
    "Imaging-derived scalar",
    "Bulk image/file ID",
    "Questionnaire/self-report",
    "Linked health record",
    "Curated local variable",
    "Device/QC metadata",
    "Missing/failure note",
    "Identifier/join key",
    "Administrative/visit field",
}

REPORT_GENERIC_CONCEPTS = {
    "",
    "NA",
    "Other UKB phenotypes",
    "Other UKB phenotypes/metadata",
    "Unclassified / Other UKB phenotypes",
    "Unclassified UKB field groups",
    "UKB field",
    "Disease outcome fields",
    "Outcome dates and sources",
    "Questionnaire and self-report features",
    "Lifestyle questionnaire",
    "Mental health questionnaire",
    "Participant and visit metadata",
    "Genetic metadata",
    "Genetic metadata fields (not genomic data)",
    "Genetic data files and metadata",
    "Assessment device/QC metadata",
    "Activity/location metadata",
    "File row order metadata",
    "General physical measures",
    "Assessment physical measurements",
    "Follow-up / censoring metadata",
}

REPORT_CONCEPT_REPLACEMENTS = {
    "Diet and food intake questionnaire": "Diet and food intake",
    "Physical activity questionnaire": "Physical activity",
    "Sleep questionnaire": "Sleep",
    "Mental health questionnaire": "Mental health",
    "Family history of illness": "Family history",
    "Sociodemographic and psychosocial context": "Sociodemographic and psychosocial context",
    "Self-reported diagnosis/procedure history": "Self-reported diagnoses and procedures",
    "Medication and supplement self-report": "Medication and supplements",
    "Medication and supplements": "Medication and supplements",
    "Online dietary recall nutrients": "Diet and food intake",
    "Bioimpedance body-composition traits": "Bioimpedance body composition",
    "Derived anthropometry/risk factors": "Derived anthropometry and activity risk factors",
    "Derived clinical biomarkers and risk factors": "Derived clinical biomarkers and risk factors",
    "Unnamed local derived z-score variables": "Unverified local z-score variables",
    "Participant and visit metadata": "Participant characteristics and assessment visit",
}

REPORT_TREE_OVERRIDES = {
    "22032": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "22033": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "22034": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "22035": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "22036": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "22037": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "22038": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "22039": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "22040": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "104900": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "104910": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "104920": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Physical activity"),
    "5186": ("Eye and vision", "Device/QC metadata", "Visual acuity testing"),
    "5188": ("Eye and vision", "Device/QC metadata", "Visual acuity testing"),
    "20079": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Diet questionnaire administration"),
    "20080": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Diet questionnaire administration"),
    "20081": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Diet questionnaire administration"),
    "20082": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Diet questionnaire administration"),
    "20083": ("Lifestyle and environmental exposures", "Questionnaire/self-report", "Diet questionnaire administration"),
    "20014": ("Clinical outcomes and disease history", "Linked health record", "Operation history"),
    "41213": ("Clinical outcomes and disease history", "Linked health record", "Hospital inpatient discharge"),
    "41250": ("Clinical outcomes and disease history", "Linked health record", "Hospital inpatient discharge"),
    "12141": ("Body composition and bone", "Device/QC metadata", "DXA assessment status"),
    "12253": ("Body composition and bone", "Device/QC metadata", "DXA assessment status"),
    "12254": ("Body composition and bone", "Device/QC metadata", "DXA assessment status"),
    "4290": ("Cognition and mental health", "Device/QC metadata", "Touchscreen/cognitive task display metadata"),
    "22700": ("Lifestyle and environmental exposures", "Curated local variable", "Home location and urbanicity"),
}


def field_id_set(group: dict[str, object]) -> set[str]:
    return {item.strip() for item in str(group.get("field_ids", "")).replace(",", ";").split(";") if item.strip()}


def group_lower_text(group: dict[str, object]) -> str:
    return " ".join(
        str(group.get(k, ""))
        for k in [
            "requested_area",
            "requested_type",
            "scientific_domain",
            "concept",
            "observation",
            "display_concept",
            "attribute",
            "tree_leaf_label",
            "field_ids",
            "raw_columns",
        ]
    ).lower()


def group_leaf_lower_text(group: dict[str, object]) -> str:
    return " ".join(
        str(group.get(k, ""))
        for k in [
            "attribute",
            "tree_leaf_label",
            "field_ids",
            "raw_columns",
        ]
    ).lower()


def reviewed_tree_override(group: dict[str, object]) -> tuple[str, str, str] | None:
    for field_id in field_id_set(group):
        if field_id in REPORT_TREE_OVERRIDES:
            return REPORT_TREE_OVERRIDES[field_id]
    return None


def text_has_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def text_has_physical_measure(text: str) -> bool:
    phrase_patterns = [
        r"\bheight\b",
        r"\bstanding height\b",
        r"\bseated height\b",
        r"\bsitting height\b",
        r"\bweight\b",
        r"\bbody mass index\b",
        r"\bbmi\b",
        r"\bwaist circumference\b",
        r"\bhip circumference\b",
        r"\bwaist-to-hip\b",
        r"\bblood pressure\b",
        r"\bpulse\b",
    ]
    return any(re.search(pattern, text) for pattern in phrase_patterns)


def text_is_self_report_or_history(text: str) -> bool:
    return text_has_any(
        text,
        [
            "self-reported",
            "self reported",
            "diagnosed",
            "ever had",
            "which eye",
            "wearing glasses",
            "contact lenses",
            "medication",
            "treatment/medication",
            "satisfaction",
            "friendships",
            "family relationship",
            "depression",
            "anxiety",
            "illness",
            "cancer",
            "operation",
            "procedure",
            "surgery",
        ],
    )


def text_is_device_or_qc(text: str) -> bool:
    return text_has_any(
        text,
        [
            "device",
            "qc",
            "quality control",
            "unreliable",
            "method",
            "measurement completed",
            "measuring method",
            "believed safe",
            "duration screen displayed",
            "time since interview start",
            "direct or mirror view",
            "distance of viewer",
            "applanation curve",
            "displayed letters",
            "letters shown",
            "number of letters shown",
            "final number of letters displayed",
            "visual acuity measured",
            "index of best",
            "result unreliable",
            "operator indicated",
            "measurement batch",
            "measurement plate",
            "measurement well",
            "recommended genomic analysis exclusions",
            "heterozygosity",
            "dna concentration",
            "sex inference",
            "sex chromosome aneuploidy",
            "used in genetic principal components",
            "unrelatedness indicator",
        ],
    )


def text_is_genetic_file_id(text: str) -> bool:
    return text_has_any(
        text,
        [
            "genotype results",
            "imputation and haplotype results",
            "variant calls",
            "mitochondrial genotype results",
            "exome",
        ],
    )


def report_domain_for_group(group: dict[str, object]) -> str:
    override = reviewed_tree_override(group)
    if override:
        return override[0]
    modality = str(group.get("modality_category", ""))
    area = str(group.get("requested_area", ""))
    req_type = str(group.get("requested_type", ""))
    concept = str(group.get("display_concept") or group.get("concept") or "")
    text = group_lower_text(group)

    if modality in {"alignment_id", "row_index"} or area == "Data management":
        return "Data linkage and source administration"
    if req_type in {"Derived anthropometry/risk factors"}:
        return "Cardiometabolic physical measures"
    if text_has_any(text, ["fi5 :", "fluid intelligence", "numeric memory", "trail making", "symbol digit", "pairs test"]):
        return "Cognition and mental health"
    if text_has_any(text, ["depression", "anxiety", "mental health"]):
        return "Cognition and mental health"
    if text_has_any(text, ["family relationship", "friendships satisfaction", "relationship satisfaction"]):
        return "Demographics and assessment context"
    if area == "Genetic data" or "genetic" in text or "genotype" in text or "whole-genome" in text or "exome" in text:
        return "Genetics and genomics"
    if modality == "metabolomics_nmr" or "metabolomic" in text or "nmr" in text:
        return "Metabolomics and molecular biomarkers"
    if modality == "clinical_biomarker_blood_urine" or req_type in {"Biochemistry biomarkers", "Blood count", "Derived clinical biomarkers"}:
        return "Clinical laboratory biomarkers"
    if modality == "brain_mri_imaging" or "brain mri" in text or "rfmri" in text or "tfmri" in text:
        return "Brain and neuroimaging"
    if modality in {"eye_imaging", "eye_vision"} or "ophthalmic" in text or "fundus" in text or "oct" in text or "visual acuity" in text or "glasses" in text or "contact lenses" in text:
        return "Eye and vision"
    if modality in {"dxa_body_composition_imaging", "bioimpedance_body_composition"} or "dxa" in text or "bone" in text or "bmd" in text or "bmc" in text or "fat mass" in text or "fat-free" in text or "impedance" in text:
        return "Body composition and bone"
    if text_has_any(text, ["high blood pressure diagnosed", "medication for cholesterol", "medication for cholesterol, blood pressure", "diabetes"]):
        return "Clinical outcomes and disease history"
    if area == "Health records data" or modality in {"linked_health_outcome", "derived_or_linked_outcome", "eligibility_flag"} or req_type in {"Self-reported medical history", "Prevalent disease covariates"}:
        return "Clinical outcomes and disease history"
    if req_type in {"Mental health / wellbeing", "Cognitive function"} or "cognitive" in text or "mental health" in text or "depression" in text or "anxiety" in text:
        return "Cognition and mental health"
    if area in {"Environmental data", "Demographic and lifestyle data"} and req_type not in {"Sociodemographics / assessment metadata", "Sociodemographics / psychosocial"}:
        return "Lifestyle and environmental exposures"
    if area == "Questionnaire data" and req_type in {"Diet / food preferences"}:
        return "Lifestyle and environmental exposures"
    if area == "Questionnaire data" and req_type == "Self-reported medical history":
        return "Clinical outcomes and disease history"
    if area == "Questionnaire data" and req_type in {"Mental health / wellbeing", "Cognitive function"}:
        return "Cognition and mental health"
    if req_type in {"Sociodemographics / assessment metadata", "Sociodemographics / psychosocial", "Family history"}:
        return "Demographics and assessment context" if req_type != "Family history" else "Clinical outcomes and disease history"
    if modality == "physical_measure":
        if any(k in concept.lower() for k in ["blood pressure", "pulse", "anthropometry", "body size"]) or text_has_physical_measure(text):
            return "Cardiometabolic physical measures"
        return "Cardiometabolic physical measures"
    if area == "Other retrieved data" and "z-score" in req_type.lower():
        return "Metabolomics and molecular biomarkers"
    return "Demographics and assessment context"


def acquisition_type_for_group(group: dict[str, object]) -> str:
    override = reviewed_tree_override(group)
    if override:
        return override[1]
    storage = str(group.get("storage_pattern", ""))
    function = str(group.get("semantic_function", ""))
    modality = str(group.get("modality_category", ""))
    area = str(group.get("requested_area", ""))
    req_type = str(group.get("requested_type", ""))
    text = group_lower_text(group)
    leaf_text = group_leaf_lower_text(group)
    domain = report_domain_for_group(group)

    if function == "ID" or storage == "alignment_metadata" or modality in {"alignment_id", "row_index"}:
        return "Identifier/join key"
    if storage in {"bulk_image_id", "bulk_file_id"}:
        return "Bulk image/file ID"
    if domain == "Genetics and genomics" and text_is_genetic_file_id(leaf_text):
        return "Bulk image/file ID"
    if function == "missing_note":
        return "Missing/failure note"
    if function == "device_or_qc":
        return "Device/QC metadata"
    if modality == "metabolomics_nmr":
        return "Molecular assay"
    if modality == "clinical_biomarker_blood_urine":
        return "Clinical lab measurement"
    if modality in {"brain_mri_imaging", "dxa_body_composition_imaging"}:
        return "Imaging-derived scalar"
    if modality == "eye_imaging":
        return "Device/QC metadata"
    if domain == "Genetics and genomics":
        if text_is_device_or_qc(leaf_text):
            return "Device/QC metadata"
        return "Curated local variable"
    if area == "Health records data":
        if "curated" in req_type.lower() or "derived" in req_type.lower() or "eligibility" in text or "follow-up" in req_type.lower():
            return "Curated local variable"
        return "Linked health record"
    if text_is_self_report_or_history(leaf_text):
        return "Questionnaire/self-report"
    if text_is_device_or_qc(leaf_text):
        return "Device/QC metadata"
    if area == "Questionnaire data" or req_type in {"Lifestyle / medication", "Diet / food preferences", "Self-reported medical history"}:
        return "Questionnaire/self-report"
    if area == "Demographic and lifestyle data" and req_type in {"Lifestyle / medication", "Family history", "Sociodemographics / psychosocial"}:
        return "Questionnaire/self-report"
    if area == "Environmental data":
        return "Curated local variable"
    if req_type in {"Derived clinical biomarkers", "Derived anthropometry/risk factors", "Unverified local z-score variables", "Local covariates and eligibility metadata"}:
        return "Curated local variable"
    if modality in {"physical_measure", "bioimpedance_body_composition", "eye_vision"}:
        return "Assessment measurement"
    if req_type in {"General physical measures", "Vision / ophthalmic measurements", "Bioimpedance body composition"}:
        return "Assessment measurement"
    if domain == "Eye and vision":
        return "Assessment measurement"
    if domain == "Demographics and assessment context" or req_type == "Sociodemographics / assessment metadata":
        return "Administrative/visit field"
    if area == "Data management":
        return "Administrative/visit field"
    return "Curated local variable"


def clean_report_concept(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return REPORT_CONCEPT_REPLACEMENTS.get(text, text)


def report_concept_for_group(group: dict[str, object]) -> str:
    override = reviewed_tree_override(group)
    if override:
        return override[2]
    domain = report_domain_for_group(group)
    acquisition = acquisition_type_for_group(group)
    display = clean_report_concept(group.get("display_concept", ""))
    concept = clean_report_concept(group.get("concept", ""))
    req_type = clean_report_concept(group.get("requested_type", ""))
    attribute = clean_report_concept(group.get("attribute", ""))
    text = group_lower_text(group)

    if domain == "Metabolomics and molecular biomarkers":
        if "z-score" in text and "nmr" not in text:
            return "Unverified local z-score variables"
        return "NMR metabolomics" if "nmr" in text or "metabolomic" in text else "Molecular biomarkers"
    if domain == "Clinical laboratory biomarkers":
        return display if display not in REPORT_GENERIC_CONCEPTS else req_type
    if domain == "Brain and neuroimaging":
        return display if display not in REPORT_GENERIC_CONCEPTS else "Brain MRI"
    if domain == "Eye and vision":
        if acquisition == "Questionnaire/self-report":
            if text_has_any(text, ["glasses", "contact lenses"]):
                return "Glasses and contact lens history"
            return "Eye disease and procedure history"
        if acquisition == "Device/QC metadata":
            return display if display not in REPORT_GENERIC_CONCEPTS else "Eye examination status and QC"
        return display if display not in REPORT_GENERIC_CONCEPTS else "Eye and vision phenotypes"
    if domain == "Body composition and bone":
        if acquisition == "Device/QC metadata":
            return display if display not in REPORT_GENERIC_CONCEPTS else "DXA assessment status"
        return display if display not in REPORT_GENERIC_CONCEPTS else "Body composition and bone traits"
    if domain == "Cardiometabolic physical measures":
        if acquisition == "Questionnaire/self-report":
            return "Cardiometabolic disease and medication history"
        if req_type == "Derived anthropometry/risk factors":
            return "Derived anthropometry and activity risk factors"
        return display if display not in REPORT_GENERIC_CONCEPTS else "Physical measures"
    if domain == "Clinical outcomes and disease history":
        if acquisition == "Linked health record" and "linked disease reports" in display.lower():
            return display.replace(" linked disease reports", "")
        if acquisition == "Questionnaire/self-report":
            if text_has_any(text, ["medication", "treatment/medication"]):
                return "Medication and treatment history"
            if text_has_any(text, ["high blood pressure", "diabetes", "angina", "heart attack", "stroke"]):
                return "Cardiometabolic disease history"
            return "Self-reported diagnoses and procedures"
        return display if display not in REPORT_GENERIC_CONCEPTS else (concept if concept not in REPORT_GENERIC_CONCEPTS else "Disease history and outcomes")
    if domain == "Lifestyle and environmental exposures":
        if "physical activity" in text or "met minutes" in text or "ipaq" in text:
            return "Physical activity"
        if display not in REPORT_GENERIC_CONCEPTS:
            return display
        return req_type if req_type not in REPORT_GENERIC_CONCEPTS else "Lifestyle and environmental exposures"
    if domain == "Cognition and mental health":
        if text_has_any(text, ["fi5 :", "fluid intelligence", "numeric memory", "trail making", "symbol digit", "pairs test"]):
            return "Cognitive task results"
        if "mental health" in text or "depression" in text or "anxiety" in text:
            return "Mental health"
        return display if display not in REPORT_GENERIC_CONCEPTS else "Cognitive task results"
    if domain == "Demographics and assessment context":
        if text_has_any(text, ["family relationship", "friendships satisfaction", "relationship satisfaction", "satisfaction"]):
            return "Psychosocial context"
        return display if display not in REPORT_GENERIC_CONCEPTS else "Participant characteristics and assessment visit"
    if domain == "Genetics and genomics":
        return display if display not in REPORT_GENERIC_CONCEPTS else "Genetic data files and sample metadata"
    if domain == "Data linkage and source administration":
        if acquisition == "Identifier/join key":
            return "ID namespaces and join keys"
        return display if display not in REPORT_GENERIC_CONCEPTS else attribute
    return display or concept or req_type or "Unclassified feature group"


def report_leaf_label_for_group(group: dict[str, object]) -> str:
    return str(group.get("attribute") or group.get("tree_leaf_label") or group.get("display_concept") or "").strip()


def legacy_tree_path_for_group(group: dict[str, object]) -> list[str]:
    source = str(group["source_label"])
    area = str(group["requested_area"])
    req_type = str(group["requested_type"])
    concept = str(group["display_concept"])
    modality = str(group["modality_category"])
    path = [source]
    if area == "Other retrieved data":
        path.append("Other fields requiring review")
    elif area == "Data management" and req_type == "Participant ID / alignment":
        path.append("ID namespaces and join keys")
    else:
        path.append(area)
    if modality == "linked_health_outcome":
        path.append("Linked disease reports (case-only date/source fields)")
        path.append(concept.replace(" linked disease reports", ""))
    elif modality == "metabolomics_nmr":
        path.append("Metabolomic biomarkers")
    elif modality == "alignment_id":
        pass
    elif req_type not in REPORT_GENERIC_CONCEPTS and req_type not in {concept, path[-1]}:
        path.append(req_type)
    if concept not in REPORT_GENERIC_CONCEPTS and concept not in path and modality not in {"linked_health_outcome", "metabolomics_nmr"}:
        path.append(concept)
    return compact_tree_path(path)


def compact_tree_path(path: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in path:
        item = str(item).strip()
        if not item:
            continue
        if cleaned and item.lower() == cleaned[-1].lower():
            continue
        if any(existing.lower() == item.lower() for existing in cleaned[-2:]):
            continue
        cleaned.append(item)
    return cleaned


def tree_path_for_group(group: dict[str, object]) -> list[str]:
    source = str(group["source_label"])
    domain = str(group.get("report_domain") or report_domain_for_group(group))
    acquisition = str(group.get("acquisition_type") or acquisition_type_for_group(group))
    concept = str(group.get("report_concept") or report_concept_for_group(group))
    path = [source, domain, acquisition]
    if concept and concept not in {domain, acquisition} and concept not in REPORT_GENERIC_CONCEPTS:
        path.append(concept)
    return compact_tree_path(path)


def tree_leaf_label_for_group(group: dict[str, object]) -> str:
    return report_leaf_label_for_group(group)


def visible_search_text_for_group(group: dict[str, object], rows: list[dict[str, str]]) -> str:
    raw_cols = [raw_column_id(r) for r in rows]
    field_ids = sorted({r["field_id"] for r in rows if r["field_id"]})
    visible_parts = [
        group.get("source_id", ""),
        group.get("source_label", ""),
        group.get("report_domain", ""),
        group.get("acquisition_type", ""),
        group.get("report_concept", ""),
        group.get("report_leaf_label", ""),
        group.get("tree_path", ""),
        group.get("concept", ""),
        group.get("observation", ""),
        group.get("attribute", ""),
        " ".join(field_ids),
        " ".join(raw_cols),
        " ".join(r.get("description", "") for r in rows),
        " ".join(r.get("ukb_showcase_url", "") for r in rows),
    ]
    return " ".join(str(part) for part in visible_parts if part).lower()


def assign_tree_review_flags(groups: list[dict[str, object]]) -> None:
    for group in groups:
        flags = []
        legacy_parts = str(group.get("legacy_tree_path", "")).split(" > ")
        legacy_top = legacy_parts[1] if len(legacy_parts) > 1 else ""
        requested_area = str(group.get("requested_area", ""))
        report_domain = str(group.get("report_domain", ""))
        acquisition = str(group.get("acquisition_type", ""))
        visible_search = str(group.get("visible_search_text", ""))
        semantic_function = str(group.get("semantic_function", ""))

        if reviewed_tree_override(group):
            flags.append("reviewed_override")
        if legacy_top in {"Questionnaire data", "Data management", "Other fields requiring review"}:
            flags.append("legacy_generic_top_replaced")
        if requested_area in {"Questionnaire data", "Data management", "Other retrieved data"}:
            flags.append("requested_category_not_tree_driver")
        if semantic_function == "missing_note":
            flags.append("missing_note_integrated_under_domain")
        if semantic_function == "device_or_qc":
            flags.append("device_qc_integrated_under_domain")
        if acquisition == "Questionnaire/self-report" and report_domain in {"Eye and vision", "Clinical outcomes and disease history", "Lifestyle and environmental exposures", "Cognition and mental health"}:
            flags.append("questionnaire_nested_by_biology")
        if "genetic" in visible_search and report_domain != "Genetics and genomics":
            flags.append("check_genetic_search_match")
        if "metadata" in visible_search and semantic_function not in {"metadata", "device_or_qc", "ID", "missing_note"} and report_domain != "Genetics and genomics":
            flags.append("check_metadata_visible_text")
        if report_domain not in REPORT_DOMAIN_ORDER:
            flags.append("unknown_report_domain")
        if acquisition not in ACQUISITION_TYPES:
            flags.append("unknown_acquisition_type")
        if str(group.get("report_concept", "")) in REPORT_GENERIC_CONCEPTS:
            flags.append("generic_report_concept")
        group["tree_review_flags"] = ";".join(flags)


def build_tree_review_candidates(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for group in groups:
        flags = str(group.get("tree_review_flags", ""))
        if not flags:
            continue
        rows.append(
            {
                "source_id": group.get("source_id", ""),
                "source_label": group.get("source_label", ""),
                "field_ids": group.get("field_ids", ""),
                "description": group.get("report_leaf_label") or group.get("tree_leaf_label") or group.get("attribute", ""),
                "semantic_function": group.get("semantic_function", ""),
                "semantic_function_label": group.get("semantic_function_label", ""),
                "value_type": group.get("value_type", ""),
                "current_path": group.get("legacy_tree_path", ""),
                "proposed_path": group.get("tree_path", ""),
                "report_domain": group.get("report_domain", ""),
                "acquisition_type": group.get("acquisition_type", ""),
                "report_concept": group.get("report_concept", ""),
                "flags": flags,
                "raw_columns_sample": group.get("raw_columns_sample", ""),
                "ukb_showcase_urls": group.get("ukb_showcase_urls", ""),
            }
        )
    return rows


def search_hits(groups: list[dict[str, object]], query: str) -> list[dict[str, object]]:
    q = query.lower()
    return [group for group in groups if q in str(group.get("visible_search_text", "")).lower()]


def build_tree_audit_summary(groups: list[dict[str, object]], review_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    top_nodes = Counter(str(group.get("tree_path", "")).split(" > ")[1] for group in groups if " > " in str(group.get("tree_path", "")))
    genetic_hits = search_hits(groups, "genetic")
    metadata_hits = search_hits(groups, "metadata")
    questionnaire_hits = search_hits(groups, "questionnaire")
    rows = [
        {"metric": "semantic_groups", "value": len(groups), "notes": "Total generated semantic feature groups."},
        {"metric": "review_candidate_groups", "value": len(review_rows), "notes": "Groups with deterministic review flags."},
        {"metric": "remaining_unclassified_groups", "value": sum(1 for g in groups if str(g.get("report_concept", "")) in REPORT_GENERIC_CONCEPTS), "notes": "Groups still using generic report concepts."},
        {"metric": "top_level_questionnaire_data_nodes", "value": top_nodes.get("Questionnaire data", 0), "notes": "Must be zero in biological-domain tree."},
        {"metric": "top_level_data_management_nodes", "value": top_nodes.get("Data management", 0), "notes": "Must be zero in biological-domain tree."},
        {"metric": "top_level_other_retrieved_data_nodes", "value": top_nodes.get("Other retrieved data", 0) + top_nodes.get("Other fields requiring review", 0), "notes": "Must be zero in biological-domain tree."},
        {"metric": "search_genetic_hits", "value": len(genetic_hits), "notes": "Default app search over visible semantics."},
        {"metric": "search_genetic_non_genetics_domain_hits", "value": sum(1 for g in genetic_hits if g.get("report_domain") != "Genetics and genomics"), "notes": "Must be zero."},
        {"metric": "search_metadata_hits", "value": len(metadata_hits), "notes": "Hits contain metadata in visible report text or UKB description."},
        {"metric": "search_questionnaire_hits", "value": len(questionnaire_hits), "notes": "Hits contain questionnaire in visible report text or UKB description."},
        {"metric": "search_questionnaire_non_self_report_hits", "value": sum(1 for g in questionnaire_hits if g.get("acquisition_type") != "Questionnaire/self-report" and "questionnaire" not in str(g.get("report_concept", "")).lower() and "questionnaire" not in str(g.get("report_leaf_label", "")).lower()), "notes": "Should only occur when UKB visible descriptions contain questionnaire."},
    ]
    for domain, count in Counter(str(group.get("report_domain", "")) for group in groups).most_common():
        rows.append({"metric": f"report_domain:{domain}", "value": count, "notes": "Semantic group count by biological domain."})
    return rows


def coverage_explanation_for_group(group: dict[str, object]) -> str:
    storage = str(group["storage_pattern"])
    modality = str(group["modality_category"])
    if modality == "linked_health_outcome":
        return "Exact participant records with a non-empty linked disease report field; these fields are case-only by design."
    if storage in {"repeated_visit", "repeated_visit_array"}:
        return "Exact participant records with any non-empty value in the group; visit coverage is reported separately by UKB instance."
    if storage in {"repeated_array", "paired_repeated_array"}:
        return "Exact participant records with any non-empty repeated array slot in the group."
    if modality == "metabolomics_nmr":
        return "Exact participant records with any non-empty NMR metabolomics value in the grouped panel."
    return "Exact participant records with any non-empty raw column in this semantic group."


HOSPITAL_SUMMARY_EVENT_FIELDS = {
    "41200",
    "41201",
    "41202",
    "41203",
    "41204",
    "41205",
    "41210",
    "41256",
    "41257",
    "41258",
    "41260",
    "41262",
    "41263",
    "41270",
    "41271",
    "41272",
    "41273",
    "41280",
    "41281",
    "41282",
    "41283",
}

SELF_REPORTED_EVENT_FIELDS = {
    "84",
    "87",
    "92",
    "20001",
    "20002",
    "20003",
    "20004",
    "20006",
    "20007",
    "20008",
    "20009",
    "20010",
    "20011",
    "20012",
    "20013",
    "20014",
}

BLOOD_PRESSURE_READING_FIELDS = {"93", "94", "95", "102", "4079", "4080", "4081"}
BLOOD_PRESSURE_SCREEN_FIELDS = {"96"}
VISUAL_ACUITY_FIELDS = {
    "5074",
    "5075",
    "5076",
    "5077",
    "5078",
    "5079",
    "5080",
    "5081",
    "5082",
    "5083",
}
GENETIC_RELATEDNESS_FIELDS = {"22011", "22012", "22013"}
WGS_BULK_FILE_FIELDS = {"23181", "23182", "23183", "23184"}
FAMILY_HISTORY_ILLNESS_FIELDS = {"20107", "20110", "20111"}
HOSPITAL_ADMIN_REPEATED_FIELDS = {"41206", "41207", "41208", "41209", "41211", "41233"}
LOCATION_HISTORY_FIELDS = {"22700"}


def repeated_entry_explanation_for_group(rows: list[dict[str, str]], storage: str, arrays: list[str]) -> str:
    entry_count = len(arrays) if arrays else 1
    if entry_count <= 1:
        return ""

    field_ids = {r["field_id"] for r in rows if r["field_id"]}
    desc = " | ".join(sorted({r["description"] for r in rows if r["description"]}))
    desc_l = desc.lower()
    type_l = " ".join(r.get("ukb_type", "") for r in rows).lower()
    modality = rows[0].get("modality_category", "")

    pair_ids = {PAIR_FIELDS[field_id]["pair_id"] for field_id in field_ids if field_id in PAIR_FIELDS}
    if any("diagnosis" in pair_id for pair_id in pair_ids):
        return "Each position is one hospital diagnosis event. Code/date fields with the same entry number describe the same event, e.g. 41270-0.5 and 41280-0.5."
    if any("operation" in pair_id or "opcs" in pair_id for pair_id in pair_ids):
        return "Each position is one hospital procedure event. Procedure/date fields with the same entry number describe the same event."
    if field_ids & HOSPITAL_SUMMARY_EVENT_FIELDS:
        return "Each position is one hospital diagnosis, external-cause, or procedure entry; related fields use the same entry number."
    if field_ids & HOSPITAL_ADMIN_REPEATED_FIELDS:
        return "Each position is one hospital admission/episode attribute; related hospital fields use the same entry number."
    if field_ids & SELF_REPORTED_EVENT_FIELDS or any(
        token in desc_l
        for token in [
            "cancer code",
            "illness code",
            "operation code",
            "year/age first occurred",
            "interpolated year",
            "interpolated age",
            "method of recording time",
            "treatment/medication code",
        ]
    ):
        return "Each position is one self-reported cancer, illness, operation, or treatment item; related code/year/age fields use the same order."
    if "cause of death" in desc_l or "causes of death" in desc_l:
        return "Each position is one coded cause on a death record; contributory causes allow multiple secondary ICD10 causes."
    if modality == "eye_imaging" or storage in {"bulk_image_id", "bulk_file_id"} and any(
        token in desc_l for token in ["image", "data file", "oct", "fundus", "fda", "fds"]
    ):
        return "Each position is one bulk image/file handle for the same eye, modality, and visit. Values are file identifiers, not image pixels."
    if field_ids & WGS_BULK_FILE_FIELDS:
        return "Each position is one whole-genome sequencing file or index file handle."
    if field_ids & BLOOD_PRESSURE_READING_FIELDS or (
        "blood pressure" in desc_l
        and any(token in desc_l for token in ["automated reading", "manual reading", "method of measuring", "during blood-pressure measurement"])
    ):
        return "Each position is one repeated blood-pressure or pulse reading from the same measurement sequence."
    if field_ids & BLOOD_PRESSURE_SCREEN_FIELDS:
        return "Each position is one blood-pressure screen timestamp during the assessment."
    if field_ids & VISUAL_ACUITY_FIELDS or any(
        token in desc_l for token in ["visual acuity", "logmar", "displayed letters", "letters shown", "letters correct"]
    ):
        return "Each position is one visual-acuity round or letter-level result; displayed-letter/result fields use the same order."
    if any(
        token in desc_l
        for token in [
            "spherical power",
            "cylindrical power",
            "astigmatism",
            "weak meridian",
            "strong meridian",
            "asymmetry index",
            "asymmetry angle",
            "regularity index",
            "keratometry",
            "refractometry",
        ]
    ):
        return "Each position is one refractometry/keratometry reading for the same visit, eye, and measurement zone."
    if "genetic principal components" in desc_l or field_ids == {"22009"}:
        return "Each position is one genetic principal component, e.g. PC1 to PC40."
    if field_ids & GENETIC_RELATEDNESS_FIELDS:
        return "Each position is one inferred relatedness-pair slot; pairing/factor/IBS0 fields use the same entry number."
    if field_ids & FAMILY_HISTORY_ILLNESS_FIELDS:
        return "Each position is one selected illness code for the relative."
    if field_ids & LOCATION_HISTORY_FIELDS:
        return "Each position is one recorded location-history entry."
    if any(
        token in desc_l
        for token in [
            "round",
            "puzzle",
            "snap-button",
            "numeric path",
            "alphanumeric path",
            "values wanted",
            "values entered",
            "value entered",
            "number of digits",
            "digits entered",
            "digits entered correctly",
            "target number",
            "number entered",
            "time elapsed",
            "time first key",
            "time last key",
            "time number displayed",
            "duration to entering value",
            "keystroke",
            "item selected",
            "screen layout",
            "test array",
            "number of columns displayed",
            "number of rows displayed",
            "pattern of lights",
            "lights test",
            "words answer",
        ]
    ):
        return "Each position is one task round, trial, puzzle, path step, or item; related task fields use the same order."
    if "categorical (multiple)" in type_l:
        return "Each position is one selected answer/code for a question. Several positions allow a participant to give more than one response at the same visit."
    if any(token in desc_l for token in ["device", "method", "unreliable", "error", "qc"]):
        return "Each position is device, method, reliability, or QC metadata for the corresponding repeated measurement/task entry."
    return "Use the UKB reference for the meaning and ordering of these repeated entries."


def build_semantic_groups(
    columns: list[dict[str, str]],
    inventory: list[dict[str, str]],
    modality_counts: list[dict[str, str]],
    global_participants: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_rows = {r["source_id"]: to_int(r["row_count"]) for r in inventory}
    modality_count_map = {
        (r["source_id"], r["modality_category"]): to_int(r["participant_count"])
        for r in modality_counts
    }

    slots_by_source_field: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in columns:
        if row["field_id"] in PAIR_FIELDS:
            slots_by_source_field[(row["source_id"], row["field_id"])].add((row["instance"], row["array"]))

    paired_ready: set[tuple[str, str]] = set()
    for (source_id, field_id), slots in slots_by_source_field.items():
        partner = FIELD_PARTNER.get(field_id)
        if not partner:
            continue
        partner_slots = slots_by_source_field.get((source_id, partner), set())
        if slots & partner_slots:
            paired_ready.add((source_id, field_id))

    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in columns:
        row = dict(row)
        row["source_kind"] = normalize_source_kind(row["source_kind"])
        row["_paired_ready"] = "true" if (row["source_id"], row["field_id"]) in paired_ready else "false"
        buckets[group_key(row)].append(row)

    groups: list[dict[str, object]] = []
    raw_to_group: dict[str, str] = {}
    for key, rows in buckets.items():
        rows = sorted(rows, key=lambda r: to_int(r.get("column_index", 0)))
        first = rows[0]
        source_id = first["source_id"]
        concept, observation, attribute = concept_observation_attribute(first)
        domain = scientific_domain(first)
        requested_area, requested_type = requested_area_type(first)
        modality = first["modality_category"]
        storage = infer_storage_pattern(rows)
        semantic_function = infer_semantic_function(rows, storage)
        value_type = infer_value_type(rows, semantic_function, storage)
        role = infer_prediction_role(rows)
        warning = leakage_warning(rows, role)
        raw_cols = [raw_column_id(r) for r in rows]
        field_ids = sorted({r["field_id"] for r in rows if r["field_id"]}, key=lambda x: (len(x), x))
        instances = sorted({r["instance"] for r in rows if r["instance"] != ""}, key=lambda x: to_int(x))
        arrays = sorted({r["array"] for r in rows if r["array"] != ""}, key=lambda x: to_int(x))
        column_annotations = raw_column_annotations(rows, storage)
        paired_column_count = paired_column_count_for_group(rows)
        paired_column_explanation = paired_column_explanation_for_group(rows, paired_column_count)

        if modality == "metabolomics_nmr":
            concept = "NMR metabolomics"
            observation = "Z-scored NMR biomarker panel"
            attribute = f"NMR metabolomics panel ({len(raw_cols)} z-scored features)"
            requested_area, requested_type = "Biomarker data", "Metabolomic biomarkers"
        elif modality == "alignment_id":
            concept = "ID namespaces and join keys"
            observation = "Encoded participant identifier columns"
            if source_id == "metabolite_csv":
                attribute = "METABOLITE ID columns: eid_ckd aligns locally; eid_ageing is an alternative namespace"
            else:
                attribute = "Local encoded participant ID for this source"
            requested_area, requested_type = "Data management", "Participant ID / alignment"
        elif modality == "linked_health_outcome":
            endpoint = linked_endpoint_name(first["description"] or first["column_name"])
            family = disease_family(endpoint)
            concept = "Linked disease reports"
            observation = f"{family} linked report"
            attribute = f"{endpoint} report date/source fields"
            requested_area, requested_type = "Health records data", "Linked disease reports"
        elif is_metabolite_unnamed_v_zscore(first):
            concept = "Unnamed local derived z-score variables"
            observation = "Local METABOLITE derived z-score block"
            attribute = f"Unnamed METABOLITE local z-score variables ({len(raw_cols)} columns; {field_ids[0]}-{field_ids[-1]})"
            requested_area, requested_type = "Other retrieved data", "Unverified local z-score variables"
        elif is_metabolite_local_derived(first):
            requested_area, requested_type, concept = metabolite_local_category(first)
            observation = "Local derived METABOLITE attribute"
            if requested_type in {"Transformed date/time fields", "Transformed UKB date fields"}:
                attribute = f"Timestamp-transformed UKB fields ({len(raw_cols)} columns)"
            elif requested_type in {"METABOLITE local covariates and eligibility", "Local covariates and eligibility metadata"}:
                attribute = f"METABOLITE local covariates and eligibility ({len(raw_cols)} columns)"
            elif requested_type in {"Follow-up and temporality variables", "Follow-up / censoring time"} and len(raw_cols) > 1:
                attribute = f"{concept} ({len(raw_cols)} columns)"
            else:
                attribute = humanize_metabolite_local_name(first)

        lab_concept = display_concept(first, concept, observation, attribute)
        data_type_counts = Counter(r["ukb_type"] or "Unknown" for r in rows)
        max_non_missing = max(to_int(r["non_missing_count"]) for r in rows)
        source_n = source_rows.get(source_id, 0)
        modality_n = modality_count_map.get((source_id, modality), max_non_missing)
        group_id = (
            f'{source_id}__{slug(domain)}__{slug(concept)}__'
            f'{slug(observation)}__{slug(attribute)}__{slug("_".join(field_ids) or first["column_name"])}'
        )

        for raw in raw_cols:
            raw_to_group[raw] = group_id

        group = {
            "group_id": group_id,
            "source_id": source_id,
            "source_label": SOURCE_LABELS.get(source_id, source_id),
            "modality_category": modality,
            "modality_label": MODALITY_LABELS.get(modality, modality),
            "requested_area": requested_area,
            "requested_type": requested_type,
            "scientific_domain": domain,
            "concept": concept,
            "observation": observation,
            "display_concept": lab_concept,
            "attribute": attribute,
            "human_label": attribute if attribute != observation else observation,
            "storage_pattern": storage,
            "column_layout": storage_pattern_label(storage),
            "column_layout_detail": storage_pattern_detail(storage),
            "semantic_function": semantic_function,
            "semantic_function_label": SEMANTIC_FUNCTION_LABELS.get(semantic_function, semantic_function),
            "value_type": value_type,
            "value_representation": value_representation(rows),
            "prediction_role": role,
            "prediction_role_label": ROLE_LABELS.get(role, role),
            "leakage_or_temporality_warning": warning,
            "feature_count": len(raw_cols),
            "participant_count_proxy": max_non_missing,
            "coverage_count_type": "max_non_missing_column_in_group",
            "source_participant_count": source_n,
            "source_percent_proxy": pct(max_non_missing, source_n),
            "global_percent_proxy": pct(max_non_missing, global_participants),
            "modality_participant_count": modality_n,
            "modality_source_percent": pct(modality_n, source_n),
            "field_ids": ";".join(field_ids),
            "instances": ";".join(instances),
            "arrays": ";".join(arrays),
            "visit_instance_count": max(1, len(instances)),
            "repeated_entry_count": max(1, len(arrays)),
            "paired_column_count": paired_column_count,
            "paired_column_explanation": paired_column_explanation,
            "repeated_entry_explanation": repeated_entry_explanation_for_group(rows, storage, arrays),
            "local_derivation_explanation": metabolite_local_derivation_explanation(rows),
            "data_type_mix": "; ".join(f"{k}: {v}" for k, v in sorted(data_type_counts.items())),
            "raw_columns": ";".join(raw_cols),
            "raw_columns_sample": "; ".join(raw_cols[:12]) + ("; ..." if len(raw_cols) > 12 else ""),
            "raw_column_annotations": column_annotations,
            "raw_column_annotations_json": json.dumps(column_annotations, ensure_ascii=False),
            "ukb_showcase_urls": ";".join(sorted({r["ukb_showcase_url"] for r in rows if r["ukb_showcase_url"]})),
        }
        group["legacy_tree_path"] = " > ".join(legacy_tree_path_for_group(group))
        group["report_domain"] = report_domain_for_group(group)
        group["acquisition_type"] = acquisition_type_for_group(group)
        group["report_concept"] = report_concept_for_group(group)
        group["report_leaf_label"] = report_leaf_label_for_group(group)
        group["tree_path"] = " > ".join(tree_path_for_group(group))
        group["tree_leaf_label"] = tree_leaf_label_for_group(group)
        group["coverage_explanation"] = coverage_explanation_for_group(group)
        group["case_only_field"] = modality == "linked_health_outcome"
        group["tree_review_flags"] = ""
        group["visible_search_text"] = visible_search_text_for_group(group, rows)
        group["search_text"] = group["visible_search_text"]
        groups.append(group)

    groups.sort(
        key=lambda g: (
            str(g["source_label"]),
            str(g["report_domain"]),
            str(g["acquisition_type"]),
            str(g["tree_path"]),
            str(g["report_concept"]),
            str(g["tree_leaf_label"]),
        )
    )
    assign_tree_review_flags(groups)

    tree: dict[str, object] = {
        "name": "All retrieved UKB data",
        "children": [],
        "feature_count": sum(to_int(g["feature_count"]) for g in groups),
        "semantic_group_count": len(groups),
    }
    node_index: dict[tuple[str, ...], dict[str, object]] = {}

    def get_node(parent: dict[str, object], path: tuple[str, ...], name: str, kind: str) -> dict[str, object]:
        key = path + (name,)
        if key in node_index:
            return node_index[key]
        node = {
            "name": name,
            "kind": kind,
            "children": [],
            "feature_count": 0,
            "semantic_group_count": 0,
            "participant_count_proxy": 0,
        }
        parent.setdefault("children", []).append(node)
        node_index[key] = node
        return node

    for group in groups:
        path = ()
        parent = tree
        current = tree
        components = [p for p in str(group["tree_path"]).split(" > ") if p]
        for idx, name in enumerate(components):
            kind = "dataset" if idx == 0 else "semantic_layer"
            current = get_node(parent, path, name, kind)
            path += (name,)
            parent = current
        current.setdefault("semantic_groups", []).append(
            {
                "group_id": group["group_id"],
                "label": group["tree_leaf_label"],
                "feature_count": group["feature_count"],
                "participant_count_proxy": group["participant_count_proxy"],
                "storage_pattern": group["storage_pattern"],
                "column_layout": group["column_layout"],
                "semantic_function": group["semantic_function"],
                "value_type": group["value_type"],
                "prediction_role": group["prediction_role"],
                "warning": group["leakage_or_temporality_warning"],
            }
        )

    def rollup(node: dict[str, object]) -> tuple[int, int, int]:
        children = node.get("children") or []
        own_groups = node.get("semantic_groups") or []
        features = sum(to_int(g.get("feature_count", 0)) for g in own_groups)
        groups_n = len(own_groups)
        participants = max([to_int(g.get("participant_count_proxy", 0)) for g in own_groups] or [0])
        for child in children:
            c_features, c_groups, c_participants = rollup(child)
            features += c_features
            groups_n += c_groups
            participants = max(participants, c_participants)
        node["feature_count"] = features
        node["semantic_group_count"] = groups_n
        node["participant_count_proxy"] = participants
        return features, groups_n, participants

    rollup(tree)
    return groups, {"tree": tree, "raw_to_group": raw_to_group}


def semantic_schema_hash(groups: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for group in sorted(groups, key=lambda g: str(g["group_id"])):
        digest.update(
            "\t".join(
                [
                    str(group["group_id"]),
                    str(group["source_id"]),
                    str(group["tree_path"]),
                    str(group["raw_columns"]),
                    str(group["semantic_function"]),
                    str(group["storage_pattern"]),
                    str(group["value_type"]),
                ]
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def profile_fingerprint(groups: list[dict[str, object]]) -> dict[str, object]:
    return {
        "profile_version": PROFILE_VERSION,
        "schema_hash": semantic_schema_hash(groups),
        "source_files": source_file_stats(),
        "sample_seed": SAMPLE_SEED,
        "sample_rows_per_source": SAMPLE_ROWS_PER_SOURCE,
        "missing_value_tokens": sorted(MISSING_VALUE_TOKENS),
    }


def bitset_from_bool(values: np.ndarray, offset: int) -> int:
    arr = np.asarray(values, dtype=np.uint8)
    if arr.size == 0 or not bool(arr.any()):
        return 0
    packed = np.packbits(arr, bitorder="little").tobytes()
    return int.from_bytes(packed, "little") << offset


def source_column_name(raw: str, source_id: str) -> str:
    prefix = f"{source_id}:"
    return raw[len(prefix) :] if raw.startswith(prefix) else raw


def source_qualified(source_id: str, column: str) -> str:
    return f"{source_id}:{column}"


def group_column_names(group: dict[str, object]) -> list[str]:
    return [
        source_column_name(raw, str(group["source_id"]))
        for raw in str(group.get("raw_columns", "")).split(";")
        if raw
    ]


def should_sample_examples(group: dict[str, object]) -> bool:
    return (
        str(group.get("semantic_function")) != "ID"
        and str(group.get("storage_pattern")) not in {"bulk_image_id", "bulk_file_id", "alignment_metadata"}
    )


def example_column_names(
    source_id: str,
    base_cols: list[str],
    column_lookup: dict[tuple[str, str], dict[str, str]],
    partner_lookup: dict[tuple[str, str, str, str], str],
) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    for col in base_cols:
        if col not in seen:
            cols.append(col)
            seen.add(col)
        row = column_lookup.get((source_id, col))
        if not row:
            continue
        partner = FIELD_PARTNER.get(row.get("field_id", ""))
        if not partner:
            continue
        partner_col = partner_lookup.get((source_id, partner, row.get("instance", ""), row.get("array", "")))
        if partner_col and partner_col not in seen:
            cols.append(partner_col)
            seen.add(partner_col)
    return cols


def add_example(
    store: list[dict[str, object]],
    source_id: str,
    row: pd.Series,
    columns: list[str],
    limit: int = 12,
) -> None:
    if len(store) >= 2:
        return
    values = []
    seen_cols = set()
    for col in columns:
        if col in seen_cols or col not in row.index:
            continue
        seen_cols.add(col)
        value = row[col]
        if pd.isna(value) or is_missing_value(value):
            continue
        values.append({"column": source_qualified(source_id, col), "value": html_truncate(value)})
        if len(values) >= limit:
            break
    if values:
        store.append({"values": values})


def random_candidate_rows(source_id: str, row_count: int) -> list[int]:
    if row_count <= 0:
        return []
    n = min(SAMPLE_ROWS_PER_SOURCE, row_count)
    rng = random.Random(SAMPLE_SEED + stable_seed(source_id))
    return sorted(rng.sample(range(row_count), n))


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="latin1", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def reduce_nonempty_by_key(
    non_empty: pd.DataFrame,
    columns: list[str],
    column_to_key: dict[str, object],
) -> tuple[list[object], np.ndarray | None]:
    active = [c for c in columns if c in non_empty.columns and c in column_to_key]
    if not active:
        return [], None
    active.sort(key=lambda c: str(column_to_key[c]))
    keys = [column_to_key[c] for c in active]
    boundaries = [0]
    for idx in range(1, len(keys)):
        if keys[idx] != keys[idx - 1]:
            boundaries.append(idx)
    arr = non_empty[active].to_numpy(dtype=bool, copy=False)
    reduced = np.maximum.reduceat(arr, np.array(boundaries), axis=1)
    return [keys[idx] for idx in boundaries], reduced


def make_nonempty_reducer(columns: list[str], column_to_key: dict[str, object]) -> dict[str, object]:
    active = [c for c in columns if c in column_to_key]
    active.sort(key=lambda c: str(column_to_key[c]))
    keys = [column_to_key[c] for c in active]
    boundaries = [0]
    for idx in range(1, len(keys)):
        if keys[idx] != keys[idx - 1]:
            boundaries.append(idx)
    return {
        "columns": active,
        "column_keys": keys,
        "boundaries": np.array(boundaries),
        "keys": [keys[idx] for idx in boundaries],
    }


def apply_nonempty_reducer(non_empty: pd.DataFrame, reducer: dict[str, object]) -> tuple[list[object], np.ndarray | None]:
    cols = [c for c in reducer["columns"] if c in non_empty.columns]
    if not cols:
        return [], None
    if len(cols) != len(reducer["columns"]):
        column_to_key = {
            c: k
            for c, k in zip(reducer["columns"], reducer["column_keys"])
            if c in non_empty.columns
        }
        return reduce_nonempty_by_key(non_empty, cols, column_to_key)
    arr = non_empty[reducer["columns"]].to_numpy(dtype=bool, copy=False)
    reduced = np.maximum.reduceat(arr, reducer["boundaries"], axis=1)
    return reducer["keys"], reduced


def build_column_lookup(columns: list[dict[str, str]]):
    by_source_col = {}
    partner_lookup = {}
    for row in columns:
        source_id = row["source_id"]
        col = row["column_name"]
        by_source_col[(source_id, col)] = row
        if row["field_id"] in FIELD_PARTNER:
            partner_lookup[(source_id, row["field_id"], row["instance"], row["array"])] = col
    return by_source_col, partner_lookup


def scan_source_profile(
    source_id: str,
    path: Path,
    row_count: int,
    groups: list[dict[str, object]],
    column_lookup: dict[tuple[str, str], dict[str, str]],
    partner_lookup: dict[tuple[str, str, str, str], str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    header = set(read_header(path))
    source_groups = [g for g in groups if g["source_id"] == source_id]
    group_cols = {str(g["group_id"]): [c for c in group_column_names(g) if c in header] for g in source_groups}
    group_inst_cols: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    group_example_cols: dict[str, list[str]] = {}
    needed_cols: set[str] = set()

    for group in source_groups:
        gid = str(group["group_id"])
        cols = group_cols[gid]
        needed_cols.update(cols)
        for col in cols:
            row = column_lookup.get((source_id, col), {})
            inst = row.get("instance", "")
            if inst != "":
                group_inst_cols[gid][inst].append(col)
        group_example_cols[gid] = [c for c in example_column_names(source_id, cols, column_lookup, partner_lookup) if c in header]
        needed_cols.update(group_example_cols[gid])

    usecols = sorted(needed_cols)
    group_bits: dict[str, int] = defaultdict(int)
    group_instance_bits: dict[tuple[str, str], int] = defaultdict(int)
    random_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    fallback_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    sample_allowed = {str(g["group_id"]): should_sample_examples(g) for g in source_groups}
    coverage_cols = sorted({col for cols in group_cols.values() for col in cols})
    column_to_gid = {}
    column_to_inst_key = {}
    inst_coverage_cols = []
    for gid, cols in group_cols.items():
        for col in cols:
            column_to_gid[col] = gid
    for gid, by_inst in group_inst_cols.items():
        for inst, cols in by_inst.items():
            for col in cols:
                column_to_inst_key[col] = (gid, inst)
                inst_coverage_cols.append(col)
    inst_coverage_cols = sorted(set(inst_coverage_cols))
    group_reducer = make_nonempty_reducer(coverage_cols, column_to_gid)
    instance_reducer = make_nonempty_reducer(inst_coverage_cols, column_to_inst_key)

    candidates = random_candidate_rows(source_id, row_count)
    candidate_i = 0
    row_offset = 0

    if not usecols:
        return {}, []

    reader = pd.read_csv(
        path,
        dtype=str,
        na_filter=False,
        encoding="latin1",
        chunksize=PROFILE_CHUNK_SIZE,
        usecols=usecols,
        low_memory=False,
    )
    for chunk in reader:
        chunk_len = len(chunk)
        chunk_end = row_offset + chunk_len
        non_empty = non_missing_frame(chunk)
        candidate_locals = []
        while candidate_i < len(candidates) and candidates[candidate_i] < row_offset:
            candidate_i += 1
        j = candidate_i
        while j < len(candidates) and candidates[j] < chunk_end:
            candidate_locals.append(candidates[j] - row_offset)
            j += 1
        sample_chunk = chunk.iloc[candidate_locals] if candidate_locals else None

        group_keys, group_matrix = apply_nonempty_reducer(non_empty, group_reducer)
        inst_keys, inst_matrix = apply_nonempty_reducer(non_empty, instance_reducer)

        if group_matrix is not None:
            for idx, gid_obj in enumerate(group_keys):
                gid = str(gid_obj)
                present = group_matrix[:, idx]
                bits = bitset_from_bool(present, row_offset)
                if bits:
                    group_bits[gid] |= bits

                if not sample_allowed.get(gid):
                    continue

                cols = group_cols.get(gid) or []
                example_cols = group_example_cols.get(gid) or cols
                if sample_chunk is not None and len(random_examples[gid]) < 2:
                    sample_present = present[candidate_locals]
                    for sample_pos in np.flatnonzero(sample_present):
                        add_example(random_examples[gid], source_id, sample_chunk.iloc[int(sample_pos)], example_cols)
                        if len(random_examples[gid]) >= 2:
                            break

                if len(fallback_examples[gid]) < 2 and bool(present.any()):
                    for local_pos in np.flatnonzero(present)[: 2 - len(fallback_examples[gid])]:
                        add_example(fallback_examples[gid], source_id, chunk.iloc[int(local_pos)], example_cols)

        if inst_matrix is not None:
            for idx, key in enumerate(inst_keys):
                gid, inst = key
                inst_present = inst_matrix[:, idx]
                inst_bits = bitset_from_bool(inst_present, row_offset)
                if inst_bits:
                    group_instance_bits[(str(gid), str(inst))] |= inst_bits

        row_offset += chunk_len

    profiles = {}
    group_coverage_rows = []
    for group in source_groups:
        gid = str(group["group_id"])
        count = group_bits.get(gid, 0).bit_count()
        visit_rows = []
        for (inst_gid, inst), bits in sorted(group_instance_bits.items(), key=lambda item: (item[0][0], to_int(item[0][1]))):
            if inst_gid != gid:
                continue
            visit_rows.append(
                {
                    "source_id": source_id,
                    "group_id": gid,
                    "instance": inst,
                    "participant_count": bits.bit_count(),
                    "source_participant_count": row_count,
                    "source_percent": pct(bits.bit_count(), row_count),
                }
            )
        all_row = {
            "source_id": source_id,
            "group_id": gid,
            "instance": "all",
            "participant_count": count,
            "source_participant_count": row_count,
            "source_percent": pct(count, row_count),
        }
        group_coverage_rows.append(all_row)
        group_coverage_rows.extend(visit_rows)

        examples = random_examples.get(gid, [])[:2]
        if len(examples) < 2:
            for fallback in fallback_examples.get(gid, []):
                if len(examples) >= 2:
                    break
                examples.append(fallback)

        if not sample_allowed.get(gid):
            example_status = "Examples hidden for ID/file fields"
            examples = []
        elif examples:
            example_status = "Sampled real non-empty entries"
        else:
            example_status = "No non-empty values found during profiling"

        profiles[gid] = {
            "participant_count": count,
            "source_participant_count": row_count,
            "source_percent": pct(count, row_count),
            "visit_coverage": visit_rows,
            "examples": examples[:2],
            "example_status": example_status,
            "bitset": group_bits.get(gid, 0),
            "instance_bitsets": {inst: bits for (inst_gid, inst), bits in group_instance_bits.items() if inst_gid == gid},
        }

    return profiles, group_coverage_rows


def node_paths_for_group(group: dict[str, object]) -> list[str]:
    parts = [p for p in str(group.get("tree_path", "")).split(" > ") if p]
    return [" > ".join(parts[:i]) for i in range(1, len(parts) + 1)]


def build_node_coverage(
    groups: list[dict[str, object]],
    group_profiles: dict[str, dict[str, object]],
    source_rows: dict[str, int],
    scope: str,
) -> list[dict[str, object]]:
    node_bits: dict[tuple[str, str, str], int] = defaultdict(int)
    node_fallback_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    node_group_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    node_features: dict[tuple[str, str], int] = defaultdict(int)

    for group in groups:
        if scope == "hide_missing_note" and group.get("semantic_function") == "missing_note":
            continue
        gid = str(group["group_id"])
        source_id = str(group["source_id"])
        profile = group_profiles.get(gid, {})
        group_bits = profile.get("bitset") if isinstance(profile.get("bitset"), int) else None
        group_count = to_int(profile.get("participant_count", group.get("participant_count_proxy", 0)))
        instance_bits = profile.get("instance_bitsets", {}) if isinstance(profile.get("instance_bitsets"), dict) else {}
        visit_coverage = profile.get("visit_coverage", []) if isinstance(profile.get("visit_coverage", []), list) else []
        for path in node_paths_for_group(group):
            node_key = (source_id, path)
            node_group_ids[node_key].add(gid)
            node_features[node_key] += to_int(group.get("feature_count", 0))
            coverage_key = (source_id, path, "all")
            if group_bits is not None:
                node_bits[coverage_key] |= group_bits
            else:
                node_fallback_counts[coverage_key] = max(node_fallback_counts[coverage_key], group_count)
            if instance_bits:
                for inst, bits in instance_bits.items():
                    if isinstance(bits, int):
                        node_bits[(source_id, path, str(inst))] |= bits
            else:
                for visit_row in visit_coverage:
                    inst = str(visit_row.get("instance", ""))
                    if not inst or inst == "all":
                        continue
                    visit_key = (source_id, path, inst)
                    node_fallback_counts[visit_key] = max(
                        node_fallback_counts[visit_key],
                        to_int(visit_row.get("participant_count", 0)),
                    )

    rows = []
    coverage_keys = set(node_bits) | set(node_fallback_counts)
    for source_id, path, inst in sorted(coverage_keys, key=lambda item: (item[0], item[1], item[2])):
        node_key = (source_id, path)
        bit_count = node_bits.get((source_id, path, inst), 0).bit_count()
        count = max(bit_count, node_fallback_counts.get((source_id, path, inst), 0))
        source_n = source_rows.get(source_id, 0)
        rows.append(
            {
                "coverage_scope": scope,
                "source_id": source_id,
                "node_id": f"{source_id}__{slug(path)}",
                "tree_path": path,
                "instance": inst,
                "participant_count": count,
                "source_participant_count": source_n,
                "source_percent": pct(count, source_n),
                "group_count": len(node_group_ids[node_key]),
                "feature_count": node_features[node_key],
            }
        )
    return rows


def source_ids_by_group_from_coverage(rows: list[dict[str, object]]) -> dict[str, str]:
    by_group = {}
    for row in rows:
        gid = str(row.get("group_id", ""))
        source_id = str(row.get("source_id", ""))
        if gid and source_id:
            by_group[gid] = source_id
    return by_group


def raw_column_signature(raw_columns: object) -> str:
    return ";".join(sorted(c for c in str(raw_columns or "").split(";") if c))


def cached_group_signatures() -> dict[tuple[str, str], str]:
    path = META_DIR / "semantic_feature_groups.csv"
    if not path.exists():
        return {}
    signatures: dict[tuple[str, str], str] = {}
    for row in read_csv(path):
        gid = row.get("group_id", "")
        source_id = row.get("source_id", "")
        signature = raw_column_signature(row.get("raw_columns", ""))
        if gid and source_id and signature:
            signatures[(source_id, signature)] = gid
    return signatures


def clone_cached_profile(profile: dict[str, object], source_id: str, group_id: str) -> dict[str, object]:
    cloned = json.loads(json.dumps(profile, ensure_ascii=False))
    for row in cloned.get("visit_coverage", []) or []:
        row["source_id"] = source_id
        row["group_id"] = group_id
    return cloned


def coverage_rows_from_profiles(
    groups: list[dict[str, object]],
    profiles: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in groups:
        gid = str(group["group_id"])
        source_id = str(group["source_id"])
        profile = profiles.get(gid)
        if not profile:
            continue
        rows.append(
            {
                "source_id": source_id,
                "group_id": gid,
                "instance": "all",
                "participant_count": profile.get("participant_count", 0),
                "source_participant_count": profile.get("source_participant_count", 0),
                "source_percent": profile.get("source_percent", ""),
            }
        )
        for row in profile.get("visit_coverage", []) or []:
            updated = dict(row)
            updated["source_id"] = source_id
            updated["group_id"] = gid
            rows.append(updated)
    return rows


def reuse_cached_profiles_by_raw_columns(
    existing: dict[str, object],
    groups: list[dict[str, object]],
) -> dict[str, dict[str, object]] | None:
    old_signatures = cached_group_signatures()
    if not old_signatures:
        return None
    old_profiles = existing.get("group_profiles") or {}
    reused: dict[str, dict[str, object]] = {}
    for group in groups:
        gid = str(group["group_id"])
        source_id = str(group["source_id"])
        signature = raw_column_signature(group.get("raw_columns", ""))
        old_gid = old_signatures.get((source_id, signature))
        if not old_gid:
            return None
        profile = old_profiles.get(old_gid)
        if not isinstance(profile, dict):
            return None
        reused[gid] = clone_cached_profile(profile, source_id, gid)
    return reused


def profile_semantic_coverage(
    groups: list[dict[str, object]],
    columns: list[dict[str, str]],
    inventory: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fingerprint = profile_fingerprint(groups)
    cached = None
    partial_existing = None
    sources_to_scan: set[str] | None = None
    if PROFILE_CACHE.exists():
        try:
            existing = json.loads(PROFILE_CACHE.read_text(encoding="utf-8"))
            if existing.get("fingerprint") == fingerprint:
                cached = existing
            else:
                old_fp = existing.get("fingerprint", {})
                same_sources = old_fp.get("source_files") == fingerprint.get("source_files")
                same_sampling = old_fp.get("sample_seed") == fingerprint.get("sample_seed")
                same_missing_tokens = old_fp.get("missing_value_tokens") == fingerprint.get("missing_value_tokens")
                if same_sources and same_sampling and same_missing_tokens:
                    reused_profiles = reuse_cached_profiles_by_raw_columns(existing, groups)
                    if reused_profiles is not None:
                        source_rows = {r["source_id"]: to_int(r["row_count"]) for r in inventory}
                        group_coverage_rows = coverage_rows_from_profiles(groups, reused_profiles)
                        node_coverage_rows = build_node_coverage(groups, reused_profiles, source_rows, "hide_missing_note")
                        node_coverage_rows.extend(build_node_coverage(groups, reused_profiles, source_rows, "all"))
                        cached = {
                            "fingerprint": fingerprint,
                            "group_profiles": reused_profiles,
                            "group_coverage_rows": group_coverage_rows,
                            "node_coverage_rows": node_coverage_rows,
                        }
                        PROFILE_CACHE.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
                        print("Reusing cached semantic profiles by raw-column signature", flush=True)
                cached_group_ids = set((existing.get("group_profiles") or {}).keys())
                current_group_ids = {str(group["group_id"]) for group in groups}
                if cached is None and same_sources and same_sampling and same_missing_tokens and cached_group_ids == current_group_ids:
                    source_rows = {r["source_id"]: to_int(r["row_count"]) for r in inventory}
                    existing_profiles = existing.get("group_profiles") or {}
                    group_coverage_rows = coverage_rows_from_profiles(groups, existing_profiles)
                    node_coverage_rows = build_node_coverage(groups, existing_profiles, source_rows, "hide_missing_note")
                    node_coverage_rows.extend(build_node_coverage(groups, existing_profiles, source_rows, "all"))
                    cached = {
                        "fingerprint": fingerprint,
                        "group_profiles": existing_profiles,
                        "group_coverage_rows": group_coverage_rows,
                        "node_coverage_rows": node_coverage_rows,
                    }
                    PROFILE_CACHE.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
                    print("Reusing cached semantic profiles with rebuilt tree coverage", flush=True)
                elif cached is None and same_sources and same_sampling and same_missing_tokens:
                    cached_gid_source = source_ids_by_group_from_coverage(existing.get("group_coverage_rows") or [])
                    cached_ids_by_source: dict[str, set[str]] = defaultdict(set)
                    for gid in cached_group_ids:
                        source_id = cached_gid_source.get(gid, "")
                        if source_id:
                            cached_ids_by_source[source_id].add(gid)
                    current_ids_by_source: dict[str, set[str]] = defaultdict(set)
                    for group in groups:
                        current_ids_by_source[str(group["source_id"])].add(str(group["group_id"]))
                    changed_sources = {
                        source_id
                        for source_id, current_ids in current_ids_by_source.items()
                        if current_ids != cached_ids_by_source.get(source_id, set())
                    }
                    if changed_sources and changed_sources != set(current_ids_by_source):
                        partial_existing = existing
                        sources_to_scan = changed_sources
        except json.JSONDecodeError:
            cached = None

    if cached:
        group_profiles = cached["group_profiles"]
        group_coverage_rows = cached["group_coverage_rows"]
        node_coverage_rows = cached["node_coverage_rows"]
    else:
        column_lookup, partner_lookup = build_column_lookup(columns)
        source_rows = {r["source_id"]: to_int(r["row_count"]) for r in inventory}
        source_profiles: dict[str, dict[str, object]] = {}
        group_coverage_rows = []
        node_coverage_rows = []
        current_group_ids = {str(group["group_id"]) for group in groups}
        if partial_existing and sources_to_scan is not None:
            cached_gid_source = source_ids_by_group_from_coverage(partial_existing.get("group_coverage_rows") or [])
            reused_sources = set(SOURCE_FILES) - sources_to_scan
            for gid, profile in (partial_existing.get("group_profiles") or {}).items():
                if gid in current_group_ids and cached_gid_source.get(gid) in reused_sources:
                    source_profiles[gid] = profile
            group_coverage_rows.extend(
                row
                for row in partial_existing.get("group_coverage_rows", [])
                if str(row.get("source_id", "")) in reused_sources and str(row.get("group_id", "")) in current_group_ids
            )
            print(
                "Reusing cached coverage for "
                + ", ".join(sorted(reused_sources))
                + "; rescanning "
                + ", ".join(sorted(sources_to_scan)),
                flush=True,
            )
        scan_sources = sources_to_scan if sources_to_scan is not None else set(SOURCE_FILES)
        scanned_profiles: dict[str, dict[str, object]] = {}
        scanned_groups = [group for group in groups if str(group["source_id"]) in scan_sources]
        for source_id, path in SOURCE_FILES.items():
            if source_id not in scan_sources:
                continue
            if not path.exists():
                continue
            print(f"Profiling {source_id}: {path}", flush=True)
            profiles, coverage_rows = scan_source_profile(
                source_id,
                path,
                source_rows.get(source_id, 0),
                groups,
                column_lookup,
                partner_lookup,
            )
            print(f"Finished {source_id}: {len(profiles):,} semantic groups", flush=True)
            source_profiles.update(profiles)
            scanned_profiles.update(profiles)
            group_coverage_rows.extend(coverage_rows)

        node_coverage_rows = build_node_coverage(groups, source_profiles, source_rows, "hide_missing_note")
        node_coverage_rows.extend(build_node_coverage(groups, source_profiles, source_rows, "all"))

        group_profiles = {}
        for gid, profile in source_profiles.items():
            group_profiles[gid] = {
                "participant_count": profile["participant_count"],
                "source_participant_count": profile["source_participant_count"],
                "source_percent": profile["source_percent"],
                "visit_coverage": profile["visit_coverage"],
                "examples": profile["examples"],
                "example_status": profile["example_status"],
            }
        cache_payload = {
            "fingerprint": fingerprint,
            "group_profiles": group_profiles,
            "group_coverage_rows": group_coverage_rows,
            "node_coverage_rows": node_coverage_rows,
        }
        PROFILE_CACHE.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")

    profile_by_group = {gid: profile for gid, profile in group_profiles.items()}
    for group in groups:
        gid = str(group["group_id"])
        profile = profile_by_group.get(gid)
        if not profile:
            group["example_entries"] = []
            group["example_entries_json"] = "[]"
            group["example_status"] = "Source CSV was not profiled"
            group["visit_coverage"] = []
            group["visit_coverage_json"] = "[]"
            continue
        count = to_int(profile.get("participant_count", 0))
        source_n = to_int(profile.get("source_participant_count", group.get("source_participant_count", 0)))
        visit_coverage = profile.get("visit_coverage", [])
        examples = profile.get("examples", [])
        group["participant_count_proxy"] = count
        group["coverage_count_type"] = "exact_any_non_missing"
        group["source_participant_count"] = source_n
        group["source_percent_proxy"] = pct(count, source_n)
        group["global_percent_proxy"] = ""
        group["visit_coverage"] = visit_coverage
        group["visit_coverage_json"] = json.dumps(visit_coverage, ensure_ascii=False)
        group["example_entries"] = examples
        group["example_entries_json"] = json.dumps(examples, ensure_ascii=False)
        group["example_status"] = profile.get("example_status", "")

    return groups, group_coverage_rows, node_coverage_rows


def node_coverage_index(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {f'{r["coverage_scope"]}|{r["tree_path"]}|{r["instance"]}': r for r in rows}


BASIC_PARTICIPANT_FIELD_IDS = {
    "31",
    "34",
    "52",
    "53",
    "54",
    "55",
    "21000",
    "21003",
    "22001",
    "22006",
    "22009",
    "22020",
}
BASIC_PARTICIPANT_LOCAL_COLUMNS = {"gender", "baselineage"}
MISSINGNESS_GROUP_FIELDS = {
    "sex": ["31", "gender"],
    "age": ["21003", "baselineage"],
    "ethnicity": ["21000", "22006"],
    "assessment_center": ["54"],
}
FALSE_LIKE_VALUES = {"0", "0.0", "-1", "-3", "-7", "false", "no", "none", "nan", "na", ""}
AGE_BINS = [(0, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 200)]
PROFILE_VALUE_CODING_IDS = {
    "Sex / gender": "9",
    "Genetic sex": "9",
    "Ethnic background": "1001",
    "Assessment center": "10",
}
MISSINGNESS_GROUP_CODING_IDS = {
    "sex": "9",
    "ethnicity": "1001",
    "assessment_center": "10",
}
EXCLUDED_PROFILE_VARIABLES = {"Genetic ethnic grouping"}


def load_profile_coding_labels() -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    needed = set(PROFILE_VALUE_CODING_IDS.values()) | set(MISSINGNESS_GROUP_CODING_IDS.values())
    encoding_paths = [
        ROOT / "UKBF" / "eye and kidney" / "encoding.ukb",
        ROOT / "UKBF" / "eye and brain" / "encoding.ukb",
    ]
    text = ""
    for path in encoding_paths:
        if path.exists():
            text = path.read_text(encoding="latin1")
            break
    if not text:
        return labels
    for encoding_id in needed:
        match = re.search(r"&R#e" + re.escape(encoding_id) + r"=\[(.*?)(?=&R#e\d+=\[|\]\])", text)
        if not match:
            continue
        pairs = re.findall(r"s#va=([^&\]]+).*?s#mn=([^&\]]+)", match.group(1))
        labels[encoding_id] = {
            urllib.parse.unquote_plus(value): urllib.parse.unquote_plus(name)
            for value, name in pairs
        }
    return labels


def profile_coding_labels() -> dict[str, dict[str, str]]:
    if not hasattr(profile_coding_labels, "_cache"):
        profile_coding_labels._cache = load_profile_coding_labels()  # type: ignore[attr-defined]
    return profile_coding_labels._cache  # type: ignore[attr-defined]


def annotated_profile_value(variable: str, value: object) -> str:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return ""
    coding_id = PROFILE_VALUE_CODING_IDS.get(variable)
    if not coding_id:
        return raw
    return profile_coding_labels().get(coding_id, {}).get(raw, raw)


def annotated_missingness_group_value(group_variable: str, value: object) -> str:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return ""
    coding_id = MISSINGNESS_GROUP_CODING_IDS.get(group_variable)
    if not coding_id:
        return raw
    return profile_coding_labels().get(coding_id, {}).get(raw, raw)


def safe_float(value: object) -> float:
    try:
        if is_missing_value(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(val):
        return ""
    if abs(val) >= 1000 or (abs(val) < 0.001 and val != 0):
        return f"{val:.{digits}g}"
    return f"{val:.{digits}f}".rstrip("0").rstrip(".")


def age_band(value: object) -> str:
    val = safe_float(value)
    if not math.isfinite(val):
        return "Missing"
    for lo, hi in AGE_BINS:
        if lo <= val <= hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    return "Other"


def participant_variable(row: dict[str, str]) -> tuple[str, str] | None:
    field_id = row.get("field_id", "")
    name = row.get("column_name", "")
    label = {
        "31": "Sex / gender",
        "34": "Year of birth",
        "52": "Month of birth",
        "53": "Assessment date",
        "54": "Assessment center",
        "55": "Assessment month",
        "21000": "Ethnic background",
        "21003": "Age at assessment",
        "22001": "Genetic sex",
        "22006": "Genetic ethnic grouping",
        "22009": "Genetic principal components",
        "22020": "Used in genetic principal components",
    }.get(field_id)
    if name in BASIC_PARTICIPANT_LOCAL_COLUMNS:
        label = "Sex / gender" if name == "gender" else "Age at assessment"
    if not label:
        return None
    value_type = infer_column_value_type(row, infer_column_semantic_function(row, "scalar"), "scalar")
    profile_kind = "numeric" if value_type in {"continuous numeric", "integer"} else "categorical"
    if value_type == "date":
        profile_kind = "date"
    return label, profile_kind


def feature_distribution_theme(group: dict[str, object]) -> str:
    semantic_function = str(group.get("semantic_function", ""))
    value_type = str(group.get("value_type", ""))
    requested_area = str(group.get("requested_area", ""))
    requested_type = str(group.get("requested_type", ""))
    storage = str(group.get("storage_pattern", ""))

    if semantic_function not in {"primary", "unknown"}:
        return ""
    if value_type in {"date", "time", "ID/file-like"}:
        return ""
    if storage in {"bulk_image_id", "bulk_file_id", "alignment_metadata"}:
        return ""
    if requested_area == "Health records data":
        return ""
    if requested_area == "Biomarker data":
        return requested_type
    if requested_area == "Other retrieved data" and "z-score" in requested_type.lower():
        return "Unverified local z-scores"
    if requested_area == "Imaging data" and requested_type in {"Brain MRI", "DXA/DEXA"}:
        return requested_type
    if requested_area == "Physical measurements data" and requested_type == "Vision / ophthalmic measurements":
        return "Vision / ophthalmic measurements"
    if requested_area == "Questionnaire data" and requested_type in {"Cognitive function", "Mental health / wellbeing"}:
        return requested_type
    return ""


def repeatability_theme(group: dict[str, object]) -> str:
    if str(group.get("semantic_function", "")) != "primary":
        return ""
    if str(group.get("value_type", "")) not in {"continuous numeric", "integer"}:
        return ""
    if to_int(group.get("visit_instance_count", 1)) <= 1:
        return ""
    if to_int(group.get("repeated_entry_count", 1)) > 1:
        return ""
    if str(group.get("storage_pattern", "")) != "repeated_visit":
        return ""
    theme = feature_distribution_theme(group)
    if theme:
        return theme
    if str(group.get("requested_type", "")) == "General physical measures":
        return "General physical measures (repeatability only)"
    return ""


def group_source_columns(group: dict[str, object]) -> list[str]:
    return [
        source_column_name(raw, str(group.get("source_id", "")))
        for raw in str(group.get("raw_columns", "")).split(";")
        if raw
    ]


def participant_specs(columns: list[dict[str, str]]) -> list[dict[str, object]]:
    specs = []
    for row in columns:
        if row.get("field_id") not in BASIC_PARTICIPANT_FIELD_IDS and row.get("column_name") not in BASIC_PARTICIPANT_LOCAL_COLUMNS:
            continue
        variable = participant_variable(row)
        if not variable:
            continue
        label, profile_kind = variable
        specs.append(
            {
                "profile_id": f'participant:{row["source_id"]}:{row["column_name"]}',
                "source_id": row["source_id"],
                "source_label": SOURCE_LABELS.get(row["source_id"], row["source_id"]),
                "column": row["column_name"],
                "field_id": row.get("field_id", ""),
                "instance": row.get("instance", ""),
                "variable": label,
                "profile_kind": profile_kind,
                "description": row.get("description", ""),
            }
        )
    return specs


def build_feature_profile_specs(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    specs = []
    for group in groups:
        theme = feature_distribution_theme(group)
        if not theme:
            continue
        cols = group_source_columns(group)
        if not cols:
            continue
        specs.append(
            {
                "profile_id": f'feature:{group["group_id"]}',
                "group_id": group["group_id"],
                "source_id": group["source_id"],
                "source_label": group["source_label"],
                "theme": theme,
                "feature_label": group["tree_leaf_label"],
                "value_type": group["value_type"],
                "semantic_function": group["semantic_function"],
                "column_count": len(cols),
                "columns": cols,
            }
        )
    return specs


def build_repeatability_specs(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    specs = []
    for group in groups:
        theme = repeatability_theme(group)
        if not theme:
            continue
        by_instance: dict[str, list[str]] = defaultdict(list)
        cols = group_source_columns(group)
        instances = [item for item in str(group.get("instances", "")).split(";") if item]
        raw_cols = [raw for raw in str(group.get("raw_columns", "")).split(";") if raw]
        for raw in raw_cols:
            source_id = str(group.get("source_id", ""))
            col = source_column_name(raw, source_id)
            match = re.search(r"-(\d+)\.", col)
            if match:
                by_instance[match.group(1)].append(col)
        if len(by_instance) < 2 and len(cols) >= 2 and len(instances) >= 2:
            for inst, col in zip(instances, cols):
                by_instance[inst].append(col)
        if len(by_instance) < 2:
            continue
        specs.append(
            {
                "profile_id": f'repeatability:{group["group_id"]}',
                "group_id": group["group_id"],
                "source_id": group["source_id"],
                "source_label": group["source_label"],
                "theme": theme,
                "feature_label": group["tree_leaf_label"],
                "value_type": group["value_type"],
                "instances": sorted(by_instance, key=to_int),
                "columns_by_instance": {inst: cols[:1] for inst, cols in by_instance.items()},
            }
        )
    return specs


def metabolite_curated_disease_spec(row: dict[str, str]) -> dict[str, object] | None:
    if row.get("source_id") != "metabolite_csv":
        return None
    name = row.get("column_name", "")
    role = row.get("feature_role", "")
    is_curated = (
        "outcome" in role
        or name.startswith("incident_")
        or name.startswith("prior_")
        or name.startswith("followup_")
        or name.endswith("_2104")
        or name.endswith("_eligebility")
        or name.endswith("_eligibility")
    )
    if not is_curated:
        return None
    if name in {"gender", "baselineage"}:
        return None
    if name.startswith("followup_"):
        target_type = "follow-up years"
    elif name.endswith("_date_2104"):
        target_type = "event date"
    elif name.startswith("incident_"):
        target_type = "incident flag"
    elif name.startswith("prior_"):
        target_type = "prior/prevalent flag"
    elif name.endswith("_eligebility") or name.endswith("_eligibility"):
        target_type = "eligibility flag"
    else:
        target_type = "curated status flag"
    return {
        "profile_id": f'disease:metabolite:{name}',
        "source_id": "metabolite_csv",
        "source_label": SOURCE_LABELS["metabolite_csv"],
        "target_name": humanize_metabolite_local_name(row),
        "target_family": disease_family(name),
        "target_type": target_type,
        "columns": [name],
        "case_only": False,
        "is_binary_flag": target_type in {"incident flag", "prior/prevalent flag", "eligibility flag", "curated status flag"},
        "is_numeric": target_type == "follow-up years",
    }


def build_disease_profile_specs(groups: list[dict[str, object]], columns: list[dict[str, str]]) -> list[dict[str, object]]:
    specs = []
    seen = set()
    for group in groups:
        if str(group.get("requested_area", "")) != "Health records data":
            continue
        if str(group.get("semantic_function", "")) == "missing_note":
            continue
        cols = group_source_columns(group)
        if not cols:
            continue
        profile_id = f'disease:group:{group["group_id"]}'
        seen.add(profile_id)
        specs.append(
            {
                "profile_id": profile_id,
                "source_id": group["source_id"],
                "source_label": group["source_label"],
                "target_name": group["tree_leaf_label"],
                "target_family": disease_family(str(group.get("tree_leaf_label", ""))),
                "target_type": group["requested_type"],
                "columns": cols,
                "case_only": str(group.get("case_only_field", "")).lower() == "true",
                "is_binary_flag": False,
                "is_numeric": str(group.get("value_type", "")) in {"continuous numeric", "integer"},
            }
        )
    for row in columns:
        spec = metabolite_curated_disease_spec(row)
        if spec and spec["profile_id"] not in seen:
            specs.append(spec)
            seen.add(str(spec["profile_id"]))
    return specs


def init_numeric_acc(denominator: int, z_scored: bool = False) -> dict[str, object]:
    return {
        "denominator": denominator,
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "sumcube": 0.0,
        "zero_count": 0,
        "extreme_count": 0,
        "z_scored": z_scored,
        "min": math.inf,
        "max": -math.inf,
        "sample": [],
    }


def update_numeric_acc(acc: dict[str, object], values: np.ndarray) -> None:
    if values.size == 0:
        return
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    count = int(values.size)
    acc["count"] = to_int(acc["count"]) + count
    acc["sum"] = float(acc["sum"]) + float(values.sum())
    acc["sumsq"] = float(acc["sumsq"]) + float(np.square(values).sum())
    acc["sumcube"] = float(acc["sumcube"]) + float(np.power(values, 3).sum())
    acc["zero_count"] = to_int(acc["zero_count"]) + int(np.count_nonzero(values == 0))
    acc["extreme_count"] = to_int(acc["extreme_count"]) + int(np.count_nonzero(np.abs(values) > 5))
    acc["min"] = min(float(acc["min"]), float(values.min()))
    acc["max"] = max(float(acc["max"]), float(values.max()))
    sample = acc["sample"]
    if isinstance(sample, list) and len(sample) < BASIC_PROFILE_SAMPLE_LIMIT:
        needed = BASIC_PROFILE_SAMPLE_LIMIT - len(sample)
        sample.extend(float(v) for v in values[:needed])


def init_counter_acc(denominator: int) -> dict[str, object]:
    return {
        "denominator": denominator,
        "count": 0,
        "counter": Counter(),
        "unique_overflow": False,
    }


def update_counter_acc(acc: dict[str, object], values: pd.Series) -> None:
    non_missing = values.astype(str)
    non_missing = non_missing[~non_missing.isin(MISSING_VALUE_TOKENS)]
    if non_missing.empty:
        return
    acc["count"] = to_int(acc["count"]) + int(len(non_missing))
    counts = non_missing.value_counts(dropna=False)
    counter: Counter = acc["counter"]
    for value, count in counts.items():
        key = html_truncate(value, 120)
        if key in counter or len(counter) < BASIC_PROFILE_COUNTER_LIMIT:
            counter[key] += int(count)
        else:
            acc["unique_overflow"] = True


def numeric_summary_from_acc(acc: dict[str, object]) -> dict[str, str]:
    n = to_int(acc.get("count"))
    denom = to_int(acc.get("denominator"))
    if n == 0:
        return {
            "non_missing_count": "0",
            "missing_percent": pct(denom, denom),
            "mean": "",
            "sd": "",
            "median": "",
            "q1": "",
            "q3": "",
            "min": "",
            "max": "",
            "zero_percent": "",
            "skewness": "",
            "extreme_z_percent": "",
        }
    total = float(acc["sum"])
    sumsq = float(acc["sumsq"])
    sumcube = float(acc["sumcube"])
    mean = total / n
    variance = max(0.0, (sumsq / n) - mean * mean)
    sd = math.sqrt(variance)
    if variance > 0:
        central3 = (sumcube / n) - 3 * mean * (sumsq / n) + 2 * mean**3
        skewness = central3 / (variance ** 1.5)
    else:
        skewness = math.nan
    sample = np.array(acc.get("sample") or [], dtype=float)
    q1 = median = q3 = math.nan
    if sample.size:
        q1, median, q3 = np.quantile(sample, [0.25, 0.5, 0.75])
    missing = max(0, denom - n)
    extreme = to_int(acc.get("extreme_count")) if acc.get("z_scored") else 0
    return {
        "non_missing_count": str(n),
        "missing_percent": pct(missing, denom),
        "mean": fmt_float(mean),
        "sd": fmt_float(sd),
        "median": fmt_float(median),
        "q1": fmt_float(q1),
        "q3": fmt_float(q3),
        "min": fmt_float(acc.get("min")),
        "max": fmt_float(acc.get("max")),
        "zero_percent": pct(to_int(acc.get("zero_count")), n),
        "skewness": fmt_float(skewness),
        "extreme_z_percent": pct(extreme, n) if acc.get("z_scored") else "",
    }


def positive_values(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    return (~text.isin(FALSE_LIKE_VALUES)) & (~values.astype(str).isin(MISSING_VALUE_TOKENS))


def values_for_columns(chunk: pd.DataFrame, columns: list[str]) -> pd.Series:
    cols = [col for col in columns if col in chunk.columns]
    if not cols:
        return pd.Series(dtype=str)
    if len(cols) == 1:
        return chunk[cols[0]]
    return chunk[cols].stack()


def numeric_values_for_columns(chunk: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = values_for_columns(chunk, columns)
    if values.empty:
        return np.array([], dtype=float)
    values = values.astype(str)
    values = values[~values.isin(MISSING_VALUE_TOKENS)]
    return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)


def selected_column_schema(specs: dict[str, list[dict[str, object]]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(specs):
        for spec in sorted(specs[key], key=lambda item: str(item.get("profile_id", ""))):
            cols = spec.get("columns", [spec.get("column", "")])
            digest.update(
                "\t".join(
                    [
                        key,
                        str(spec.get("profile_id", "")),
                        str(spec.get("source_id", "")),
                        str(spec.get("theme", "")),
                        str(spec.get("value_type", "")),
                        ";".join(str(c) for c in cols if c),
                    ]
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def basic_profile_fingerprint(specs: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {
        "profile_version": BASIC_PROFILE_VERSION,
        "source_files": source_file_stats(),
        "missing_value_tokens": sorted(MISSING_VALUE_TOKENS),
        "selected_column_schema": selected_column_schema(specs),
    }


def current_group_id_by_cached_group_id(groups: list[dict[str, object]]) -> dict[str, str]:
    old_signatures = cached_group_signatures()
    current_by_signature = {
        (str(group["source_id"]), raw_column_signature(group.get("raw_columns", ""))): str(group["group_id"])
        for group in groups
    }
    mapping = {}
    for signature_key, old_gid in old_signatures.items():
        new_gid = current_by_signature.get(signature_key)
        if new_gid:
            mapping[old_gid] = new_gid
    return mapping


def remap_cached_basic_profiles(
    profiles: dict[str, list[dict[str, object]]],
    groups: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    group_id_map = current_group_id_by_cached_group_id(groups)
    groups_by_id = {str(group["group_id"]): group for group in groups}
    remapped = json.loads(json.dumps(profiles, ensure_ascii=False))
    for key in ["feature_distributions", "repeatability_cv_icc"]:
        for row in remapped.get(key, []) or []:
            old_gid = str(row.get("group_id", ""))
            new_gid = group_id_map.get(old_gid, old_gid)
            if not new_gid:
                continue
            row["group_id"] = new_gid
            group = groups_by_id.get(new_gid)
            if group:
                row["feature_label"] = group.get("tree_leaf_label", row.get("feature_label", ""))
                row["value_type"] = group.get("value_type", row.get("value_type", ""))
                if "column_count" in row:
                    row["column_count"] = group.get("feature_count", row.get("column_count", ""))
                theme = feature_distribution_theme(group)
                if theme and key == "feature_distributions":
                    row["theme"] = theme
                repeat_theme = repeatability_theme(group)
                if repeat_theme and key == "repeatability_cv_icc":
                    row["theme"] = repeat_theme
    return remapped


def can_reuse_basic_profile_cache(cached: dict[str, object], fingerprint: dict[str, object]) -> bool:
    old_fp = cached.get("fingerprint", {})
    return (
        old_fp.get("profile_version") == fingerprint.get("profile_version")
        and old_fp.get("source_files") == fingerprint.get("source_files")
        and old_fp.get("missing_value_tokens") == fingerprint.get("missing_value_tokens")
    )


def source_specs(specs: list[dict[str, object]], source_id: str) -> list[dict[str, object]]:
    return [spec for spec in specs if str(spec.get("source_id")) == source_id]


def first_source_column(columns: list[dict[str, str]], source_id: str, candidates: list[str]) -> str:
    source_rows = [row for row in columns if row["source_id"] == source_id]
    for candidate in candidates:
        matching = [
            row
            for row in source_rows
            if row.get("field_id") == candidate or row.get("column_name") == candidate
        ]
        if not matching:
            continue
        matching.sort(key=lambda row: (to_int(row.get("instance")), to_int(row.get("array")), to_int(row.get("column_index"))))
        return matching[0]["column_name"]
    return ""


def disease_status_specs_for_missingness(disease_specs: list[dict[str, object]]) -> list[dict[str, object]]:
    specs = [
        spec
        for spec in disease_specs
        if spec.get("source_id") == "metabolite_csv"
        and spec.get("is_binary_flag")
        and not str(spec.get("target_type", "")).startswith("eligibility")
        and not str(spec.get("target_name", "")).lower().startswith("incident")
        and not str(spec.get("target_name", "")).lower().startswith("prior")
    ]
    return specs[:40]


def cooccurrence_specs(disease_specs: list[dict[str, object]]) -> list[dict[str, object]]:
    specs = [
        spec
        for spec in disease_specs
        if spec.get("source_id") == "metabolite_csv"
        and spec.get("is_binary_flag")
        and str(spec.get("target_type")) == "curated status flag"
    ]
    return specs[:36]


def disease_positive_mode(spec: dict[str, object]) -> str:
    target_type = str(spec.get("target_type", "")).lower()
    target_name = str(spec.get("target_name", "")).lower()
    if spec.get("source_id") == "metabolite_csv" and target_type in {"curated disease labels", "prevalent disease covariates"}:
        return ""
    if "eligibility" in target_type or "follow-up" in target_type or "censoring" in target_type:
        return ""
    if "method of recording" in target_name:
        return ""
    if spec.get("is_binary_flag"):
        return "binary"
    if target_name.startswith("number of self-reported"):
        return "numeric_positive"
    if any(k in target_type for k in ["date", "linked endpoint", "death data", "cancer data", "hospital inpatient", "self-reported non-cancer illness"]):
        return "nonempty"
    if any(k in target_name for k in ["code", "diagnos", "illness", "cancer", "death", "cause", "age ", "year/age", "fracture", "glaucoma", "asthma", "diabetes"]):
        return "nonempty"
    return ""


def disease_family_prevalence_specs(disease_specs: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for spec in disease_specs:
        mode = disease_positive_mode(spec)
        if not mode:
            continue
        cloned = dict(spec)
        cloned["positive_mode"] = mode
        by_family[str(spec.get("target_family", "Other chronic disease / phenotype"))].append(cloned)
    return by_family


def update_disease_family_counts_from_chunk(
    counts: Counter,
    family_specs: dict[str, list[dict[str, object]]],
    chunk: pd.DataFrame,
) -> None:
    non_empty = non_missing_frame(chunk)
    for family, specs_for_family in family_specs.items():
        family_mask = np.zeros(len(chunk), dtype=bool)
        for spec in specs_for_family:
            cols = [str(col) for col in spec["columns"] if str(col) in chunk.columns]
            if not cols:
                continue
            mode = str(spec.get("positive_mode", ""))
            if mode == "binary":
                for col in cols:
                    family_mask |= positive_values(chunk[col]).to_numpy()
            elif mode == "numeric_positive":
                numeric = chunk[cols].apply(pd.to_numeric, errors="coerce")
                family_mask |= (numeric > 0).any(axis=1).to_numpy()
            elif mode == "nonempty":
                for col in cols:
                    family_mask |= positive_values(chunk[col]).to_numpy()
        counts[family] += int(family_mask.sum())


def disease_family_prevalence_rows(
    disease_specs: list[dict[str, object]],
    inventory: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows = []
    source_rows = {row["source_id"]: to_int(row["row_count"]) for row in inventory}
    for source_id, path in SOURCE_FILES.items():
        specs = source_specs(disease_specs, source_id)
        family_specs = disease_family_prevalence_specs(specs)
        usecols = sorted({str(col) for family in family_specs.values() for spec in family for col in spec["columns"]})
        if not path.exists() or not usecols:
            continue
        header = set(pd.read_csv(path, nrows=0, dtype=str, na_filter=False, encoding="latin1").columns)
        usecols = [col for col in usecols if col in header]
        if not usecols:
            continue
        counts: Counter = Counter()
        reader = pd.read_csv(
            path,
            dtype=str,
            na_filter=False,
            encoding="latin1",
            chunksize=BASIC_PROFILE_CHUNK_SIZE,
            usecols=usecols,
            low_memory=False,
        )
        for chunk in reader:
            update_disease_family_counts_from_chunk(counts, family_specs, chunk)
        row_count = source_rows.get(source_id, 0)
        for family, count in sorted(counts.items()):
            if count <= 0:
                continue
            rows.append(
                {
                    "source_id": source_id,
                    "source_label": SOURCE_LABELS.get(source_id, source_id),
                    "profile_type": "disease_family_prevalence",
                    "target_family": family,
                    "target_name": f"Any positive {family} record",
                    "target_type": "participants with at least one positive disease indicator",
                    "columns": "",
                    "case_only": False,
                    "participant_count": count,
                    "target_count": count,
                    "percent": pct(count, row_count),
                    "summary_json": "",
                }
            )
    return rows


def attach_disease_family_prevalence(
    profiles: dict[str, list[dict[str, object]]],
    disease_specs: list[dict[str, object]],
    inventory: list[dict[str, str]],
) -> dict[str, list[dict[str, object]]]:
    cloned = {key: [dict(row) for row in value] for key, value in profiles.items()}
    cloned["disease_targets"] = [
        row
        for row in cloned.get("disease_targets", [])
        if row.get("profile_type") != "disease_family_prevalence"
    ]
    cloned["disease_targets"].extend(disease_family_prevalence_rows_fast(cloned, disease_specs, inventory))
    return cloned


def target_summary_supports_prevalence_proxy(row: dict[str, object]) -> bool:
    name = str(row.get("target_name", "")).lower()
    target_type = str(row.get("target_type", "")).lower()
    if any(k in target_type for k in ["follow-up", "censoring", "eligibility"]):
        return False
    if name.startswith("number of ") or name.startswith("method of "):
        return False
    return any(
        k in name
        for k in [
            " code",
            "date of ",
            "date/source",
            "first diagnosed",
            "age ",
            "year/age",
            "cause of death",
            "non-cancer illness",
        ]
    ) or str(row.get("case_only", "")).lower() == "true"


def prevalence_row(
    source_id: str,
    family: str,
    count: int,
    row_count: int,
    count_type: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_label": SOURCE_LABELS.get(source_id, source_id),
        "profile_type": "disease_family_prevalence",
        "target_family": family,
        "target_name": f"Any positive {family} record",
        "target_type": count_type,
        "columns": "",
        "case_only": False,
        "participant_count": count,
        "target_count": count,
        "percent": pct(count, row_count),
        "summary_json": "",
    }


def cached_source_prevalence_proxy_rows(
    profiles: dict[str, list[dict[str, object]]],
    inventory: list[dict[str, str]],
) -> list[dict[str, object]]:
    source_rows = {row["source_id"]: to_int(row["row_count"]) for row in inventory}
    best: dict[tuple[str, str], int] = {}
    for row in profiles.get("disease_targets", []):
        if row.get("profile_type") != "target_summary" or row.get("source_id") == "metabolite_csv":
            continue
        if not target_summary_supports_prevalence_proxy(row):
            continue
        key = (str(row.get("source_id", "")), str(row.get("target_family", "")))
        best[key] = max(best.get(key, 0), to_int(row.get("target_count")))
    return [
        prevalence_row(source_id, family, count, source_rows.get(source_id, 0), "source-specific positive-record proxy")
        for (source_id, family), count in sorted(best.items())
        if count > 0
    ]


def metabolite_exact_prevalence_rows(
    disease_specs: list[dict[str, object]],
    inventory: list[dict[str, str]],
) -> list[dict[str, object]]:
    source_id = "metabolite_csv"
    path = SOURCE_FILES[source_id]
    source_rows = {row["source_id"]: to_int(row["row_count"]) for row in inventory}
    specs = []
    for spec in disease_specs:
        if spec.get("source_id") != source_id:
            continue
        if not str(spec.get("profile_id", "")).startswith("disease:metabolite:"):
            continue
        mode = disease_positive_mode(spec)
        if not mode:
            continue
        cloned = dict(spec)
        cloned["positive_mode"] = mode
        specs.append(cloned)
    family_specs = disease_family_prevalence_specs(specs)
    usecols = sorted({str(col) for family in family_specs.values() for spec in family for col in spec["columns"]})
    if not path.exists() or not usecols:
        return []
    header = set(pd.read_csv(path, nrows=0, dtype=str, na_filter=False, encoding="latin1").columns)
    usecols = [col for col in usecols if col in header]
    if not usecols:
        return []
    counts: Counter = Counter()
    reader = pd.read_csv(
        path,
        dtype=str,
        na_filter=False,
        encoding="latin1",
        chunksize=BASIC_PROFILE_CHUNK_SIZE,
        usecols=usecols,
        low_memory=False,
    )
    for chunk in reader:
        update_disease_family_counts_from_chunk(counts, family_specs, chunk)
    row_count = source_rows.get(source_id, 0)
    return [
        prevalence_row(source_id, family, count, row_count, "participants with at least one positive curated disease indicator")
        for family, count in sorted(counts.items())
        if count > 0
    ]


def disease_family_prevalence_rows_fast(
    profiles: dict[str, list[dict[str, object]]],
    disease_specs: list[dict[str, object]],
    inventory: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows = cached_source_prevalence_proxy_rows(profiles, inventory)
    rows.extend(metabolite_exact_prevalence_rows(disease_specs, inventory))
    return rows


def normalize_basic_profiles(profiles: dict[str, list[dict[str, object]]]) -> dict[str, list[dict[str, object]]]:
    normalized = {key: [dict(row) for row in rows] for key, rows in profiles.items()}
    participant_rows = []
    for row in normalized.get("participant_groups", []):
        if row.get("variable") in EXCLUDED_PROFILE_VARIABLES:
            continue
        if row.get("profile_type") == "category_count":
            row["value_label"] = annotated_profile_value(str(row.get("variable", "")), row.get("value", ""))
        else:
            row.setdefault("value_label", "")
        participant_rows.append(row)
    normalized["participant_groups"] = participant_rows

    disease_rows = []
    for row in normalized.get("disease_targets", []):
        if row.get("profile_type") == "cooccurrence":
            continue
        target_text = f'{row.get("target_name", "")} {row.get("columns", "")}'
        row["target_family"] = disease_family(target_text)
        if disease_family(target_text) == "Self-reported non-cancer illness":
            row["target_type"] = "Self-reported non-cancer illness"
        target_type = str(row.get("target_type", "")).lower()
        is_status_like = "flag" in target_type or "eligibility" in target_type
        participant_count = to_int(row.get("participant_count"))
        if "positive_count" in row:
            positive = to_int(row.get("positive_count"))
            row["target_count"] = positive if is_status_like else participant_count
        else:
            row["target_count"] = to_int(row.get("target_count", participant_count))
        row.pop("positive_count", None)
        row.pop("top_values_json", None)
        disease_rows.append(row)
    normalized["disease_targets"] = disease_rows
    for row in normalized.get("feature_distributions", []):
        row.pop("top_values_json", None)
    missing_rows = []
    for row in normalized.get("missingness_patterns", []):
        if row.get("source_id") == "eye_brain_ukb43216" and row.get("group_variable") == "ethnicity":
            continue
        row["group_value_label"] = annotated_missingness_group_value(str(row.get("group_variable", "")), row.get("group_value", ""))
        missing_rows.append(row)
    normalized["missingness_patterns"] = missing_rows
    return normalized


def scan_basic_profile_source(
    source_id: str,
    path: Path,
    row_count: int,
    columns: list[dict[str, str]],
    part_specs: list[dict[str, object]],
    feature_specs: list[dict[str, object]],
    repeat_specs: list[dict[str, object]],
    disease_specs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    header = set(read_header(path))
    part_specs = [spec for spec in part_specs if spec["column"] in header]
    feature_specs = [
        {**spec, "columns": [col for col in spec["columns"] if col in header]}
        for spec in feature_specs
    ]
    feature_specs = [spec for spec in feature_specs if spec["columns"]]
    disease_specs = [
        {**spec, "columns": [col for col in spec["columns"] if col in header]}
        for spec in disease_specs
    ]
    disease_specs = [spec for spec in disease_specs if spec["columns"]]
    repeat_specs = [
        {
            **spec,
            "columns_by_instance": {
                inst: [col for col in cols if col in header]
                for inst, cols in spec["columns_by_instance"].items()
            },
        }
        for spec in repeat_specs
    ]
    repeat_specs = [
        spec for spec in repeat_specs if sum(bool(cols) for cols in spec["columns_by_instance"].values()) >= 2
    ]

    group_cols = {
        name: first_source_column(columns, source_id, candidates)
        for name, candidates in MISSINGNESS_GROUP_FIELDS.items()
    }
    group_cols = {name: col for name, col in group_cols.items() if col in header}
    theme_cols: dict[str, list[str]] = defaultdict(list)
    for spec in feature_specs:
        theme_cols[str(spec["theme"])].extend(str(col) for col in spec["columns"])

    usecols = set(group_cols.values())
    usecols.update(str(spec["column"]) for spec in part_specs)
    for spec in feature_specs + disease_specs:
        usecols.update(str(col) for col in spec["columns"])
    for spec in repeat_specs:
        for cols in spec["columns_by_instance"].values():
            usecols.update(str(col) for col in cols)
    usecols = sorted(col for col in usecols if col in header)
    if not usecols:
        return [], [], [], [], []

    participant_acc = {}
    for spec in part_specs:
        denom = row_count
        if spec["profile_kind"] == "numeric":
            participant_acc[spec["profile_id"]] = init_numeric_acc(denom)
            if "age" in str(spec["variable"]).lower():
                participant_acc[spec["profile_id"]]["hist"] = Counter()
        else:
            participant_acc[spec["profile_id"]] = init_counter_acc(denom)

    feature_acc = {}
    for spec in feature_specs:
        denom = row_count * len(spec["columns"])
        z_scored = "z-score" in str(spec["theme"]).lower() or "z-scored" in str(spec["feature_label"]).lower()
        if spec["value_type"] in {"continuous numeric", "integer"}:
            feature_acc[spec["profile_id"]] = init_numeric_acc(denom, z_scored=z_scored)
        else:
            feature_acc[spec["profile_id"]] = init_counter_acc(denom)

    disease_acc = {}
    for spec in disease_specs:
        disease_acc[spec["profile_id"]] = {
            "participant_count": 0,
            "positive_count": 0,
            "counter": Counter(),
            "numeric": init_numeric_acc(row_count * len(spec["columns"])) if spec.get("is_numeric") else None,
        }

    repeat_acc = {
        spec["profile_id"]: {
            "n_subjects": 0,
            "n_obs": 0,
            "within_ss": 0.0,
            "within_df": 0,
            "mean_sum": 0.0,
            "mean_sumsq": 0.0,
            "cv_sum": 0.0,
            "cv_count": 0,
            "pair_count": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_x2": 0.0,
            "sum_y2": 0.0,
            "sum_xy": 0.0,
            "visit_pair": "",
        }
        for spec in repeat_specs
    }

    missing_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"participants": 0, "with_data": 0})
    burden_counts: Counter = Counter()
    disease_missing_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"participants": 0, "with_data": 0})
    co_specs: list[dict[str, object]] = []
    co_matrix = np.zeros((0, 0), dtype=np.int64)

    theme_names = sorted(theme_cols)
    column_to_theme = {}
    for theme, cols in theme_cols.items():
        for col in cols:
            column_to_theme[col] = theme
    theme_reducer = make_nonempty_reducer(sorted(column_to_theme), column_to_theme)
    disease_status_specs = disease_status_specs_for_missingness(disease_specs)
    disease_family_specs = disease_family_prevalence_specs(disease_specs)
    disease_family_counts: Counter = Counter()

    reader = pd.read_csv(
        path,
        dtype=str,
        na_filter=False,
        encoding="latin1",
        chunksize=BASIC_PROFILE_CHUNK_SIZE,
        usecols=usecols,
        low_memory=False,
    )
    for chunk in reader:
        non_empty = non_missing_frame(chunk)

        for spec in part_specs:
            col = str(spec["column"])
            if col not in chunk.columns:
                continue
            acc = participant_acc[spec["profile_id"]]
            if spec["profile_kind"] == "numeric":
                values = numeric_values_for_columns(chunk, [col])
                update_numeric_acc(acc, values)
                if "hist" in acc:
                    acc["hist"].update(age_band(value) for value in values)
            else:
                update_counter_acc(acc, chunk[col])

        for spec in feature_specs:
            acc = feature_acc[spec["profile_id"]]
            cols = [str(col) for col in spec["columns"] if str(col) in chunk.columns]
            if not cols:
                continue
            if spec["value_type"] in {"continuous numeric", "integer"}:
                update_numeric_acc(acc, numeric_values_for_columns(chunk, cols))
            else:
                update_counter_acc(acc, values_for_columns(chunk, cols))

        for spec in disease_specs:
            cols = [str(col) for col in spec["columns"] if str(col) in chunk.columns]
            if not cols:
                continue
            acc = disease_acc[spec["profile_id"]]
            present = non_empty[cols].any(axis=1)
            acc["participant_count"] += int(present.sum())
            values = values_for_columns(chunk, cols)
            update_counter_acc({"count": 0, "counter": acc["counter"], "unique_overflow": False, "denominator": 0}, values)
            if spec.get("is_binary_flag"):
                positive = positive_values(chunk[cols[0]])
                acc["positive_count"] += int(positive.sum())
            if spec.get("is_numeric") and acc["numeric"] is not None:
                update_numeric_acc(acc["numeric"], numeric_values_for_columns(chunk, cols))

        for family, specs_for_family in disease_family_specs.items():
            family_mask = np.zeros(len(chunk), dtype=bool)
            for spec in specs_for_family:
                cols = [str(col) for col in spec["columns"] if str(col) in chunk.columns]
                if not cols:
                    continue
                mode = str(spec.get("positive_mode", ""))
                if mode == "binary":
                    for col in cols:
                        family_mask |= positive_values(chunk[col]).to_numpy()
                elif mode == "numeric_positive":
                    numeric = chunk[cols].apply(pd.to_numeric, errors="coerce")
                    family_mask |= (numeric > 0).any(axis=1).to_numpy()
                elif mode == "nonempty":
                    for col in cols:
                        family_mask |= positive_values(chunk[col]).to_numpy()
            disease_family_counts[family] += int(family_mask.sum())

        for spec in repeat_specs:
            acc = repeat_acc[spec["profile_id"]]
            inst_cols = []
            inst_labels = []
            for inst in spec["instances"]:
                cols = [col for col in spec["columns_by_instance"].get(inst, []) if col in chunk.columns]
                if not cols:
                    continue
                inst_labels.append(str(inst))
                inst_cols.append(cols[0])
            if len(inst_cols) < 2:
                continue
            frame = chunk[inst_cols].apply(pd.to_numeric, errors="coerce")
            arr = frame.to_numpy(dtype=float)
            valid = np.isfinite(arr)
            k = valid.sum(axis=1)
            rows = k >= 2
            if rows.any():
                arr_rows = arr[rows]
                means = np.nanmean(arr_rows, axis=1)
                centered = arr_rows - means[:, None]
                centered[~np.isfinite(centered)] = 0
                within_ss = np.square(centered).sum(axis=1)
                sd = np.sqrt(within_ss / np.maximum(1, k[rows] - 1))
                cv_mask = np.abs(means) > 0
                cvs = sd[cv_mask] / np.abs(means[cv_mask])
                acc["n_subjects"] += int(rows.sum())
                acc["n_obs"] += int(k[rows].sum())
                acc["within_ss"] += float(within_ss.sum())
                acc["within_df"] += int((k[rows] - 1).sum())
                acc["mean_sum"] += float(means.sum())
                acc["mean_sumsq"] += float(np.square(means).sum())
                acc["cv_sum"] += float(cvs.sum()) if cvs.size else 0.0
                acc["cv_count"] += int(cvs.size)
            pair_valid = valid[:, 0] & valid[:, 1]
            if pair_valid.any():
                x = arr[pair_valid, 0]
                y = arr[pair_valid, 1]
                acc["pair_count"] += int(pair_valid.sum())
                acc["sum_x"] += float(x.sum())
                acc["sum_y"] += float(y.sum())
                acc["sum_x2"] += float(np.square(x).sum())
                acc["sum_y2"] += float(np.square(y).sum())
                acc["sum_xy"] += float((x * y).sum())
                acc["visit_pair"] = f"{inst_labels[0]}-{inst_labels[1]}"

        theme_keys, theme_matrix = apply_nonempty_reducer(non_empty, theme_reducer)
        theme_index = {str(theme): idx for idx, theme in enumerate(theme_keys)}
        if theme_matrix is not None:
            theme_counts = theme_matrix.sum(axis=1)
            burden_counts.update(int(v) for v in theme_counts)
            for theme in theme_names:
                idx = theme_index.get(theme)
                if idx is None:
                    continue
                present = theme_matrix[:, idx]
                missing_counts[("ALL", "ALL", theme)]["participants"] += len(chunk)
                missing_counts[("ALL", "ALL", theme)]["with_data"] += int(present.sum())

            group_values = {}
            if "sex" in group_cols:
                group_values["sex"] = chunk[group_cols["sex"]].astype(str)
            if "age" in group_cols:
                group_values["age_band"] = chunk[group_cols["age"]].map(age_band).astype(str)
            if "ethnicity" in group_cols:
                group_values["ethnicity"] = chunk[group_cols["ethnicity"]].astype(str)
            if "assessment_center" in group_cols:
                group_values["assessment_center"] = chunk[group_cols["assessment_center"]].astype(str)
            for group_name, values in group_values.items():
                valid_group = ~values.isin(MISSING_VALUE_TOKENS)
                for value in sorted(values[valid_group].unique())[:100]:
                    mask = (values == value).to_numpy()
                    participants = int(mask.sum())
                    if participants == 0:
                        continue
                    for theme in theme_names:
                        idx = theme_index.get(theme)
                        if idx is None:
                            continue
                        present = theme_matrix[:, idx]
                        rec = missing_counts[(group_name, html_truncate(value, 80), theme)]
                        rec["participants"] += participants
                        rec["with_data"] += int((present & mask).sum())

            if source_id == "metabolite_csv":
                for disease_spec in disease_status_specs:
                    col = disease_spec["columns"][0]
                    if col not in chunk.columns:
                        continue
                    pos = positive_values(chunk[col]).to_numpy()
                    for status_label, status_mask in [("positive", pos), ("not_positive", ~pos)]:
                        participants = int(status_mask.sum())
                        if participants == 0:
                            continue
                        group_name = f'disease_status:{disease_spec["target_name"]}'
                        for theme in theme_names:
                            idx = theme_index.get(theme)
                            if idx is None:
                                continue
                            rec = disease_missing_counts[(group_name, status_label, theme)]
                            rec["participants"] += participants
                            rec["with_data"] += int((theme_matrix[:, idx] & status_mask).sum())

        if source_id == "metabolite_csv" and co_specs:
            bool_cols = []
            active_specs = []
            for spec in co_specs:
                col = spec["columns"][0]
                if col in chunk.columns:
                    bool_cols.append(positive_values(chunk[col]).to_numpy(dtype=np.int8))
                    active_specs.append(spec)
            if bool_cols:
                mat = np.vstack(bool_cols).T
                active_indices = [co_specs.index(spec) for spec in active_specs]
                co_matrix[np.ix_(active_indices, active_indices)] += mat.T @ mat

    participant_rows = []
    for spec in part_specs:
        acc = participant_acc.get(spec["profile_id"])
        if not acc:
            continue
        base = {
            "source_id": source_id,
            "source_label": SOURCE_LABELS.get(source_id, source_id),
            "profile_type": "participant_group",
            "variable": spec["variable"],
            "column": spec["column"],
            "field_id": spec["field_id"],
            "instance": spec["instance"],
        }
        if spec["profile_kind"] == "numeric":
            summary = numeric_summary_from_acc(acc)
            participant_rows.append({**base, "profile_type": "numeric_summary", **summary, "value": "", "count": summary["non_missing_count"], "percent": pct(to_int(summary["non_missing_count"]), row_count)})
            hist = acc.get("hist")
            if isinstance(hist, Counter) and hist:
                for lo, hi in AGE_BINS:
                    label = f"{lo}-{hi}" if hi < 200 else f"{lo}+"
                    count = hist.get(label, 0)
                    participant_rows.append({**base, "profile_type": "histogram", "value": label, "count": count, "percent": pct(count, to_int(summary["non_missing_count"]))})
        else:
            counter: Counter = acc["counter"]
            for value, count in counter.most_common(40):
                participant_rows.append({**base, "profile_type": "category_count", "value": value, "count": count, "percent": pct(count, row_count)})

    feature_rows = []
    for spec in feature_specs:
        acc = feature_acc.get(spec["profile_id"])
        if not acc:
            continue
        base = {
            "source_id": source_id,
            "source_label": SOURCE_LABELS.get(source_id, source_id),
            "theme": spec["theme"],
            "group_id": spec["group_id"],
            "feature_label": spec["feature_label"],
            "value_type": spec["value_type"],
            "column_count": spec["column_count"],
        }
        if spec["value_type"] in {"continuous numeric", "integer"}:
            feature_rows.append({**base, "profile_type": "numeric_summary", **numeric_summary_from_acc(acc), "top_values_json": "", "summary_basis": "exact moments; quantiles from deterministic sample"})
        else:
            counter: Counter = acc["counter"]
            top = [{"value": value, "count": count} for value, count in counter.most_common(10)]
            feature_rows.append(
                {
                    **base,
                    "profile_type": "categorical_summary",
                    "non_missing_count": to_int(acc.get("count")),
                    "missing_percent": pct(max(0, to_int(acc.get("denominator")) - to_int(acc.get("count"))), to_int(acc.get("denominator"))),
                    "observed_levels": len(counter),
                    "top_values_json": json.dumps(top, ensure_ascii=False),
                    "summary_basis": "exact counts unless unique_overflow is true",
                    "unique_overflow": acc.get("unique_overflow", False),
                }
            )

    disease_rows = []
    for spec in disease_specs:
        acc = disease_acc.get(spec["profile_id"])
        if not acc:
            continue
        counter: Counter = acc["counter"]
        top = [{"value": value, "count": count} for value, count in counter.most_common(10)]
        row = {
            "source_id": source_id,
            "source_label": SOURCE_LABELS.get(source_id, source_id),
            "profile_type": "target_summary",
            "target_family": spec["target_family"],
            "target_name": spec["target_name"],
            "target_type": spec["target_type"],
            "columns": ";".join(spec["columns"]),
            "case_only": spec.get("case_only", False),
            "participant_count": acc["participant_count"],
            "positive_count": acc["positive_count"],
            "percent": pct(acc["positive_count"] if spec.get("is_binary_flag") else acc["participant_count"], row_count),
            "top_values_json": json.dumps(top, ensure_ascii=False),
            "summary_json": "",
        }
        if spec.get("is_numeric") and acc.get("numeric"):
            row["summary_json"] = json.dumps(numeric_summary_from_acc(acc["numeric"]), ensure_ascii=False)
        disease_rows.append(row)
    for family, count in sorted(disease_family_counts.items()):
        if count <= 0:
            continue
        disease_rows.append(
            {
                "source_id": source_id,
                "source_label": SOURCE_LABELS.get(source_id, source_id),
                "profile_type": "disease_family_prevalence",
                "target_family": family,
                "target_name": f"Any positive {family} record",
                "target_type": "participants with at least one positive disease indicator",
                "columns": "",
                "case_only": False,
                "participant_count": count,
                "positive_count": count,
                "percent": pct(count, row_count),
                "top_values_json": "",
                "summary_json": "",
            }
        )
    if source_id == "metabolite_csv" and co_specs:
        for i, spec_a in enumerate(co_specs):
            for j, spec_b in enumerate(co_specs):
                if j < i:
                    continue
                count = int(co_matrix[i, j])
                if count == 0:
                    continue
                disease_rows.append(
                    {
                        "source_id": source_id,
                        "source_label": SOURCE_LABELS.get(source_id, source_id),
                        "profile_type": "cooccurrence",
                        "target_family": "Disease co-occurrence",
                        "target_name": f'{spec_a["target_name"]} + {spec_b["target_name"]}',
                        "target_type": "curated status flag pair",
                        "columns": f'{spec_a["columns"][0]};{spec_b["columns"][0]}',
                        "case_only": False,
                        "participant_count": count,
                        "positive_count": count,
                        "percent": pct(count, row_count),
                        "top_values_json": "",
                        "summary_json": "",
                    }
                )

    repeat_rows = []
    for spec in repeat_specs:
        acc = repeat_acc.get(spec["profile_id"], {})
        n = to_int(acc.get("n_subjects"))
        if n == 0:
            continue
        mean = float(acc["mean_sum"]) / n
        between_var = max(0.0, float(acc["mean_sumsq"]) / n - mean * mean)
        within_var = float(acc["within_ss"]) / max(1, to_int(acc["within_df"]))
        icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else math.nan
        between_cv = math.sqrt(between_var) / abs(mean) if mean else math.nan
        within_cv = float(acc["cv_sum"]) / to_int(acc["cv_count"]) if to_int(acc["cv_count"]) else math.nan
        pair_n = to_int(acc.get("pair_count"))
        corr = math.nan
        if pair_n > 1:
            sx = float(acc["sum_x"])
            sy = float(acc["sum_y"])
            denom = math.sqrt((pair_n * float(acc["sum_x2"]) - sx * sx) * (pair_n * float(acc["sum_y2"]) - sy * sy))
            if denom > 0:
                corr = (pair_n * float(acc["sum_xy"]) - sx * sy) / denom
        repeat_rows.append(
            {
                "source_id": source_id,
                "source_label": SOURCE_LABELS.get(source_id, source_id),
                "theme": spec["theme"],
                "group_id": spec["group_id"],
                "feature_label": spec["feature_label"],
                "visit_pair": acc.get("visit_pair", ""),
                "participant_count_ge2_visits": n,
                "observation_count": acc["n_obs"],
                "within_person_cv": fmt_float(within_cv),
                "between_person_cv": fmt_float(between_cv),
                "icc_approx": fmt_float(icc),
                "visit_pair_count": pair_n,
                "visit_pair_correlation": fmt_float(corr),
            }
        )

    missing_rows = []
    for (group_variable, group_value, theme), counts in sorted(missing_counts.items()):
        participants = counts["participants"]
        with_data = counts["with_data"]
        missing_rows.append(
            {
                "source_id": source_id,
                "source_label": SOURCE_LABELS.get(source_id, source_id),
                "profile_type": "theme_missingness",
                "group_variable": group_variable,
                "group_value": group_value,
                "theme": theme,
                "participant_count": participants,
                "with_data_count": with_data,
                "missing_count": max(0, participants - with_data),
                "present_percent": pct(with_data, participants),
            }
        )
    for burden, count in sorted(burden_counts.items()):
        missing_rows.append(
            {
                "source_id": source_id,
                "source_label": SOURCE_LABELS.get(source_id, source_id),
                "profile_type": "participant_theme_burden",
                "group_variable": "theme_count",
                "group_value": str(burden),
                "theme": "All profiled themes",
                "participant_count": count,
                "with_data_count": count,
                "missing_count": "",
                "present_percent": pct(count, row_count),
            }
        )
    for (group_variable, group_value, theme), counts in sorted(disease_missing_counts.items()):
        participants = counts["participants"]
        with_data = counts["with_data"]
        missing_rows.append(
            {
                "source_id": source_id,
                "source_label": SOURCE_LABELS.get(source_id, source_id),
                "profile_type": "missingness_by_disease_status",
                "group_variable": group_variable,
                "group_value": group_value,
                "theme": theme,
                "participant_count": participants,
                "with_data_count": with_data,
                "missing_count": max(0, participants - with_data),
                "present_percent": pct(with_data, participants),
            }
        )

    return participant_rows, disease_rows, feature_rows, repeat_rows, missing_rows


def build_basic_profiles(
    groups: list[dict[str, object]],
    columns: list[dict[str, str]],
    inventory: list[dict[str, str]],
) -> dict[str, list[dict[str, object]]]:
    specs = {
        "participant": participant_specs(columns),
        "feature": build_feature_profile_specs(groups),
        "repeatability": build_repeatability_specs(groups),
        "disease": build_disease_profile_specs(groups, columns),
    }
    fingerprint = basic_profile_fingerprint(specs)
    if BASIC_PROFILE_CACHE.exists():
        try:
            cached = json.loads(BASIC_PROFILE_CACHE.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                profiles = normalize_basic_profiles(cached["profiles"])
                return attach_disease_family_prevalence(profiles, specs["disease"], inventory)
            if can_reuse_basic_profile_cache(cached, fingerprint):
                print("Reusing cached basic profiles after semantic group remapping", flush=True)
                profiles = normalize_basic_profiles(remap_cached_basic_profiles(cached["profiles"], groups))
                return attach_disease_family_prevalence(profiles, specs["disease"], inventory)
        except json.JSONDecodeError:
            pass

    source_rows = {row["source_id"]: to_int(row["row_count"]) for row in inventory}
    profiles = {
        "participant_groups": [],
        "disease_targets": [],
        "feature_distributions": [],
        "repeatability_cv_icc": [],
        "missingness_patterns": [],
    }
    for source_id, path in SOURCE_FILES.items():
        if not path.exists():
            continue
        print(f"Basic profiling {source_id}: {path}", flush=True)
        part_rows, disease_rows, feature_rows, repeat_rows, missing_rows = scan_basic_profile_source(
            source_id,
            path,
            source_rows.get(source_id, 0),
            columns,
            source_specs(specs["participant"], source_id),
            source_specs(specs["feature"], source_id),
            source_specs(specs["repeatability"], source_id),
            source_specs(specs["disease"], source_id),
        )
        profiles["participant_groups"].extend(part_rows)
        profiles["disease_targets"].extend(disease_rows)
        profiles["feature_distributions"].extend(feature_rows)
        profiles["repeatability_cv_icc"].extend(repeat_rows)
        profiles["missingness_patterns"].extend(missing_rows)
        print(
            f"Finished basic profiling {source_id}: "
            f"{len(part_rows):,} participant rows, {len(disease_rows):,} disease rows, "
            f"{len(feature_rows):,} feature rows, {len(repeat_rows):,} repeatability rows",
            flush=True,
        )

    profiles = normalize_basic_profiles(profiles)
    BASIC_PROFILE_CACHE.write_text(
        json.dumps({"fingerprint": fingerprint, "profiles": profiles}, ensure_ascii=False),
        encoding="utf-8",
    )
    return profiles


def svg_text(x, y, text, size=13, fill=None, weight="400", anchor="start"):
    fill = fill or COLORS["text"]
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(str(text))}</text>"
    )


def svg_rect(x, y, w, h, fill, stroke="none", rx=0):
    return f'<rect x="{x}" y="{y}" width="{max(0, w)}" height="{max(0, h)}" rx="{rx}" fill="{fill}" stroke="{stroke}" />'


def svg_line(x1, y1, x2, y2, stroke=None, width=1):
    stroke = stroke or COLORS["line"]
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" />'


def save_svg(path: Path, width: int, height: int, parts: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                svg_rect(0, 0, width, height, COLORS["paper"]),
                *parts,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def build_dataset_tree_svg(inventory, groups):
    width, height = 1500, 760
    parts = [
        svg_text(36, 42, "Retrieved UKB Dataset Structure", 26, weight="700"),
        svg_text(36, 70, "Dataset-first view using biological domains and acquisition types for report navigation.", 14, COLORS["muted"]),
    ]
    source_rows = {r["source_id"]: to_int(r["row_count"]) for r in inventory}
    by_source_type: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"features": 0, "participants": 0}))
    for group in groups:
        source_id = str(group["source_id"])
        label = f'{group.get("report_domain", "")} / {group.get("acquisition_type", "")}'
        by_source_type[source_id][label]["features"] += to_int(group["feature_count"])
        by_source_type[source_id][label]["participants"] = max(
            by_source_type[source_id][label]["participants"],
            to_int(group["participant_count_proxy"]),
        )
    x_positions = [44, 412, 780, 1148]
    y0 = 112
    for idx, source_id in enumerate(SOURCE_LABELS):
        x = x_positions[idx]
        parts.append(svg_rect(x, y0, 308, 46, COLORS["blue"], rx=6))
        parts.append(svg_text(x + 14, y0 + 28, SOURCE_LABELS[source_id], 16, "#FFFFFF", "700"))
        parts.append(svg_text(x + 14, y0 + 70, f"Participant records: {fmt_int(source_rows.get(source_id, 0))}", 13, COLORS["muted"]))
        y = y0 + 96
        rows = sorted(by_source_type.get(source_id, {}).items(), key=lambda item: item[1]["features"], reverse=True)[:10]
        for label, counts in rows:
            label = f'{label} ({fmt_int(counts["features"])})'
            parts.append(svg_rect(x, y, 308, 28, COLORS["soft"], COLORS["line"], rx=4))
            parts.append(svg_text(x + 10, y + 19, short(label, 38), 12))
            parts.append(svg_text(x + 298, y + 19, fmt_int(counts["participants"]), 12, COLORS["muted"], anchor="end"))
            y += 34
    save_svg(SUPPORT_DIR / "dataset_tree_overview.svg", width, height, parts)


def build_storage_pattern_svg():
    width, height = 1500, 700
    parts = [
        svg_text(36, 42, "Raw UKB Column Decoder", 26, weight="700"),
        svg_text(36, 70, "UKB column names use field-instance.array. The array part is shown as repeated entries in the report view.", 14, COLORS["muted"]),
    ]

    # Column-name decoder.
    parts.append(svg_rect(44, 106, 1412, 148, COLORS["soft"], COLORS["line"], rx=8))
    parts.append(svg_text(68, 138, "Example: 50-1.0", 18, weight="700"))
    parts.append(svg_text(68, 164, "Standing height measured at visit/instance 1; repeated-entry position 0.", 14, COLORS["muted"]))
    token_boxes = [
        (82, "50", "Field ID", "official UKB field"),
        (362, "1", "Instance", "assessment visit/timepoint"),
        (642, "0", "Repeated entry", "item within field"),
        (922, "cell value", "Participant value", "one value per record"),
    ]
    for x, token, title, desc in token_boxes:
        parts.append(svg_rect(x, 178, 238, 54, COLORS["paper"], COLORS["line"], rx=6))
        parts.append(svg_text(x + 14, 201, token, 17, COLORS["blue"], "700"))
        parts.append(svg_text(x + 14, 222, f"{title}: {desc}", 11, COLORS["muted"]))
    for x in [320, 600, 880]:
        parts.append(svg_line(x, 205, x + 38, 205, COLORS["blue"], 2))
        parts.append(svg_text(x + 16, 210, ">", 18, COLORS["blue"], "700"))
    parts.append(svg_rect(1192, 178, 222, 54, COLORS["paper"], COLORS["line"], rx=6))
    parts.append(svg_text(1206, 201, "semantic feature", 17, COLORS["green"], "700"))
    parts.append(svg_text(1206, 222, "Standing height", 11, COLORS["muted"]))

    examples = [
        {
            "layout": "Repeated visits",
            "raw": "50-0.0, 50-1.0",
            "meaning": "same measurement concept, different visits",
        },
        {
            "layout": "Paired repeated entries",
            "raw": "41270-0.189 + 41280-0.189",
            "meaning": "one diagnosis event: code + first date",
        },
        {
            "layout": "Missing/failure notes",
            "raw": "30515-0.0",
            "meaning": "result flag, not the numeric creatinine measurement",
        },
        {
            "layout": "File handles vs traits",
            "raw": "21015-0.0 vs MRI volume columns",
            "meaning": "image asset reference vs tabular derived feature",
        },
    ]
    y = 294
    for example in examples:
        parts.append(svg_rect(44, y, 1412, 72, COLORS["paper"], COLORS["line"], rx=8))
        parts.append(svg_text(68, y + 28, example["layout"], 17, weight="700"))
        parts.append(svg_text(360, y + 28, example["raw"], 14, COLORS["blue"], "700"))
        parts.append(svg_text(760, y + 28, example["meaning"], 14, COLORS["green"], "700"))
        y += 88

    parts.append(svg_rect(44, 650, 1412, 1, COLORS["line"]))
    parts.append(svg_text(68, 676, "Exhaustive mapping: Dataset Structure Tree and semantic_feature_groups.csv.", 13, COLORS["muted"]))
    save_svg(SUPPORT_DIR / "storage_pattern_explanation.svg", width, height, parts)


def build_coverage_svg(modality_counts):
    sources = list(SOURCE_LABELS)
    modalities = [
        "metabolomics_nmr",
        "clinical_biomarker_blood_urine",
        "brain_mri_imaging",
        "dxa_body_composition_imaging",
        "eye_imaging",
        "cognitive_assessment",
        "physical_measure",
        "lifestyle_medication_diet",
        "derived_or_linked_outcome",
        "linked_health_outcome",
    ]
    lookup = {(r["source_id"], r["modality_category"]): r for r in modality_counts}
    width, height = 1500, 720
    left, top, row_h, col_w = 310, 116, 46, 275
    parts = [
        svg_text(36, 42, "Coverage by Dataset and Modality", 26, weight="700"),
        svg_text(36, 70, "Cell labels show participants with any data / feature columns.", 14, COLORS["muted"]),
    ]
    for j, sid in enumerate(sources):
        parts.append(svg_text(left + j * col_w + col_w / 2, 100, SOURCE_LABELS[sid], 13, weight="700", anchor="middle"))
    max_n = max(to_int(r["participant_count"]) for r in modality_counts)
    for i, modality in enumerate(modalities):
        y = top + i * row_h
        parts.append(svg_text(36, y + 28, MODALITY_LABELS[modality], 13))
        for j, sid in enumerate(sources):
            x = left + j * col_w
            row = lookup.get((sid, modality))
            if not row:
                fill = "#F2F4F7"
                label = "not present"
            else:
                n = to_int(row["participant_count"])
                alpha = 0.15 + 0.75 * (math.log1p(n) / math.log1p(max_n))
                fill = f"rgba(47,111,159,{alpha:.2f})"
                label = f'{fmt_int(n)} / {fmt_int(row["feature_count"])}'
            parts.append(svg_rect(x, y, col_w - 14, 34, fill, COLORS["line"], rx=4))
            parts.append(svg_text(x + (col_w - 14) / 2, y + 22, label, 12, anchor="middle"))
    save_svg(SUPPORT_DIR / "modality_coverage_figure.svg", width, height, parts)


def requested_data_status_rows() -> list[dict[str, str]]:
    rows = [
        {"requested_area": "Imaging data", "requested_type": "Brain MRI", "availability_status": "Partial", "notes": "Derived scalar MRI traits and image file identifiers are present; actual image assets are not retrieved."},
        {"requested_area": "Imaging data", "requested_type": "Heart MRI", "availability_status": "Missing", "notes": "No cardiac MRI fields or assets detected locally."},
        {"requested_area": "Imaging data", "requested_type": "Abdomen MRI", "availability_status": "Missing", "notes": "No abdominal MRI fields or assets detected locally."},
        {"requested_area": "Imaging data", "requested_type": "DXA/DEXA", "availability_status": "Available", "notes": "DXA/body-composition derived traits are present in ukb43216."},
        {"requested_area": "Imaging data", "requested_type": "Carotid ultrasound", "availability_status": "Missing", "notes": "No carotid ultrasound fields detected locally."},
        {"requested_area": "Imaging data", "requested_type": "Retina/OCT", "availability_status": "Partial", "notes": "Fundus/OCT file IDs and acquisition metadata are present; actual image assets are not retrieved."},
        {"requested_area": "Biomarker data", "requested_type": "Proteomics", "availability_status": "Missing", "notes": "No Olink/SomaScan or equivalent whole proteomics panel detected locally."},
        {"requested_area": "Biomarker data", "requested_type": "Biochemistry biomarkers", "availability_status": "Available", "notes": "Routine blood/urine clinical biomarker fields are present."},
        {"requested_area": "Biomarker data", "requested_type": "Metabolomic biomarkers", "availability_status": "Available", "notes": "NMR metabolomics fields and z-scored metabolite columns are present in METABOLITE.csv."},
        {"requested_area": "Biomarker data", "requested_type": "Blood count", "availability_status": "Partial", "notes": "Some haematology-style fields exist, but this is not a dedicated complete blood-count panel in the current layer."},
        {"requested_area": "Biomarker data", "requested_type": "Infectious disease markers", "availability_status": "Missing", "notes": "No infectious disease marker panel detected locally."},
        {"requested_area": "Genetic data", "requested_type": "Genotyping / imputation / WES / WGS / telomere", "availability_status": "Missing", "notes": "No genome-wide genotype, imputation, exome, genome, or telomere files detected locally."},
        {"requested_area": "Health records data", "requested_type": "Hospital inpatient / cancer / death / first occurrences", "availability_status": "Partial", "notes": "Hospital ICD/OPCS repeated-entry summaries and selected linked endpoints are present; full linked-record tables are not all present."},
        {"requested_area": "Health records data", "requested_type": "Primary care", "availability_status": "Missing", "notes": "No primary-care table detected locally."},
        {"requested_area": "Questionnaire data", "requested_type": "Diet / cognitive / mental health / health wellbeing", "availability_status": "Partial", "notes": "Diet, cognitive, lifestyle, and mental-health-related fields are present, but not as every requested questionnaire module."},
        {"requested_area": "Physical measurements data", "requested_type": "General physical / vision / memory", "availability_status": "Available", "notes": "Physical measures, eye/vision, and cognitive/memory fields are present."},
        {"requested_area": "Physical measurements data", "requested_type": "Arterial stiffness / hearing / fitness / activity monitor / ECG", "availability_status": "Partial", "notes": "Some activity/location metadata exists; the complete dedicated modalities are not clearly present."},
        {"requested_area": "Demographic and lifestyle data", "requested_type": "Sociodemographics / medical history / lifestyle / family / psychosocial", "availability_status": "Partial", "notes": "Demographic, lifestyle, medication, and self-reported history fields are present, but not necessarily every requested submodule."},
        {"requested_area": "Environmental data", "requested_type": "Air / noise / greenspace / built environment / water mineral", "availability_status": "Partial", "notes": "Some location/deprivation-style metadata exists; the full environmental exposure suite is not detected."},
    ]
    return sorted(rows, key=lambda r: (AVAILABILITY_ORDER[r["availability_status"]], r["requested_area"], r["requested_type"]))


def target_family(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["mri", "brain volume", "grey matter", "white matter", "ventricular"]):
        return "Brain MRI phenotypes"
    if any(k in lower for k in ["dxa", "bone", "bmd", "bmc", "body composition", "fat", "lean mass"]):
        return "DXA/body-composition phenotypes"
    if any(k in lower for k in ["cognitive", "memory", "reaction time"]):
        return "Cognitive phenotypes"
    return disease_family(text)


def target_type_from_role(row: dict[str, str]) -> str:
    role = row["feature_role"]
    name = row["column_name"]
    if role == "incident_outcome_flag" or name.startswith("incident_"):
        return "incident binary outcome"
    if role == "prevalent_or_prior_outcome_flag" or name.startswith("prior_"):
        return "prevalent/prior binary outcome"
    if role == "outcome_date" or name.endswith("_date_2104"):
        return "event/date outcome"
    if role == "outcome_flag_or_status":
        return "status/binary outcome"
    return "target candidate"


def target_catalog_rows(columns: list[dict[str, str]], inventory: list[dict[str, str]]) -> list[dict[str, object]]:
    source_rows = {r["source_id"]: to_int(r["row_count"]) for r in inventory}
    rows: list[dict[str, object]] = []

    for row in columns:
        if row["source_id"] != "metabolite_csv":
            continue
        name = row["column_name"]
        role = row["feature_role"]
        is_outcome = (
            "outcome" in role
            or bool(re.match(r"^(incident|prior)_", name))
            or name.endswith("_date_2104")
        )
        if not is_outcome:
            continue
        participant_count = to_int(row["non_missing_count"]) or source_rows.get(row["source_id"], 0)
        rows.append(
            {
                "target_id": f'{row["source_id"]}:{name}',
                "target_name": name,
                "target_family": target_family(name),
                "target_type": target_type_from_role(row),
                "source_id": row["source_id"],
                "source_label": SOURCE_LABELS.get(row["source_id"], row["source_id"]),
                "feature_count": 1,
                "participant_count_proxy": participant_count,
                "coverage_count_type": "source row count used when outcome non-missing count was not profiled",
                "exact_target_columns": f'{row["source_id"]}:{name}',
                "exact_target_columns_sample": f'{row["source_id"]}:{name}',
                "notes": "Exact curated disease/status/date column from METABOLITE.csv.",
            }
        )

    linked_by_source: dict[str, dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in columns:
        if row["modality_category"] != "linked_health_outcome":
            continue
        desc = row["description"]
        lower = desc.lower()
        endpoint = ""
        attr = ""
        if lower.startswith("date of "):
            endpoint = re.sub(r"\s+report$", "", desc[8:], flags=re.IGNORECASE)
            attr = "date"
        elif lower.startswith("source of "):
            endpoint = re.sub(r"\s+report$", "", desc[10:], flags=re.IGNORECASE)
            attr = "source"
        if endpoint:
            linked_by_source[row["source_id"]][endpoint][attr] = row

    for source_id, endpoints in linked_by_source.items():
        for endpoint, attrs in endpoints.items():
            if "date" not in attrs:
                continue
            date_row = attrs["date"]
            source_row = attrs.get("source")
            raw_cols = [f'{source_id}:{date_row["column_name"]}']
            if source_row:
                raw_cols.append(f'{source_id}:{source_row["column_name"]}')
            participant_count = max(to_int(date_row.get("non_missing_count")), to_int(source_row.get("non_missing_count") if source_row else 0))
            rows.append(
                {
                    "target_id": f'{source_id}:linked_endpoint:{slug(endpoint)}',
                    "target_name": f"Date/source of {endpoint} report" if source_row else f"Date of {endpoint} report",
                    "target_family": target_family(endpoint),
                    "target_type": "linked endpoint date/source",
                    "source_id": source_id,
                    "source_label": SOURCE_LABELS.get(source_id, source_id),
                    "feature_count": len(raw_cols),
                    "participant_count_proxy": participant_count,
                    "coverage_count_type": "max non-missing count across endpoint date/source columns",
                    "exact_target_columns": ";".join(raw_cols),
                    "exact_target_columns_sample": "; ".join(raw_cols),
                    "notes": "UKB linked endpoint fields. Use date/source as target-definition fields, not predictors.",
                }
            )

    aggregate_specs = [
        {
            "target_name": "Brain MRI derived scalar phenotypes",
            "target_family": "Brain MRI phenotypes",
            "target_type": "continuous imaging phenotype group",
            "source_id": "eye_brain_ukb42577",
            "predicate": lambda r: r["source_id"] == "eye_brain_ukb42577"
            and r["modality_category"] == "brain_mri_imaging"
            and not any(k in r["description"].lower() for k in ["dicom", "nifti", "images"]),
            "notes": "Group of exact brain MRI scalar traits; use individual columns as continuous targets after selecting an endpoint.",
        },
        {
            "target_name": "DXA/body-composition derived phenotypes",
            "target_family": "DXA/body-composition phenotypes",
            "target_type": "continuous imaging phenotype group",
            "source_id": "eye_brain_ukb43216",
            "predicate": lambda r: r["source_id"] == "eye_brain_ukb43216"
            and r["modality_category"] == "dxa_body_composition_imaging",
            "notes": "Group of exact DXA/body-composition traits; use individual columns as continuous targets.",
        },
        {
            "target_name": "Cognitive task phenotypes",
            "target_family": "Cognitive phenotypes",
            "target_type": "continuous/categorical assessment phenotype group",
            "source_id": "eye_brain_ukb42577",
            "predicate": lambda r: r["source_id"] == "eye_brain_ukb42577"
            and r["modality_category"] == "cognitive_assessment",
            "notes": "Group of exact cognitive assessment fields; use individual columns as targets or covariates depending on the model.",
        },
        {
            "target_name": "Kidney-function clinical biomarker phenotypes",
            "target_family": "CKD / kidney function",
            "target_type": "continuous clinical biomarker phenotype group",
            "source_id": "eye_kidney_ukb42408",
            "predicate": lambda r: r["source_id"] == "eye_kidney_ukb42408"
            and r["modality_category"] == "clinical_biomarker_blood_urine"
            and any(k in r["description"].lower() for k in ["creatinine", "cystatin", "microalbumin", "renal"]),
            "notes": "Group of exact kidney-function laboratory traits; can be continuous targets or predictors depending on analysis design.",
        },
    ]
    for spec in aggregate_specs:
        matched = sorted([r for r in columns if spec["predicate"](r)], key=lambda r: to_int(r["column_index"]))
        if not matched:
            continue
        raw_cols = [f'{r["source_id"]}:{r["column_name"]}' for r in matched]
        participant_count = max(to_int(r["non_missing_count"]) for r in matched)
        rows.append(
            {
                "target_id": f'{spec["source_id"]}:aggregate_target:{slug(spec["target_name"])}',
                "target_name": spec["target_name"],
                "target_family": spec["target_family"],
                "target_type": spec["target_type"],
                "source_id": spec["source_id"],
                "source_label": SOURCE_LABELS.get(spec["source_id"], spec["source_id"]),
                "feature_count": len(raw_cols),
                "participant_count_proxy": participant_count,
                "coverage_count_type": "max non-missing count across phenotype columns",
                "exact_target_columns": ";".join(raw_cols),
                "exact_target_columns_sample": "; ".join(raw_cols[:40]) + ("; ..." if len(raw_cols) > 40 else ""),
                "notes": spec["notes"],
            }
        )

    return sorted(rows, key=lambda r: (str(r["target_family"]), str(r["target_name"])))


def target_namespace(source_id: str) -> str:
    if source_id in {"eye_kidney_ukb42408", "metabolite_csv"}:
        return "CKD/eid_ckd namespace, bridgeable to eye-brain through bridge_brain.dta"
    if source_id in {"eye_brain_ukb42577", "eye_brain_ukb43216"}:
        return "eye-brain namespace, bridgeable to eid_ckd through bridge_brain.dta"
    return "unknown ID namespace"


def target_predictor_availability_rows(target_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    predictors = [
        "Conventional covariates / metadata",
        "Routine clinical biomarkers",
        "NMR metabolomics",
        "Brain MRI scalar traits",
        "DXA/body-composition traits",
        "Eye imaging IDs/acquisition metadata",
        "Cognitive assessment features",
        "Proteomics panel",
        "Genetics / WES / WGS",
        "Actual imaging assets",
    ]
    rows: list[dict[str, str]] = []
    for target in target_rows:
        source_id = str(target["source_id"])
        namespace = target_namespace(source_id)
        for predictor in predictors:
            status = "Partial"
            reason = "Present locally, but exact participant/timepoint/temporality curation is still required."
            if predictor == "Proteomics panel":
                status = "Missing"
                reason = "Derived from requested-data status: no whole proteomics panel was retrieved locally."
            elif predictor == "Genetics / WES / WGS":
                status = "Missing"
                reason = "Derived from requested-data status: no genotype, imputation, exome, genome, or telomere files were retrieved locally."
            elif predictor == "Actual imaging assets":
                status = "Missing"
                reason = "Derived from requested-data status: file IDs are present for some imaging, but the image files themselves are not retrieved."
            elif predictor == "Conventional covariates / metadata":
                status = "Available"
                reason = f"Demographic, visit, physical, lifestyle, and questionnaire metadata are present in the {namespace}."
            elif predictor == "Routine clinical biomarkers":
                status = "Available"
                reason = f"Routine blood/urine biomarker fields are present and are alignable within the {namespace}."
            elif predictor == "NMR metabolomics":
                status = "Available"
                reason = "METABOLITE.csv aligns with eye-kidney by eid_ckd and with eye-brain through bridge_brain.dta."
            elif predictor == "Brain MRI scalar traits":
                status = "Available"
                reason = "Brain MRI scalar traits are in ukb42577 and are bridge-alignable to eid_ckd through bridge_brain.dta."
            elif predictor == "DXA/body-composition traits":
                status = "Available"
                reason = "DXA/body-composition traits are in ukb43216 and are bridge-alignable to eid_ckd through bridge_brain.dta."
            elif predictor == "Eye imaging IDs/acquisition metadata":
                status = "Available"
                reason = "Eye imaging file identifiers and acquisition metadata exist in the retrieved application returns; this does not include image pixels."
            elif predictor == "Cognitive assessment features":
                status = "Available"
                reason = "Cognitive assessment fields are in the eye-brain returns and are bridge-alignable to eid_ckd through bridge_brain.dta."
            rows.append(
                {
                    "target_id": str(target["target_id"]),
                    "target_name": str(target["target_name"]),
                    "target_family": str(target["target_family"]),
                    "predictor": predictor,
                    "availability_status": status,
                    "reason": reason,
                }
            )
    return rows


def build_prediction_target_overview_svg(target_rows):
    by_family = Counter(r["target_family"] for r in target_rows)
    by_type = Counter(r["target_type"] for r in target_rows)
    width, height = 1500, 740
    parts = [
        svg_text(36, 42, "Prediction Target Catalog", 26, weight="700"),
        svg_text(36, 70, "Exact target bubbles are interactive in the HTML; this static overview shows target families and phenotype types.", 14, COLORS["muted"]),
    ]
    x0, y0 = 48, 126
    max_count = max(by_family.values()) if by_family else 1
    for i, (family, count) in enumerate(sorted(by_family.items(), key=lambda item: (-item[1], item[0]))):
        col = i % 3
        row = i // 3
        x = x0 + col * 470
        y = y0 + row * 72
        parts.append(svg_rect(x, y, 430, 44, COLORS["soft"], COLORS["line"], rx=6))
        w = int(250 * count / max_count)
        parts.append(svg_rect(x, y, w, 44, COLORS["blue_light"], rx=6))
        parts.append(svg_text(x + 14, y + 27, short(family, 44), 13, weight="700"))
        parts.append(svg_text(x + 410, y + 27, fmt_int(count), 13, COLORS["muted"], "700", anchor="end"))
    y = 520
    parts.append(svg_text(48, y, "Target phenotype types", 18, weight="700"))
    y += 34
    for target_type, count in sorted(by_type.items(), key=lambda item: (-item[1], item[0])):
        parts.append(svg_text(70, y, f"{target_type}: {fmt_int(count)}", 13))
        y += 24
    save_svg(SUPPORT_DIR / "prediction_target_overview.svg", width, height, parts)
    save_svg(SUPPORT_DIR / "prediction_feasibility_matrix.svg", width, height, parts)


def build_missing_requested_svg(rows):
    width, height = 1500, 760
    parts = [
        svg_text(36, 42, "Requested Data Type Status", 26, weight="700"),
        svg_text(36, 70, "Compared with the UKB proposal. Status terms are Available, Partial, or Missing.", 14, COLORS["muted"]),
        svg_text(40, 108, "Requested area", 13, weight="700"),
        svg_text(260, 108, "Requested type", 13, weight="700"),
        svg_text(880, 108, "Availability", 13, weight="700"),
    ]
    y = 130
    for i, row in enumerate(rows):
        if i % 2 == 0:
            parts.append(svg_rect(32, y - 20, 1430, 36, COLORS["soft"]))
        status = row["availability_status"]
        color = COLORS["green"] if status == "Available" else COLORS["red"] if status == "Missing" else COLORS["amber"]
        parts.append(svg_text(40, y, row["requested_area"], 12))
        parts.append(svg_text(260, y, short(row["requested_type"], 76), 12))
        parts.append(svg_text(880, y, status, 12, color, "700"))
        y += 38
    save_svg(SUPPORT_DIR / "missing_requested_data_types.svg", width, height, parts)


def id_namespace_rows(dataset_alignment: list[dict[str, str]], bridge_alignment_summary: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    bridge_alignment_summary = bridge_alignment_summary or []
    bridge_counts = {row.get("bridge_file", ""): row.get("row_count", "") for row in bridge_alignment_summary}
    return [
        {
            "id_namespace": "Eye-kidney ukb42408:eid",
            "role": "CKD/eid_ckd namespace",
            "notes": "Direct key for eye-kidney fields; matches METABOLITE.csv:eid_ckd and bridge_brain.dta:eid_ckd.",
        },
        {
            "id_namespace": "METABOLITE.csv:eid_ckd",
            "role": "Metabolite CKD key",
            "notes": "Use this ID column for kidney/metabolite/brain fusion; it maps to eye-brain through bridge_brain.dta.",
        },
        {
            "id_namespace": "Eye-brain ukb42577/ukb43216:eid",
            "role": "Eye-brain namespace",
            "notes": "Same eye-brain namespace across ukb42577 and ukb43216; bridge_brain.dta maps it to eid_ckd.",
        },
        {
            "id_namespace": "bridge_brain.dta:eid_brain -> eid_ckd",
            "role": "Kidney-brain crosswalk",
            "notes": f"One-to-one bridge with {bridge_counts.get('bridge_brain.dta', '502493')} rows; this resolves local kidney-brain ID alignment.",
        },
        {
            "id_namespace": "METABOLITE.csv:eid_ageing",
            "role": "Alternative metabolite namespace",
            "notes": "Same METABOLITE.csv rows, different encoded ID values; bridge_ageing.dta maps it to eid_ckd.",
        },
        {
            "id_namespace": "bridge_ageing.dta:eid_ageing -> eid_ckd",
            "role": "Ageing-to-CKD crosswalk",
            "notes": f"One-to-one bridge with {bridge_counts.get('bridge_ageing.dta', '502493')} rows; useful for interpreting METABOLITE.csv:eid_ageing.",
        },
        {
            "id_namespace": "bridge_heart.dta:eid_heart -> eid_ckd",
            "role": "Future heart-data crosswalk",
            "notes": f"One-to-one bridge with {bridge_counts.get('bridge_heart.dta', '502493')} rows; no local heart feature table is currently profiled.",
        },
    ]


def build_tree_summary(groups):
    by_source = Counter(g["source_label"] for g in groups)
    by_role = Counter(g["prediction_role"] for g in groups)
    by_storage = Counter(g["storage_pattern"] for g in groups)
    by_semantic_function = Counter(g["semantic_function"] for g in groups)
    by_value_type = Counter(g["value_type"] for g in groups)
    by_report_domain = Counter(g.get("report_domain", "") for g in groups)
    by_acquisition_type = Counter(g.get("acquisition_type", "") for g in groups)
    return {
        "semantic_group_count": len(groups),
        "feature_count": sum(to_int(g["feature_count"]) for g in groups),
        "by_source": dict(by_source),
        "by_report_domain": dict(by_report_domain),
        "by_acquisition_type": dict(by_acquisition_type),
        "by_prediction_role": dict(by_role),
        "by_storage_pattern": dict(by_storage),
        "by_semantic_function": dict(by_semantic_function),
        "by_value_type": dict(by_value_type),
    }


def render_table(rows, fields, max_rows=None, labels=None):
    labels = labels or {}
    body = []
    shown = rows[:max_rows] if max_rows else rows
    body.append("<table>")
    body.append(
        "<thead><tr>"
        + "".join(f"<th>{html.escape(labels.get(f, f.replace('_', ' ').title()))}</th>" for f in fields)
        + "</tr></thead>"
    )
    body.append("<tbody>")
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(f, '')))}</td>" for f in fields) + "</tr>")
    body.append("</tbody></table>")
    return "\n".join(body)


def embed_svg(path: Path) -> str:
    if not path.exists():
        return f"<p>Missing figure: {html.escape(path.name)}</p>"
    return path.read_text(encoding="utf-8")


def build_html(
    payload,
    requested_rows,
    target_rows,
    target_predictor_rows,
    namespace_rows,
    modality_counts,
    dataset_alignment,
    bridge_alignment_summary,
    alignment_set_summary,
):
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    requested_table = render_table(
        requested_rows,
        ["availability_status", "requested_area", "requested_type"],
        labels={"availability_status": "Availability", "requested_area": "Requested area", "requested_type": "Requested data type"},
    )
    figures = {
        "dataset_tree": embed_svg(SUPPORT_DIR / "dataset_tree_overview.svg"),
        "upset_alignment": embed_svg(PLOTS_DIR / "upset_dataset_id_alignment.svg"),
    }

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UKB Dataset Interactive Visualization</title>
<style>
:root {{
  --text:#1F2933; --muted:#637381; --line:#D9E2EC; --soft:#F6F8FA;
  --blue:#2F6F9F; --green:#2F855A; --amber:#B7791F; --red:#C53030;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:var(--text); background:#fff; }}
header {{ padding:28px 36px 18px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:28px; }}
h2 {{ margin:28px 0 12px; font-size:22px; }}
h3 {{ margin:18px 0 8px; font-size:17px; }}
p {{ line-height:1.45; }}
.muted {{ color:var(--muted); }}
.tabs {{ display:flex; gap:8px; flex-wrap:wrap; padding:12px 36px; border-bottom:1px solid var(--line); position:sticky; top:0; background:#fff; z-index:3; }}
.tabs button {{ border:1px solid var(--line); background:#fff; padding:8px 12px; border-radius:6px; cursor:pointer; font-weight:600; }}
.tabs button.active {{ background:var(--blue); color:#fff; border-color:var(--blue); }}
.subtabs {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0 16px; }}
.subtabs button {{ border:1px solid var(--line); background:#fff; padding:7px 10px; border-radius:6px; cursor:pointer; font-weight:600; }}
.subtabs button.active {{ background:var(--blue); color:#fff; border-color:var(--blue); }}
main {{ padding:0 36px 40px; }}
.view {{ display:none; }}
.view.active {{ display:block; }}
.grid {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin:18px 0; }}
.metric {{ border:1px solid var(--line); border-radius:8px; padding:14px; background:var(--soft); }}
.metric .value {{ font-size:24px; font-weight:700; margin-bottom:4px; }}
.controls {{ display:grid; grid-template-columns:2fr repeat(2, minmax(150px, 1fr)) minmax(180px, auto); gap:10px; margin:16px 0; align-items:center; }}
.check-control {{ display:flex; align-items:center; gap:7px; font-size:13px; color:var(--text); }}
.check-control input {{ width:auto; }}
input, select {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:9px; font-size:13px; }}
.tree-wrap {{ display:grid; grid-template-columns:minmax(0, 1fr) 420px; gap:18px; align-items:start; }}
.tree {{ border:1px solid var(--line); border-radius:8px; padding:12px; max-height:760px; overflow:auto; }}
details {{ margin:4px 0 4px 14px; }}
summary {{ cursor:pointer; padding:4px; border-radius:4px; }}
summary:hover {{ background:var(--soft); }}
.node-meta {{ color:var(--muted); font-size:12px; margin-left:6px; }}
.group {{ display:block; margin:3px 0 3px 24px; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:#fff; cursor:pointer; }}
.group:hover {{ border-color:var(--blue); }}
.pill {{ display:inline-block; padding:2px 7px; border-radius:999px; background:var(--soft); border:1px solid var(--line); margin:2px; font-size:12px; }}
.pill.note {{ color:var(--amber); border-color:#FEEBC8; background:#FFFAF0; }}
.detail {{ border:1px solid var(--line); border-radius:8px; padding:14px; position:sticky; top:76px; max-height:760px; overflow:auto; }}
.raw-list {{ font-family:Consolas, monospace; font-size:12px; background:var(--soft); padding:10px; border-radius:6px; max-height:280px; overflow:auto; white-space:pre-wrap; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0 22px; font-size:13px; }}
th, td {{ border:1px solid var(--line); padding:7px 8px; vertical-align:top; }}
th {{ background:var(--soft); text-align:left; }}
.figure {{ border:1px solid var(--line); border-radius:8px; padding:10px; margin:16px 0 24px; overflow:auto; background:#fff; }}
.figure svg {{ max-width:100%; height:auto; }}
.profile-panel {{ border:1px solid var(--line); border-radius:8px; padding:14px; margin:14px 0 22px; background:#fff; }}
.profile-controls {{ display:grid; grid-template-columns:1fr 1fr 2fr; gap:10px; margin:12px 0; align-items:center; }}
.profile-visuals {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin:12px 0 20px; }}
.profile-collapse {{ border:1px solid var(--line); border-radius:8px; margin:12px 0 14px; padding:0; }}
.profile-collapse > summary {{ font-weight:700; padding:10px 12px; }}
.profile-collapse > table {{ margin:0; }}
.pie-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }}
.pie-card h4 {{ margin:0 0 10px; font-size:14px; }}
.pie-card-body {{ display:grid; grid-template-columns:150px minmax(0,1fr); gap:12px; align-items:start; }}
.pie-chart {{ width:140px; height:140px; border-radius:50%; border:1px solid var(--line); background:var(--soft); }}
.pie-legend {{ display:grid; gap:5px; font-size:12px; max-height:190px; overflow:auto; }}
.pie-legend-row {{ display:grid; grid-template-columns:12px minmax(0,1fr) auto; gap:6px; align-items:center; }}
.pie-swatch {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
.mini-bars {{ display:grid; gap:6px; margin:10px 0 18px; max-width:900px; }}
.mini-bar-row {{ display:grid; grid-template-columns:180px minmax(0,1fr) 90px; gap:8px; align-items:center; font-size:12px; }}
.mini-bar-track {{ height:14px; background:var(--soft); border:1px solid var(--line); border-radius:4px; overflow:hidden; }}
.mini-bar-fill {{ height:100%; background:var(--blue); }}
.table-note {{ color:var(--muted); font-size:12px; margin-top:-12px; }}
.mapping-note {{ border-left:4px solid var(--blue); background:var(--soft); padding:12px 14px; margin:14px 0 18px; }}
.mapping-grid {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:12px; margin:14px 0 18px; }}
.mapping-rule {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }}
.mapping-rule h3 {{ margin-top:0; }}
.mapping-rule p {{ margin-bottom:0; }}
.mapping-example-table th:first-child {{ width:190px; }}
.target-layout {{ display:grid; grid-template-columns:minmax(0, 1fr) 430px; gap:18px; align-items:start; margin-top:16px; }}
.target-family {{ margin:16px 0 8px; font-weight:700; }}
.target-bubbles {{ display:flex; flex-wrap:wrap; gap:8px; }}
.target-bubble {{ border:1px solid var(--line); background:#fff; border-radius:999px; padding:8px 11px; cursor:pointer; font-size:13px; max-width:360px; white-space:normal; text-align:left; }}
.target-bubble:hover, .target-bubble.active {{ border-color:var(--blue); color:var(--blue); }}
.target-detail {{ border:1px solid var(--line); border-radius:8px; padding:14px; position:sticky; top:76px; max-height:760px; overflow:auto; }}
.status-Available {{ color:var(--green); font-weight:700; }}
.status-Partial {{ color:var(--amber); font-weight:700; }}
.status-Missing {{ color:var(--red); font-weight:700; }}
code {{ background:var(--soft); padding:2px 4px; border-radius:4px; }}
@media (max-width:1100px) {{
  .grid {{ grid-template-columns:repeat(2, minmax(0,1fr)); }}
  .mapping-grid {{ grid-template-columns:1fr; }}
  .controls {{ grid-template-columns:1fr; }}
  .profile-controls {{ grid-template-columns:1fr; }}
  .profile-visuals {{ grid-template-columns:1fr; }}
  .pie-card-body {{ grid-template-columns:1fr; }}
  .tree-wrap {{ grid-template-columns:1fr; }}
  .target-layout {{ grid-template-columns:1fr; }}
  .detail {{ position:static; }}
  .target-detail {{ position:static; }}
}}
</style>
</head>
<body>
<header>
  <h1>Interactive UKB Dataset Visualization</h1>
  <p class="muted">Report-facing browser for the retrieved UKB application returns, metabolomics table, feature semantics, exact coverage, alignment, and internal-use non-ID example values. Participant identifiers are not embedded.</p>
</header>
<nav class="tabs">
  <button class="active" data-view="overview">Overview</button>
  <button data-view="tree">Dataset Structure Tree</button>
  <button data-view="coverage">Coverage & Alignment</button>
  <button data-view="profiling">Basic Profiling</button>
  <button data-view="requested">Requested Data Status</button>
</nav>
<main>
<section id="overview" class="view active">
  <div class="grid" id="metrics"></div>
  <h2>Dataset-first structure</h2>
  <div class="figure">{figures["dataset_tree"]}</div>
  <h2>Report purpose</h2>
  <p>This display translates UKB's wide field-instance-array storage into a lab-readable hierarchy. The top structure follows biological domains first, then acquisition/data type, scientific concepts, and exact raw-column traceability.</p>
</section>

<section id="tree" class="view">
  <h2>Dataset Structure Tree</h2>
  <p class="muted">Exhaustive semantic feature browser. The hierarchy starts with source CSV and biological domain, then acquisition/data type and concept. Click a parent node or leaf group to see exact participant-record coverage inside that source CSV.</p>
  <p class="muted"><strong>Coverage</strong> is based on non-empty values. Repeated visits are shown separately by UKB instance where present.</p>
  <div class="controls">
    <input id="search" placeholder="search">
    <select id="functionFilter"><option value="">All semantic functions</option></select>
    <select id="valueTypeFilter"><option value="">All value types</option></select>
    <label class="check-control"><input id="hideMissingNotes" type="checkbox" checked> Hide missing-note fields</label>
  </div>
  <div class="tree-wrap">
    <div class="tree" id="treeRoot"></div>
    <aside class="detail" id="detailBox"><h3>Select a node or feature group</h3><p class="muted">Details will appear here.</p></aside>
  </div>
</section>

<section id="coverage" class="view">
  <h2>Dataset alignment</h2>
  <div class="figure">{figures["upset_alignment"]}</div>
</section>

<section id="profiling" class="view">
  <h2>Basic Profiling</h2>
  <p class="muted">Aggregate participant, target, and missingness summaries for verified available fields.</p>
  <div class="subtabs" id="profileSubtabs">
    <button class="active" data-profile="participant">Participant Groups</button>
    <button data-profile="disease">Disease / Targets</button>
    <button data-profile="missingness">Missingness Patterns</button>
  </div>
  <div class="profile-controls">
    <select id="profileSource"><option value="">All sources</option></select>
    <select id="profileTheme"><option value="">All themes</option></select>
    <input id="profileSearch" placeholder="search">
  </div>
  <div id="profileContent" class="profile-panel"></div>
</section>

<section id="requested" class="view">
  <h2>Requested Data Type Status</h2>
  <p class="muted">Sorted by availability. <strong>Available</strong> means the retrieved files contain that data type in usable metadata/table form; <strong>Partial</strong> means incomplete data, missing assets, or incomplete requested submodules; <strong>Missing</strong> means that requested data type was not detected locally.</p>
  {requested_table}
</section>
</main>
<script id="dataset-json" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('dataset-json').textContent);
const groups = DATA.groups;
const targets = DATA.target_catalog || [];
const targetPredictors = DATA.target_predictor_availability || [];
const nodeCoverage = DATA.node_coverage_index || {{}};
const nodeCoverageRows = DATA.semantic_node_coverage || [];
const sourceDescriptions = DATA.source_descriptions || {{}};
let selectedDetailRenderer = null;

function fmt(n) {{ return Number(n || 0).toLocaleString(); }}
function unique(arr) {{ return [...new Set(arr)].filter(Boolean).sort(); }}
function optionize(id, values) {{
  const sel = document.getElementById(id);
  values.forEach(v => {{
    const opt = document.createElement('option');
    opt.value = v; opt.textContent = v;
    sel.appendChild(opt);
  }});
}}
function initMetrics() {{
  const summary = DATA.summary;
  const metrics = [
    ['Raw feature columns', summary.feature_count],
    ['Semantic groups', summary.semantic_group_count],
    ['Retrieved sources', Object.keys(summary.by_source).length],
    ['Participant records profiled', Math.max(...(DATA.inventory || []).map(r => Number(r.row_count || 0)))],
  ];
  document.getElementById('metrics').innerHTML = metrics.map(m =>
    `<div class="metric"><div class="value">${{fmt(m[1])}}</div><div class="muted">${{m[0]}}</div></div>`
  ).join('');
}}
function coverageScope() {{
  const cb = document.getElementById('hideMissingNotes');
  return cb && cb.checked ? 'hide_missing_note' : 'all';
}}
function coverageForPath(path, instance='all') {{
  return nodeCoverage[`${{coverageScope()}}|${{path}}|${{instance}}`];
}}
function coverageRowsForPath(path) {{
  return nodeCoverageRows
    .filter(r => r.coverage_scope === coverageScope() && r.tree_path === path && r.instance !== 'all')
    .sort((a, b) => Number(a.instance) - Number(b.instance));
}}
function groupPasses(g) {{
  const q = document.getElementById('search').value.trim().toLowerCase();
  const semanticFunction = document.getElementById('functionFilter').value;
  const valueType = document.getElementById('valueTypeFilter').value;
  const hideMissing = document.getElementById('hideMissingNotes').checked;
  if (hideMissing && g.semantic_function === 'missing_note') return false;
  const searchText = String(g.visible_search_text || g.search_text || '');
  if (q && !searchText.includes(q)) return false;
  if (semanticFunction && g.semantic_function_label !== semanticFunction) return false;
  if (valueType && g.value_type !== valueType) return false;
  return true;
}}
function nestedTree(filtered) {{
  const root = {{}};
  for (const g of filtered) {{
    const path = String(g.tree_path || '').split(' > ').filter(Boolean);
    let cur = root;
    let lastNode = null;
    const running = [];
    for (const p of path) {{
      running.push(p);
      cur[p] = cur[p] || {{ __groups: [], __children: {{}}, __path: running.join(' > ') }};
      lastNode = cur[p];
      cur = cur[p].__children;
    }}
    if (lastNode) lastNode.__groups.push(g);
  }}
  return root;
}}
function countFeatures(node) {{
  let total = 0, groupsN = 0;
  for (const key of Object.keys(node)) {{
    const item = node[key];
    for (const g of item.__groups) {{ total += Number(g.feature_count || 0); groupsN++; }}
    const c = countFeatures(item.__children); total += c.total; groupsN += c.groupsN;
  }}
  return {{total, groupsN}};
}}
function renderNode(name, item, depth=0) {{
  const counts = countFeatures({{[name]: item}});
  const cov = coverageForPath(item.__path);
  const covText = cov ? ` - ${{fmt(cov.participant_count)}} records (${{cov.source_percent}})` : '';
  const details = document.createElement('details');
  details.open = depth < 2;
  const summary = document.createElement('summary');
  summary.innerHTML = `${{escapeHtml(name)}} <span class="node-meta">${{fmt(counts.total)}} columns, ${{fmt(counts.groupsN)}} groups${{covText}}</span>`;
  summary.addEventListener('click', () => setTimeout(() => showNodeDetail(name, item, counts), 0));
  details.appendChild(summary);
  for (const g of item.__groups) {{
    const div = document.createElement('div');
    div.className = 'group';
    const note = g.semantic_function === 'missing_note' ? ' - missing note' : '';
    div.innerHTML = `${{escapeHtml(g.tree_leaf_label || g.attribute)}} <span class="node-meta">${{fmt(g.feature_count)}} columns - ${{escapeHtml(g.value_type || '')}}${{note}}</span>`;
    div.onclick = () => showDetail(g);
    details.appendChild(div);
  }}
  for (const key of Object.keys(item.__children).sort()) {{
    details.appendChild(renderNode(key, item.__children[key], depth + 1));
  }}
  return details;
}}
function renderTree() {{
  const filtered = groups.filter(groupPasses);
  const tree = nestedTree(filtered);
  const root = document.getElementById('treeRoot');
  root.innerHTML = `<p class="muted">${{fmt(filtered.length)}} semantic groups shown</p>`;
  for (const key of Object.keys(tree).sort()) root.appendChild(renderNode(key, tree[key]));
}}
function renderCoverageRows(rows) {{
  if (!rows || !rows.length) return '<p class="muted">No repeated-visit coverage rows for this selection.</p>';
  return `<table><thead><tr><th>Visit</th><th>Count</th><th>%</th></tr></thead><tbody>${{rows.map(r => `<tr><td>${{escapeHtml(r.instance)}}</td><td>${{fmt(r.participant_count)}}</td><td>${{escapeHtml(r.source_percent)}}</td></tr>`).join('')}}</tbody></table>`;
}}
function renderVisitSection(rows) {{
  if (!rows || !rows.length) return '';
  return `<h3>Visit-specific coverage</h3>${{renderCoverageRows(rows)}}`;
}}
function compactPath(parts) {{
  const out = [];
  for (const value of parts) {{
    const text = String(value || '').trim();
    if (!text) continue;
    if (out.some(existing => existing.toLowerCase() === text.toLowerCase())) continue;
    out.push(text);
  }}
  return out.join(' > ') || 'NA';
}}
function reportGrouping(g) {{
  return compactPath([g.report_domain, g.acquisition_type, g.report_concept]);
}}
function originalUkbConcept(g) {{
  return compactPath([g.concept, g.observation]);
}}
function collectNodeGroups(item) {{
  const out = [...(item.__groups || [])];
  for (const child of Object.values(item.__children || {{}})) out.push(...collectNodeGroups(child));
  return out;
}}
function countBy(items, getter) {{
  const counts = new Map();
  for (const item of items) {{
    const key = String(getter(item) || 'Unknown');
    counts.set(key, (counts.get(key) || 0) + 1);
  }}
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}}
function formatMix(entries, maxItems=6) {{
  if (!entries || !entries.length) return 'NA';
  return entries.map(([label, count]) => `${{escapeHtml(label)}} (${{fmt(count)}})`).join('<br>');
}}
function countEncodedItems(value) {{
  const items = String(value || '')
    .split(';')
    .map(v => v.trim())
    .filter(v => v && v.toUpperCase() !== 'NA');
  return Math.max(1, items.length);
}}
function numericCount(value, fallback) {{
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}}
function featureColumnDetail(g, raw) {{
  const visitCount = numericCount(g.visit_instance_count, countEncodedItems(g.instances));
  const repeatedCount = numericCount(g.repeated_entry_count, countEncodedItems(g.arrays));
  const pairedCount = numericCount(g.paired_column_count, 1);
  const pairedText = pairedCount > 1 ? `; paired:${{fmt(pairedCount)}}` : '';
  const annotateColumns = g.value_type === 'mixed' || g.semantic_function === 'mixed';
  const annotations = Array.isArray(g.raw_column_annotations) && g.raw_column_annotations.length
    ? g.raw_column_annotations
    : raw.map(column => ({{column, value_type: '', semantic_function_label: ''}}));
  const columnLines = annotations.map(item => {{
    const bits = annotateColumns
      ? [item.value_type, item.semantic_function_label || item.semantic_function].filter(Boolean)
      : [];
    return `${{escapeHtml(item.column)}}${{bits.length ? ` [${{bits.map(escapeHtml).join('; ')}}]` : ''}}`;
  }}).join('\\n');
  return `
    ${{fmt(g.feature_count)}}<br>
    <span class="muted">Visits: ${{fmt(visitCount)}}; repeated entries: ${{fmt(repeatedCount)}}${{pairedText}}</span>
    <details>
      <summary>Column list (${{fmt(raw.length)}})</summary>
      <div class="raw-list">${{columnLines}}</div>
    </details>
  `;
}}
function repeatedEntryRow(g) {{
  const repeatedCount = numericCount(g.repeated_entry_count, countEncodedItems(g.arrays));
  if (repeatedCount <= 1 || !g.repeated_entry_explanation) return '';
  return `<tr><th>Repeated entries</th><td>${{escapeHtml(g.repeated_entry_explanation)}}</td></tr>`;
}}
function pairedColumnRow(g) {{
  const pairedCount = numericCount(g.paired_column_count, 1);
  if (pairedCount <= 1 || !g.paired_column_explanation) return '';
  return `<tr><th>Paired columns</th><td>${{escapeHtml(g.paired_column_explanation)}}</td></tr>`;
}}
function localDerivationRow(g) {{
  if (!g.local_derivation_explanation) return '';
  return `<tr><th>Local derivation</th><td>${{escapeHtml(g.local_derivation_explanation)}}</td></tr>`;
}}
function coverageText(g) {{
  const label = g.case_only_field === true || String(g.case_only_field).toLowerCase() === 'true'
    ? ' (Case-only field)'
    : '';
  return `${{fmt(g.participant_count_proxy)}} (${{escapeHtml(g.source_percent_proxy)}})${{label}}`;
}}
function sourceInfoRow(name, item) {{
  const isSourceNode = item.__path === name;
  const text = isSourceNode ? sourceDescriptions[name] : '';
  return text ? `<tr><th>File meaning</th><td>${{escapeHtml(text)}}</td></tr>` : '';
}}
function showNodeDetail(name, item, counts) {{
  selectedDetailRenderer = () => showNodeDetail(name, item, counts);
  const cov = coverageForPath(item.__path);
  const visitRows = coverageRowsForPath(item.__path);
  const nodeGroups = collectNodeGroups(item);
  document.getElementById('detailBox').innerHTML = `
    <h3>${{escapeHtml(name)}}</h3>
    <p><span class="pill">${{escapeHtml(coverageScope() === 'hide_missing_note' ? 'Missing notes hidden' : 'All fields shown')}}</span></p>
    <table>
      ${{sourceInfoRow(name, item)}}
      <tr><th>Feature columns</th><td>${{fmt(counts.total)}}</td></tr>
      <tr><th>Feature groups</th><td>${{fmt(counts.groupsN)}}</td></tr>
      <tr><th>Coverage</th><td>${{cov ? `${{fmt(cov.participant_count)}} (${{escapeHtml(cov.source_percent)}})` : 'Not profiled'}}</td></tr>
      <tr><th>Value types</th><td>${{formatMix(countBy(nodeGroups, g => g.value_type))}}</td></tr>
      <tr><th>Semantic functions</th><td>${{formatMix(countBy(nodeGroups, g => g.semantic_function_label || g.semantic_function))}}</td></tr>
    </table>
    ${{renderVisitSection(visitRows)}}
    <p class="muted">Coverage summarizes visible fields under this node.</p>
  `;
}}
function showDetail(g) {{
  selectedDetailRenderer = () => showDetail(g);
  const raw = (g.raw_columns || '').split(';').filter(Boolean);
  const urls = (g.ukb_showcase_urls || '').split(';').filter(Boolean);
  document.getElementById('detailBox').innerHTML = `
    <h3>${{escapeHtml(g.tree_leaf_label || g.attribute)}}</h3>
    <p><span class="pill">${{escapeHtml(g.source_label)}}</span><span class="pill">${{escapeHtml(g.value_type || '')}}</span><span class="pill ${{g.semantic_function === 'missing_note' ? 'note' : ''}}">${{escapeHtml(g.semantic_function_label || g.semantic_function)}}</span></p>
    <table>
      <tr><th>Source CSV</th><td>${{escapeHtml(g.source_label)}}</td></tr>
      <tr><th>Grouping</th><td>${{escapeHtml(reportGrouping(g))}}</td></tr>
      <tr><th>Original UKB grouping</th><td>${{escapeHtml(originalUkbConcept(g))}}</td></tr>
      <tr><th>Feature columns</th><td>${{featureColumnDetail(g, raw)}}</td></tr>
      ${{repeatedEntryRow(g)}}
      ${{pairedColumnRow(g)}}
      ${{localDerivationRow(g)}}
      <tr><th>Coverage</th><td>${{coverageText(g)}}</td></tr>
      <tr><th>Value type</th><td>${{escapeHtml(g.value_type || 'mixed/unknown')}}</td></tr>
      <tr><th>Semantic function</th><td>${{escapeHtml(g.semantic_function_label || g.semantic_function)}}</td></tr>
    </table>
    ${{renderVisitSection(g.visit_coverage || [])}}
    ${{renderExamples(g)}}
    <h3>UKB reference</h3>
    ${{urls.length ? `<p>${{urls.map(u => `<a href="${{escapeHtml(u)}}">${{escapeHtml(u)}}</a>`).join('<br>')}}</p>` : '<p class="muted">No UKB reference link available.</p>'}}
  `;
}}
function renderExamples(g) {{
  const examples = g.example_entries || [];
  if (!examples.length) return `<p class="muted">${{escapeHtml(g.example_status || 'No examples available')}}</p>`;
  return examples.map((example, i) => {{
    const values = example.values || [];
    return `<h3>Example ${{i + 1}}</h3><div class="raw-list">${{values.map(v => `${{escapeHtml(v.column)}} = ${{escapeHtml(v.value)}}`).join('\\n')}}</div>`;
  }}).join('');
}}
function escapeHtml(value) {{
  const map = {{'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;'}};
  return String(value ?? '').replace(/[&<>"]/g, ch => map[ch]);
}}
function initTargets() {{
  const container = document.getElementById('targetBubbles');
  if (!container || !targets.length) return;
  const byFamily = {{}};
  for (const target of targets) {{
    byFamily[target.target_family] = byFamily[target.target_family] || [];
    byFamily[target.target_family].push(target);
  }}
  container.innerHTML = '';
  for (const family of Object.keys(byFamily).sort()) {{
    const heading = document.createElement('div');
    heading.className = 'target-family';
    heading.textContent = `${{family}} (${{fmt(byFamily[family].length)}})`;
    container.appendChild(heading);
    const wrap = document.createElement('div');
    wrap.className = 'target-bubbles';
    for (const target of byFamily[family]) {{
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'target-bubble';
      btn.innerHTML = `${{escapeHtml(target.target_name)}} <span class="node-meta">(${{fmt(target.feature_count)}})</span>`;
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.target-bubble').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        showTarget(target);
      }});
      wrap.appendChild(btn);
    }}
    container.appendChild(wrap);
  }}
  showTarget(targets[0]);
}}
function showTarget(target) {{
  const rows = targetPredictors.filter(r => r.target_id === target.target_id);
  const raw = String(target.exact_target_columns || '').split(';').filter(Boolean);
  document.getElementById('targetDetail').innerHTML = `
    <h3>${{escapeHtml(target.target_name)}}</h3>
    <p><span class="pill">${{escapeHtml(target.target_family)}}</span><span class="pill">${{escapeHtml(target.target_type)}}</span><span class="pill">${{escapeHtml(target.source_label)}}</span></p>
    <table>
      <tr><th>Feature columns</th><td>${{fmt(target.feature_count)}}</td></tr>
      <tr><th>Coverage proxy</th><td>${{fmt(target.participant_count_proxy)}} participant records</td></tr>
      <tr><th>Coverage count type</th><td>${{escapeHtml(target.coverage_count_type)}}</td></tr>
      <tr><th>Notes</th><td>${{escapeHtml(target.notes)}}</td></tr>
    </table>
    <h3>Exact target columns</h3>
    <div class="raw-list">${{raw.map(escapeHtml).join('\\n')}}</div>
    <h3>Predictor availability</h3>
    <table>
      <thead><tr><th>Predictor</th><th>Availability</th><th>Reason</th></tr></thead>
      <tbody>
        ${{rows.map(r => `<tr><td>${{escapeHtml(r.predictor)}}</td><td class="status-${{escapeHtml(r.availability_status)}}">${{escapeHtml(r.availability_status)}}</td><td>${{escapeHtml(r.reason)}}</td></tr>`).join('')}}
      </tbody>
    </table>
  `;
}}
const basicProfiles = DATA.basic_profiles || {{}};
let activeProfileSubtab = 'participant';
const PROFILE_SOURCE_PRIORITY = ['METABOLITE.csv','Eye-kidney ukb42408','Eye-brain ukb42577','Eye-brain ukb43216'];
function profileRows(name) {{ return basicProfiles[name] || []; }}
function allProfileRows() {{
  return [
    ...profileRows('participant_groups'),
    ...profileRows('disease_targets'),
    ...profileRows('missingness_patterns'),
  ];
}}
function rowText(row) {{
  return Object.values(row || {{}}).join(' ').toLowerCase();
}}
function selectedProfileSource() {{
  return document.getElementById('profileSource')?.value || '';
}}
function selectedProfileTheme() {{
  return document.getElementById('profileTheme')?.value || '';
}}
function profileSearchText() {{
  return (document.getElementById('profileSearch')?.value || '').trim().toLowerCase();
}}
function filteredProfileRows(rows, useTheme=true) {{
  const source = selectedProfileSource();
  const theme = selectedProfileTheme();
  const q = profileSearchText();
  return rows.filter(row => {{
    if (source && row.source_label !== source) return false;
    if (useTheme && theme && row.theme !== theme && row.target_family !== theme && row.variable !== theme) return false;
    if (q && !rowText(row).includes(q)) return false;
    return true;
  }});
}}
const PIE_COLORS = ['#2F6F9F','#2F855A','#B7791F','#C53030','#6B46C1','#0F766E','#C05621','#475569','#8A4B7D','#4A5568','#2B6CB0','#718096'];
function numberValue(value) {{
  if (value === null || value === undefined || value === '') return 0;
  const n = Number(String(value).replace(/[% ,]/g, ''));
  return Number.isFinite(n) ? n : 0;
}}
function profileValue(row, field) {{
  const value = row[field];
  if (value === true || value === 'True' || value === 'true') return 'yes';
  if (value === false || value === 'False' || value === 'false') return 'no';
  return escapeHtml(value ?? '');
}}
function profileTable(rows, fields, labels={{}}) {{
  const shown = rows;
  if (!shown.length) return '<p class="muted">No rows for the current filters.</p>';
  const header = fields.map(field => `<th>${{escapeHtml(labels[field] || field)}}</th>`).join('');
  const body = shown.map(row => `<tr>${{fields.map(field => `<td>${{profileValue(row, field)}}</td>`).join('')}}</tr>`).join('');
  return `<table><thead><tr>${{header}}</tr></thead><tbody>${{body}}</tbody></table>`;
}}
function collapsedProfileSection(title, content) {{
  return `<details class="profile-collapse"><summary>${{escapeHtml(title)}}</summary>${{content}}</details>`;
}}
function aggregateEntries(rows, labelGetter, countGetter) {{
  const map = new Map();
  rows.forEach(row => {{
    const rawLabel = typeof labelGetter === 'function' ? labelGetter(row) : row[labelGetter];
    const rawCount = typeof countGetter === 'function' ? countGetter(row) : row[countGetter];
    const label = String(rawLabel || 'Unspecified');
    const count = numberValue(rawCount);
    if (count <= 0) return;
    map.set(label, (map.get(label) || 0) + count);
  }});
  return [...map.entries()]
    .map(([label, count]) => ({{label, count}}))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}}
function countEntries(rows, labelGetter) {{
  return aggregateEntries(rows, labelGetter, () => 1);
}}
function preferredProfileSource(rows) {{
  const selected = selectedProfileSource();
  if (selected && rows.some(row => row.source_label === selected)) return selected;
  for (const source of PROFILE_SOURCE_PRIORITY) {{
    if (rows.some(row => row.source_label === source)) return source;
  }}
  const labels = unique(rows.map(row => row.source_label));
  return labels[0] || '';
}}
function canonicalSourceRows(rows) {{
  if (!rows.length) return [];
  const source = preferredProfileSource(rows);
  return source ? rows.filter(row => row.source_label === source) : rows;
}}
function groupTotal(rows, field='count') {{
  return rows.reduce((sum, row) => sum + numberValue(row[field]), 0);
}}
function rowInstanceRank(row) {{
  const column = String(row.column || '');
  const instance = String(row.instance ?? '');
  if (column === 'gender' || column === 'baselineage') return -1;
  if (instance === '0') return 0;
  if (instance === '') return 1;
  const n = Number(instance);
  return Number.isFinite(n) ? 2 + n : 99;
}}
function canonicalVariableRows(rows, variable, profileType='') {{
  let subset = rows.filter(row => row.variable === variable && (!profileType || row.profile_type === profileType));
  if (!subset.length) return [];
  subset = canonicalSourceRows(subset);
  const grouped = new Map();
  subset.forEach(row => {{
    const key = String(row.source_label || '') + '||' + String(row.column || '') + '||' + String(row.instance ?? '');
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }});
  return [...grouped.values()].sort((a, b) => {{
    const rank = rowInstanceRank(a[0]) - rowInstanceRank(b[0]);
    if (rank) return rank;
    return groupTotal(b) - groupTotal(a);
  }})[0] || [];
}}
function chartTitleWithSource(title, rows) {{
  if (!rows.length) return title;
  const first = rows[0];
  const source = first.source_label ? ` (${{escapeHtml(first.source_label)}}` : '';
  const instance = first.instance !== undefined && first.instance !== null && String(first.instance) !== '' ? `, visit ${{escapeHtml(first.instance)}}` : '';
  return source ? `${{title}}${{source}}${{instance}})` : title;
}}
function pieChartFromEntries(entries, title) {{
  const clean = entries.filter(entry => entry.count > 0);
  if (!clean.length) return '';
  const total = clean.reduce((sum, entry) => sum + entry.count, 0);
  let cursor = 0;
  const stops = clean.map((entry, idx) => {{
    const next = cursor + (100 * entry.count / total);
    const color = PIE_COLORS[idx % PIE_COLORS.length];
    const stop = `${{color}} ${{cursor.toFixed(4)}}% ${{next.toFixed(4)}}%`;
    cursor = next;
    return stop;
  }});
  const legend = clean.map((entry, idx) => {{
    const color = PIE_COLORS[idx % PIE_COLORS.length];
    const pctValue = total ? (100 * entry.count / total).toFixed(1) : '0.0';
    return `<div class="pie-legend-row"><span class="pie-swatch" style="background:${{color}}"></span><span>${{escapeHtml(entry.label)}}</span><span>${{fmt(entry.count)}} (${{pctValue}}%)</span></div>`;
  }}).join('');
  return `
    <div class="pie-card">
      <h4>${{escapeHtml(title)}}</h4>
      <div class="pie-card-body">
        <div class="pie-chart" style="background:conic-gradient(${{stops.join(',')}})"></div>
        <div class="pie-legend">${{legend}}</div>
      </div>
    </div>
  `;
}}
function pieSection(title, cards) {{
  const html = cards.filter(Boolean).join('');
  return html ? `<h3>${{escapeHtml(title)}}</h3><div class="profile-visuals">${{html}}</div>` : '';
}}
function categoryPieSection(rows, variables, title) {{
  const cards = variables.map(variable => {{
    const subset = canonicalVariableRows(rows, variable, 'category_count');
    if (!subset.length) return '';
    return pieChartFromEntries(aggregateEntries(subset, row => row.value_label || row.value, 'count'), chartTitleWithSource(variable, subset));
  }});
  return pieSection(title, cards);
}}
function miniBars(rows, labelField, countField, title) {{
  const sorted = aggregateEntries(rows, labelField, countField);
  if (!sorted.length) return '';
  const maxValue = Math.max(...sorted.map(row => row.count));
  return `
    <h3>${{escapeHtml(title)}}</h3>
    <div class="mini-bars">
      ${{sorted.map(row => {{
        const value = row.count;
        const width = maxValue ? Math.max(1, 100 * value / maxValue) : 0;
        return `<div class="mini-bar-row"><div>${{escapeHtml(row.label)}}</div><div class="mini-bar-track"><div class="mini-bar-fill" style="width:${{width}}%"></div></div><div>${{fmt(value)}}</div></div>`;
      }}).join('')}}
    </div>
  `;
}}
function updateProfileThemeOptions() {{
  const sel = document.getElementById('profileTheme');
  if (!sel) return;
  const current = sel.value;
  let values = [];
  if (activeProfileSubtab === 'disease') {{
    values = unique(profileRows('disease_targets').map(row => row.target_family));
  }} else if (activeProfileSubtab === 'participant') {{
    values = unique(profileRows('participant_groups').map(row => row.variable));
  }} else {{
    values = unique(profileRows('missingness_patterns').map(row => row.theme));
  }}
  sel.innerHTML = '<option value="">All themes</option>';
  values.forEach(value => {{
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = value;
    sel.appendChild(opt);
  }});
  if (values.includes(current)) sel.value = current;
}}
function renderParticipantProfile() {{
  const rows = filteredProfileRows(profileRows('participant_groups'));
  const hist = canonicalVariableRows(rows, 'Age at assessment', 'histogram');
  const numeric = rows.filter(row => row.profile_type === 'numeric_summary');
  const categorical = rows.filter(row => row.profile_type === 'category_count');
  const numericTable = profileTable(numeric, ['source_label','variable','column','instance','count','percent','mean','sd','median','q1','q3','min','max'], {{source_label:'Source', variable:'Variable'}});
  const categoricalTable = profileTable(categorical, ['source_label','variable','column','instance','value_label','value','count','percent'], {{source_label:'Source', variable:'Variable', value_label:'Label', value:'Code'}});
  return `
    <p class="muted">Participant composition summaries for sex, age, assessment metadata, ethnicity, ancestry, and genetic QC grouping fields.</p>
    ${{miniBars(hist, 'value', 'count', 'Age histogram')}}
    ${{categoryPieSection(categorical, ['Sex / gender','Genetic sex','Ethnic background','Assessment center'], 'Categorical group distributions')}}
    ${{collapsedProfileSection('Numeric summaries', numericTable)}}
    ${{collapsedProfileSection('Category counts', categoricalTable)}}
  `;
}}
function renderDiseaseProfile() {{
  const rows = filteredProfileRows(profileRows('disease_targets'));
  const summary = rows.filter(row => row.profile_type === 'target_summary');
  const prevalence = canonicalSourceRows(rows.filter(row => row.profile_type === 'disease_family_prevalence'));
  const linked = summary.filter(row => String(row.case_only).toLowerCase() === 'true');
  const familyPie = pieChartFromEntries(aggregateEntries(prevalence, 'target_family', 'participant_count'), chartTitleWithSource('Disease-positive participants', prevalence));
  const summaryTable = profileTable(summary, ['source_label','target_family','target_name','target_type','case_only','target_count','participant_count','percent'], {{source_label:'Source', target_family:'Family', target_count:'Cases / records', participant_count:'Records with data'}});
  const linkedTable = profileTable(linked, ['source_label','target_family','target_name','target_count','participant_count','percent'], {{source_label:'Source', target_family:'Family', target_count:'Cases / records', participant_count:'Records with data'}});
  return `
    <p class="muted">Disease and target summaries from hospital, self-report, death, linked endpoint, and METABOLITE curated disease fields.</p>
    ${{pieSection('Disease / target composition', [familyPie])}}
    ${{collapsedProfileSection('Target summaries', summaryTable)}}
    ${{collapsedProfileSection('Case-only linked reports', linkedTable)}}
  `;
}}
function renderMissingnessProfile() {{
  const rows = filteredProfileRows(profileRows('missingness_patterns'));
  const block = rows.filter(row => row.profile_type === 'theme_missingness' && row.group_variable === 'ALL');
  const burden = rows.filter(row => row.profile_type === 'participant_theme_burden');
  const grouped = rows.filter(row => row.profile_type !== 'participant_theme_burden' && row.group_variable !== 'ALL');
  const themePie = pieChartFromEntries(aggregateEntries(block, 'theme', 'with_data_count'), 'Records with data by profiled theme');
  const burdenPie = pieChartFromEntries(aggregateEntries(burden, 'group_value', 'participant_count'), 'Participant records by number of themes present');
  const blockTable = profileTable(block, ['source_label','theme','participant_count','with_data_count','missing_count','present_percent'], {{source_label:'Source'}});
  const burdenTable = profileTable(burden, ['source_label','group_value','participant_count','present_percent'], {{source_label:'Source', group_value:'Number of themes with data'}});
  const groupedTable = profileTable(grouped, ['source_label','profile_type','group_variable','group_value_label','group_value','theme','participant_count','with_data_count','missing_count','present_percent'], {{source_label:'Source', group_value_label:'Label', group_value:'Code'}});
  return `
    <p class="muted">Missingness summaries by participant grouping fields, disease status, and profiled feature themes.</p>
    ${{pieSection('Missingness composition', [themePie, burdenPie])}}
    ${{collapsedProfileSection('Theme-level missingness', blockTable)}}
    ${{collapsedProfileSection('Participant theme burden', burdenTable)}}
    ${{collapsedProfileSection('Missingness by group', groupedTable)}}
  `;
}}
function renderBasicProfiling() {{
  updateProfileThemeOptions();
  const container = document.getElementById('profileContent');
  if (!container) return;
  if (activeProfileSubtab === 'participant') container.innerHTML = renderParticipantProfile();
  else if (activeProfileSubtab === 'disease') container.innerHTML = renderDiseaseProfile();
  else container.innerHTML = renderMissingnessProfile();
}}
function initBasicProfiling() {{
  const sourceSelect = document.getElementById('profileSource');
  if (!sourceSelect) return;
  unique(allProfileRows().map(row => row.source_label)).forEach(source => {{
    const opt = document.createElement('option');
    opt.value = source;
    opt.textContent = source;
    sourceSelect.appendChild(opt);
  }});
  document.querySelectorAll('#profileSubtabs button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('#profileSubtabs button').forEach(item => item.classList.remove('active'));
      btn.classList.add('active');
      activeProfileSubtab = btn.dataset.profile;
      document.getElementById('profileTheme').value = '';
      renderBasicProfiling();
    }});
  }});
  ['profileSource','profileTheme','profileSearch'].forEach(id => {{
    document.getElementById(id).addEventListener('input', renderBasicProfiling);
    document.getElementById(id).addEventListener('change', renderBasicProfiling);
  }});
  renderBasicProfiling();
}}
document.querySelectorAll('.tabs button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.view).classList.add('active');
  }});
}});
optionize('functionFilter', unique(groups.map(g => g.semantic_function_label)));
optionize('valueTypeFilter', unique(groups.map(g => g.value_type)));
function resetTreeControls() {{
  document.getElementById('search').value = '';
  document.getElementById('functionFilter').value = '';
  document.getElementById('valueTypeFilter').value = '';
  document.getElementById('hideMissingNotes').checked = true;
}}
resetTreeControls();
['search','functionFilter','valueTypeFilter'].forEach(id => {{
  document.getElementById(id).addEventListener('input', renderTree);
  document.getElementById(id).addEventListener('change', renderTree);
}});
document.getElementById('hideMissingNotes').addEventListener('change', () => {{
  renderTree();
  if (selectedDetailRenderer) selectedDetailRenderer();
}});
window.addEventListener('pageshow', () => {{
  resetTreeControls();
  renderTree();
  if (selectedDetailRenderer) selectedDetailRenderer();
}});
initMetrics();
renderTree();
initTargets();
initBasicProfiling();
</script>
</body>
</html>"""
    (OUT_DIR / "index.html").write_text(html_text, encoding="utf-8")


def validate(groups, columns, tree_payload):
    raw_columns = [raw_column_id(r) for r in columns]
    mapped = set(tree_payload["raw_to_group"])
    raw_set = set(raw_columns)
    missing = sorted(raw_set - mapped)
    extras = sorted(mapped - raw_set)

    rows_by_field_source: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in columns:
        if row["field_id"] in PAIR_FIELDS:
            rows_by_field_source[(row["source_id"], row["field_id"])].add((row["instance"], row["array"]))
    pair_checks = []
    pair_pairs = [
        ("41270", "41280"),
        ("41271", "41281"),
        ("41202", "41262"),
        ("41203", "41263"),
        ("41272", "41282"),
        ("41273", "41283"),
        ("41200", "41260"),
        ("41201", "41257"),
    ]
    for source_id in SOURCE_LABELS:
        for a, b in pair_pairs:
            a_slots = rows_by_field_source.get((source_id, a), set())
            b_slots = rows_by_field_source.get((source_id, b), set())
            if not a_slots and not b_slots:
                continue
            pair_checks.append(
                {
                    "source_id": source_id,
                    "field_a": a,
                    "field_b": b,
                    "field_a_slots": len(a_slots),
                    "field_b_slots": len(b_slots),
                    "matched_slots": len(a_slots & b_slots),
                    "a_without_b": len(a_slots - b_slots),
                    "b_without_a": len(b_slots - a_slots),
                }
            )

    validation = {
        "raw_columns": len(raw_columns),
        "unique_raw_columns": len(raw_set),
        "semantic_groups": len(groups),
        "raw_columns_mapped": len(mapped),
        "missing_raw_columns": len(missing),
        "extra_mapped_columns": len(extras),
        "pair_checks": pair_checks,
        "status": "PASS" if not missing and not extras else "CHECK",
    }
    (SUPPORT_DIR / "validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def export_structure_source_files(validation: dict[str, object]) -> None:
    SOURCE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        ("Generator script", Path(__file__).resolve(), "build_interactive_dataset_visualization.py"),
        ("Column-level metadata", META_DIR / "main_column_metadata.csv", "main_column_metadata.csv"),
        ("Field-level metadata", META_DIR / "main_field_metadata.csv", "main_field_metadata.csv"),
        ("Source file inventory", META_DIR / "main_file_inventory.csv", "main_file_inventory.csv"),
        ("Semantic feature groups", SUPPORT_DIR / "semantic_feature_groups.csv", "semantic_feature_groups.csv"),
        ("Semantic tree payload", SUPPORT_DIR / "semantic_feature_tree.json", "semantic_feature_tree.json"),
        ("Leaf/group coverage", SUPPORT_DIR / "semantic_group_coverage.csv", "semantic_group_coverage.csv"),
        ("Tree node coverage", SUPPORT_DIR / "semantic_node_coverage.csv", "semantic_node_coverage.csv"),
        ("Tree review candidates", SUPPORT_DIR / "tree_structure_review_candidates.csv", "tree_structure_review_candidates.csv"),
        ("Tree audit summary", SUPPORT_DIR / "tree_structure_audit_summary.csv", "tree_structure_audit_summary.csv"),
        ("Requested data status", SUPPORT_DIR / "requested_data_type_status.csv", "requested_data_type_status.csv"),
        ("ID namespace notes", SUPPORT_DIR / "id_namespace_interpretation.csv", "id_namespace_interpretation.csv"),
        ("Bridge alignment summary", SUPPORT_DIR / "dataset_alignment_bridge_summary.csv", "dataset_alignment_bridge_summary.csv"),
        ("Bridge-aware source counts", SUPPORT_DIR / "dataset_alignment_set_summary.csv", "dataset_alignment_set_summary.csv"),
        ("Validation report", SUPPORT_DIR / "validation_report.json", "validation_report.json"),
    ]
    copied_rows = []
    for label, src, filename in files:
        status = "missing"
        if src.exists():
            shutil.copy2(src, SOURCE_EXPORT_DIR / filename)
            status = "copied"
        copied_rows.append((label, filename, status))

    lines = [
        "# Dataset Structure Tree Source Files",
        "",
        "This folder is an audit bundle for understanding how the Dataset Structure Tree was generated. The browser visualization uses the embedded payload inside the outer `index.html`; these files are included for traceability and report review.",
        "",
        "## Contents",
        "",
    ]
    for label, filename, status in copied_rows:
        lines.append(f"- `{filename}`: {label} ({status}).")
    lines.extend(
        [
            "",
            f"Validation status: `{validation.get('status', 'UNKNOWN')}`; mapped raw columns: {to_int(validation.get('raw_columns_mapped', 0)):,}.",
        ]
    )
    (SOURCE_EXPORT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    columns = read_csv(META_DIR / "main_column_metadata.csv")
    fields = read_csv(META_DIR / "main_field_metadata.csv")
    inventory = read_csv(META_DIR / "main_file_inventory.csv")
    modality_counts = read_csv(PLOTS_DIR / "modality_counts_by_cohort.csv")
    modality_intersections = read_csv(PLOTS_DIR / "modality_intersections_by_cohort.csv")
    dataset_alignment = read_csv(PLOTS_DIR / "dataset_alignment_intersections.csv")
    bridge_alignment_summary = read_csv(PLOTS_DIR / "dataset_alignment_bridge_summary.csv")
    alignment_set_summary = read_csv(PLOTS_DIR / "dataset_alignment_set_summary.csv")
    global_participants = sum(to_int(r["participant_count"]) for r in dataset_alignment)

    groups, tree_payload = build_semantic_groups(columns, inventory, modality_counts, global_participants)
    groups, group_coverage_rows, node_coverage_rows = profile_semantic_coverage(groups, columns, inventory)
    summary = build_tree_summary(groups)

    requested_rows = requested_data_status_rows()
    target_rows = target_catalog_rows(columns, inventory)
    target_predictor_rows = target_predictor_availability_rows(target_rows)
    namespace_rows = id_namespace_rows(dataset_alignment, bridge_alignment_summary)
    basic_profiles = build_basic_profiles(groups, columns, inventory)
    tree_review_rows = build_tree_review_candidates(groups)
    tree_audit_rows = build_tree_audit_summary(groups, tree_review_rows)

    group_fields = [
        "group_id",
        "source_id",
        "source_label",
        "modality_category",
        "modality_label",
        "requested_area",
        "requested_type",
        "scientific_domain",
        "concept",
        "observation",
        "display_concept",
        "attribute",
        "legacy_tree_path",
        "report_domain",
        "acquisition_type",
        "report_concept",
        "report_leaf_label",
        "tree_path",
        "tree_leaf_label",
        "visible_search_text",
        "tree_review_flags",
        "storage_pattern",
        "column_layout",
        "column_layout_detail",
        "semantic_function",
        "semantic_function_label",
        "value_type",
        "value_representation",
        "prediction_role",
        "prediction_role_label",
        "leakage_or_temporality_warning",
        "coverage_explanation",
        "case_only_field",
        "feature_count",
        "participant_count_proxy",
        "coverage_count_type",
        "source_participant_count",
        "source_percent_proxy",
        "global_percent_proxy",
        "modality_participant_count",
        "modality_source_percent",
        "field_ids",
        "instances",
        "arrays",
        "visit_instance_count",
        "repeated_entry_count",
        "paired_column_count",
        "paired_column_explanation",
        "repeated_entry_explanation",
        "local_derivation_explanation",
        "data_type_mix",
        "raw_columns",
        "raw_columns_sample",
        "raw_column_annotations_json",
        "visit_coverage_json",
        "example_status",
        "example_entries_json",
        "ukb_showcase_urls",
    ]
    write_csv(META_DIR / "semantic_feature_groups.csv", groups, group_fields)
    write_csv(SUPPORT_DIR / "semantic_feature_groups.csv", groups, group_fields)
    write_csv(
        META_DIR / "semantic_group_coverage.csv",
        group_coverage_rows,
        ["source_id", "group_id", "instance", "participant_count", "source_participant_count", "source_percent"],
    )
    write_csv(
        SUPPORT_DIR / "semantic_group_coverage.csv",
        group_coverage_rows,
        ["source_id", "group_id", "instance", "participant_count", "source_participant_count", "source_percent"],
    )
    write_csv(
        META_DIR / "semantic_node_coverage.csv",
        node_coverage_rows,
        ["coverage_scope", "source_id", "node_id", "tree_path", "instance", "participant_count", "source_participant_count", "source_percent", "group_count", "feature_count"],
    )
    write_csv(
        SUPPORT_DIR / "semantic_node_coverage.csv",
        node_coverage_rows,
        ["coverage_scope", "source_id", "node_id", "tree_path", "instance", "participant_count", "source_participant_count", "source_percent", "group_count", "feature_count"],
    )
    write_csv(SUPPORT_DIR / "requested_data_type_status.csv", requested_rows, ["availability_status", "requested_area", "requested_type", "notes"])
    write_csv(
        SUPPORT_DIR / "tree_structure_review_candidates.csv",
        tree_review_rows,
        [
            "source_id",
            "source_label",
            "field_ids",
            "description",
            "semantic_function",
            "semantic_function_label",
            "value_type",
            "current_path",
            "proposed_path",
            "report_domain",
            "acquisition_type",
            "report_concept",
            "flags",
            "raw_columns_sample",
            "ukb_showcase_urls",
        ],
    )
    write_csv(
        SUPPORT_DIR / "tree_structure_audit_summary.csv",
        tree_audit_rows,
        ["metric", "value", "notes"],
    )
    target_fields = [
        "target_id",
        "target_name",
        "target_family",
        "target_type",
        "source_id",
        "source_label",
        "feature_count",
        "participant_count_proxy",
        "coverage_count_type",
        "exact_target_columns",
        "exact_target_columns_sample",
        "notes",
    ]
    write_csv(SUPPORT_DIR / "target_catalog.csv", target_rows, target_fields)
    write_csv(
        SUPPORT_DIR / "target_predictor_availability.csv",
        target_predictor_rows,
        ["target_id", "target_name", "target_family", "predictor", "availability_status", "reason"],
    )
    write_csv(
        SUPPORT_DIR / "prediction_feasibility_matrix.csv",
        target_predictor_rows,
        ["target_id", "target_name", "target_family", "predictor", "availability_status", "reason"],
    )
    write_csv(
        SUPPORT_DIR / "id_namespace_interpretation.csv",
        namespace_rows,
        ["id_namespace", "role", "notes"],
    )
    write_csv(
        SUPPORT_DIR / "dataset_alignment_bridge_summary.csv",
        bridge_alignment_summary,
        [
            "bridge_file",
            "left_namespace",
            "right_namespace",
            "row_count",
            "left_unique",
            "right_unique",
            "duplicate_rows",
            "left_duplicates",
            "right_duplicates",
            "status",
            "notes",
            "left_overlap_with_local_source",
            "right_overlap_with_local_source",
        ],
    )
    write_csv(
        SUPPORT_DIR / "dataset_alignment_set_summary.csv",
        alignment_set_summary,
        ["dataset", "aligned_participant_count", "alignment_basis"],
    )
    write_csv(
        SUPPORT_DIR / "profile_participant_groups.csv",
        basic_profiles["participant_groups"],
        [
            "source_id",
            "source_label",
            "profile_type",
            "variable",
            "column",
            "field_id",
            "instance",
            "value",
            "value_label",
            "count",
            "percent",
            "non_missing_count",
            "missing_percent",
            "mean",
            "sd",
            "median",
            "q1",
            "q3",
            "min",
            "max",
            "zero_percent",
            "skewness",
            "extreme_z_percent",
        ],
    )
    write_csv(
        SUPPORT_DIR / "profile_disease_targets.csv",
        basic_profiles["disease_targets"],
        [
            "source_id",
            "source_label",
            "profile_type",
            "target_family",
            "target_name",
            "target_type",
            "columns",
            "case_only",
            "participant_count",
            "target_count",
            "percent",
            "summary_json",
        ],
    )
    write_csv(
        SUPPORT_DIR / "profile_feature_distributions.csv",
        basic_profiles["feature_distributions"],
        [
            "source_id",
            "source_label",
            "theme",
            "group_id",
            "feature_label",
            "profile_type",
            "value_type",
            "column_count",
            "non_missing_count",
            "missing_percent",
            "mean",
            "sd",
            "median",
            "q1",
            "q3",
            "min",
            "max",
            "zero_percent",
            "skewness",
            "extreme_z_percent",
            "observed_levels",
            "summary_basis",
            "unique_overflow",
        ],
    )
    write_csv(
        SUPPORT_DIR / "profile_repeatability_cv_icc.csv",
        basic_profiles["repeatability_cv_icc"],
        [
            "source_id",
            "source_label",
            "theme",
            "group_id",
            "feature_label",
            "visit_pair",
            "participant_count_ge2_visits",
            "observation_count",
            "within_person_cv",
            "between_person_cv",
            "icc_approx",
            "visit_pair_count",
            "visit_pair_correlation",
        ],
    )
    write_csv(
        SUPPORT_DIR / "profile_missingness_patterns.csv",
        basic_profiles["missingness_patterns"],
        [
            "source_id",
            "source_label",
            "profile_type",
            "group_variable",
            "group_value",
            "group_value_label",
            "theme",
            "participant_count",
            "with_data_count",
            "missing_count",
            "present_percent",
        ],
    )

    basic_profiles_for_app = {
        key: basic_profiles[key]
        for key in ["participant_groups", "disease_targets", "missingness_patterns"]
    }

    payload = {
        "summary": summary,
        "global_participants": global_participants,
        "inventory": inventory,
        "modality_counts": modality_counts,
        "modality_intersections": modality_intersections,
        "dataset_alignment": dataset_alignment,
        "bridge_alignment_summary": bridge_alignment_summary,
        "alignment_set_summary": alignment_set_summary,
        "semantic_group_coverage": group_coverage_rows,
        "semantic_node_coverage": node_coverage_rows,
        "node_coverage_index": node_coverage_index(node_coverage_rows),
        "id_namespace_interpretation": namespace_rows,
        "requested_data_status": requested_rows,
        "target_catalog": target_rows,
        "target_predictor_availability": target_predictor_rows,
        "basic_profiles": basic_profiles_for_app,
        "tree_structure_audit_summary": tree_audit_rows,
        "groups": groups,
        "tree": tree_payload["tree"],
        "source_descriptions": SOURCE_DESCRIPTIONS,
    }
    (META_DIR / "semantic_feature_tree.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (SUPPORT_DIR / "semantic_feature_tree.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    build_dataset_tree_svg(inventory, groups)
    build_storage_pattern_svg()
    stale_modality_svg = SUPPORT_DIR / "modality_coverage_figure.svg"
    if stale_modality_svg.exists():
        stale_modality_svg.unlink()
    build_prediction_target_overview_svg(target_rows)
    build_missing_requested_svg(requested_rows)
    build_html(
        payload,
        requested_rows,
        target_rows,
        target_predictor_rows,
        namespace_rows,
        modality_counts,
        dataset_alignment,
        bridge_alignment_summary,
        alignment_set_summary,
    )

    validation = validate(groups, columns, tree_payload)
    (SUPPORT_DIR / "README.md").write_text(
        "# Interactive UKB Dataset Visualization\n\n"
        "Open `index.html` in a browser. The HTML is self-contained for interactive viewing and includes aggregate coverage plus two non-ID example values per sampled feature group for internal lab use.\n\n"
        "## Moving or copying\n\n"
        "You can copy this whole `interactive_dataset_visualization` folder to another drive or computer and open the outer `index.html` there. The app does not depend on local absolute paths. Support files live in `supporting_files/`; keep that folder if you want the exported CSV/JSON tables, static SVG figures, and this README. Tree audit/source files live in `source_files/`. The HTML itself embeds the data needed for the browser view, so moving support/source files does not change the visualization. The only absolute links are external UKB Showcase web URLs used as references.\n\n"
        "## Key outputs\n\n"
        "- `index.html`: interactive report appendix.\n"
        "- `semantic_feature_groups.csv`: cognitive feature groups with raw-column traceability.\n"
        "- `semantic_group_coverage.csv`: exact group coverage overall and by UKB visit instance.\n"
        "- `semantic_node_coverage.csv`: parent-node coverage for the tree view.\n"
        "- `tree_structure_review_candidates.csv`: deterministic review queue for tree/search taxonomy corrections.\n"
        "- `tree_structure_audit_summary.csv`: tree/search audit metrics and biological-domain counts.\n"
        "- `semantic_feature_tree.json`: data payload used by the HTML.\n"
        "- `target_catalog.csv`: exact target candidates and phenotype groups.\n"
        "- `target_predictor_availability.csv`: metadata-derived predictor availability per target.\n"
        "- `dataset_alignment_bridge_summary.csv`: bridge-file uniqueness checks and local-source overlaps.\n"
        "- `dataset_alignment_set_summary.csv`: bridge-aware aligned participant counts by source.\n"
        "- `id_namespace_interpretation.csv`: report-facing ID namespace and bridge interpretation.\n"
        "- `profile_participant_groups.csv`, `profile_disease_targets.csv`, `profile_missingness_patterns.csv`: aggregate Basic Profiling tab inputs. Feature distribution and repeatability CSVs are generated as side outputs but not shown in the app.\n"
        "- `dataset_tree_overview.svg`, `storage_pattern_explanation.svg`, `prediction_target_overview.svg`, `missing_requested_data_types.svg`: static report figures.\n"
        "- `validation_report.json`: mapping and paired-array validation checks.\n\n"
        "For tree review, use `../source_files/` from this README location. It contains copied inputs, generated semantic outputs, coverage tables, audit files, and the generator script snapshot.\n\n"
        f"Validation status: `{validation['status']}`; mapped raw columns: {validation['raw_columns_mapped']:,}.\n",
        encoding="utf-8",
    )
    export_structure_source_files(validation)
    print(f"Wrote interactive visualization to {OUT_DIR}")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
