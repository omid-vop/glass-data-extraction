import os
import json
import time
from pypdf import PdfReader, PdfWriter
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional


class GlassExperiment(BaseModel):
    GS_wt_percent: Optional[str] = None
    exchanging_ion: Optional[str] = None
    exchange_side: Optional[str] = None
    E_field: Optional[str] = None
    reference_DOI: Optional[str] = None
    SiO2: float = 0.0
    P2O5: float = 0.0
    B2O3: float = 0.0
    Al2O3: float = 0.0
    CaO: float = 0.0
    MgO: float = 0.0
    BaO: float = 0.0
    ZnO: float = 0.0
    SnO2: float = 0.0
    PbO: float = 0.0
    ZrO2: float = 0.0
    Li2O: float = 0.0
    Na2O: float = 0.0
    K2O: float = 0.0
    TiO2: float = 0.0
    Tg: Optional[float] = None
    ion_exchange_time_min: Optional[float] = None
    E_field_strength_v_cm: Optional[float] = None
    ion_exchange_temperature_c: Optional[float] = None
    concentration_profile: Optional[str] = None
    DOL_um: Optional[float] = None
    CS_Mpa: Optional[float] = None
    trace: float = 0.0
    trace_notes: Optional[str] = None
    is_standard_process: bool = True
    reported_unit: Optional[str] = None


class ExtractionResult(BaseModel):
    experiments: list[GlassExperiment]


def slice_pdf(input_path: str, start_page: int, end_page: int) -> str:
    """Slices a PDF locally to save API tokens and returns the path to the temp file."""
    reader = PdfReader(input_path)
    writer = PdfWriter()

    start_idx = max(0, start_page - 1)
    end_idx = min(len(reader.pages), end_page)

    for i in range(start_idx, end_idx):
        writer.add_page(reader.pages[i])

    temp_path = "temp_sliced_upload.pdf"
    with open(temp_path, "wb") as f:
        writer.write(f)

    return temp_path


def extract_pdf_multimodal(client, pdf_path, start_page=1, end_page=999):
    """Uploads a sliced PDF, analyzes it using vision, and handles retries."""

    target_pdf = slice_pdf(pdf_path, start_page, end_page)

    try:
        uploaded_pdf = client.files.upload(file=target_pdf)
    except Exception as e:
        if os.path.exists(target_pdf):
            os.remove(target_pdf)
        return None, f"Upload failed: {e}"
    prompt = """
        You are an expert materials informatics researcher and glass scientist specializing in chemical strengthening, ion-exchange kinetics, and stress profile optimization. Extract all unique glass formulations and experimental conditions from the provided document into a structured dataset.

        CRITICAL GRAPH & VISUAL EXTRACTION INSTRUCTIONS (DOL & CS):
        Papers frequently report Depth of Layer (DOL) and Compressive Stress (CS) graphically or via textual descriptions:
        1. SURFACE COMPRESSIVE STRESS (CS, in MPa) [HIGH PRIORITY]: Carefully scan text, tables, and stress/retardation profile charts. 
           - Look for the **y-intercept** or the maximum peak value on the compressive stress vs. depth curve.
           - Look for optical retardation formulas or values ($N$, fringe counts, stress-optical coefficients).
           - **Unit Conversion:** If reported in psi, kpsi, or $10^3$ psi, convert it to MPa ($1 \\text{ MPa} \\approx 145.038 \\text{ psi}$). Do not leave CS blank if a stress curve or max stress value is present in the paper.
        2. DEPTH OF LAYER (DOL, in μm): Look at stress-vs-depth or concentration profiles. Identify the exact depth coordinate where the induced compressive stress curve crosses the neutral axis (stress = 0) or where ion penetration drops to background level. Report in micrometers (μm).
        3. CONCENTRATION PROFILE C(x,t): Visually extract 5 to 10 representative data points mapping penetration depth to concentration from EDS/EPMA profiles. Format as a string of tuples: "[(Depth_1, Conc_1), (Depth_2, Conc_2), ...]".

        STRICT OPERATIONAL RULES:
        1. MULTIPLE EXPERIMENTS: Generate a distinct experimental object for every unique combination of temperature, time, bath composition, or applied electrical field tested.
        2. COMPOSITION & OXIDES: Report base glass compositions accurately. If oxides listed in the schema are missing from the paper, strictly output 0.0 (never null).
        3. TRACE ELEMENTS: Aggregate minor unlisted components or unspecified impurities into the `trace` field and document them clearly in `trace_notes`.
        4. PROCESS ANOMALIES: Set `is_standard_process` to false if the experiment utilizes field-assisted ion exchange (FAIE), mixed-salt baths, or applied mechanical loads.
        5. UNREPORTED PARAMETERS: If physical parameters like Tg, concentration profile, DOL, or CS are genuinely absent from both text and figures, explicitly set them to null.
        6. EXCHANGE SIDE: For float glass, identify "Tin side" or "Air side" if specified. If unmentioned or non-float, default to "Air side".
        """
    max_retries = 4
    extracted_data = []
    error_msg = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[prompt, uploaded_pdf],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractionResult,
                )
            )
            result_dict = json.loads(response.text)
            extracted_data = result_dict.get("experiments", [])
            break
        except Exception as e:
            if attempt < max_retries:
                # Wait longer on each retry to let server traffic subside
                time.sleep(attempt * 6)
            else:
                error_msg = f"Extraction failed after max retries due to server load: {e}"

    try:
        client.files.delete(name=uploaded_pdf.name)
    except Exception:
        pass

    if os.path.exists(target_pdf):
        os.remove(target_pdf)

    return extracted_data, error_msg