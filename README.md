# TTB Label Check

A prototype that verifies alcohol beverage label images against COLA application data:
brand name, alcohol content, and the government health warning statement required by
27 CFR Part 16. Built for the Treasury IT Specialist (AI) take-home assessment.

**Live prototype:** https://ttb.chainward.ai — self-hosted on a 4-node K3s
cluster I operate (2 replicas) behind a Cloudflare tunnel; deployment manifests in
`deploy/k8s.yaml`. (`chainward.ai` is simply a domain I own used for hosting — this
prototype is standalone and unrelated to anything else on it.)

Single-label checks return in well under a second on typical images (measured 0.2–2 s
end-to-end); batch mode accepts a CSV of applications plus hundreds of label images at
once. The tool never auto-rejects: every finding is MATCH / REVIEW / MISMATCH / MISSING,
and the reviewing agent makes the final call.

## Quickstart (Docker — recommended)

```bash
docker build -t ttb-label-check .
docker run -p 8080:8080 ttb-label-check
# open http://localhost:8080
```

## Quickstart (local)

```bash
brew install tesseract          # or: apt-get install tesseract-ocr
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/make_samples.py    # generate demo labels
.venv/bin/uvicorn app.main:app --port 8080
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q        # 31 tests, no network required
```

## Batch mode

Upload one CSV plus the label images. Images match CSV rows **by filename**:
`COLA-12345.png` pairs with the row whose `application_id` is `COLA-12345`.

```csv
application_id,brand_name,abv
COLA-12345,Sunset Ale,5.9
COLA-12346,Stone's Throw,7.1
```

Results are sorted problems-first so agents triage top-down. Try it with
`samples/batch.csv` and the three PNGs in `samples/`.

## Optional AI assist

By default the app performs all processing locally (Tesseract OCR + OpenCV) and makes
**zero outbound network calls** — a hard requirement from the IT stakeholder interview
(agency firewall blocks cloud ML endpoints). If an `ANTHROPIC_API_KEY` environment
variable is configured, the UI offers an opt-in "AI assist" checkbox that sends
hard-to-read images (glare, angles) to Claude's vision API for transcription when local
OCR comes up near-empty. Without a key, the feature is hidden and the code path is
never taken.

## Tools used (disclosure)

- **Development:** This prototype was built with AI-assisted development (Anthropic's
  Claude, driven through Claude Code) under human direction and review: requirements
  extraction from the stakeholder interviews, test-first implementation, and this
  documentation. All regulatory text was verified against the eCFR directly.
- **Runtime:** Tesseract OCR + OpenCV locally; optional Claude vision API (documented
  above); FastAPI/Jinja2; no database, no persistence, nothing stored.

See `docs/APPROACH.md` for the full design rationale, assumptions, and limitations.
