from app.checks import WARNING_CANONICAL, check_warning

GOOD = ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
        "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
        "operate machinery, and may cause health problems.")

def test_canonical_matches_cfr_16_21():
    assert WARNING_CANONICAL == GOOD

def test_exact_warning_matches():
    r = check_warning("SUNSET ALE 5.9% ALC/VOL\n" + GOOD)
    assert r.status == "MATCH"

def test_ocr_linebreaks_and_double_spaces_still_match():
    mangled = GOOD.replace("Surgeon General,", "Surgeon  General,\n")
    assert check_warning(mangled).status == "MATCH"

def test_title_case_prefix_is_review_not_match():
    # Jenny's real rejection: right words, wrong capitalization (16.22 violation)
    r = check_warning(GOOD.replace("GOVERNMENT WARNING:", "Government Warning:"))
    assert r.status == "REVIEW"
    assert "capital" in r.detail.lower()

def test_reworded_warning_is_mismatch():
    r = check_warning(GOOD.replace("birth defects", "health issues"))
    assert r.status == "MISMATCH"

def test_absent_warning_is_missing():
    assert check_warning("SUNSET ALE 5.9% ALC/VOL 12 FL OZ").status == "MISSING"
