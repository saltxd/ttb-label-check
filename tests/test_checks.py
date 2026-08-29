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


from app.checks import check_brand, parse_abv, check_abv

def test_brand_exact():
    assert check_brand("Sunset Ale", "SUNSET ALE\nIPA 5.9% ALC/VOL").status == "MATCH"

def test_brand_case_only_difference_is_match_with_note():
    # Dave's STONE'S THROW case: same brand, different case — judgment, not rejection
    r = check_brand("Stone's Throw", "STONE'S THROW BREWING CO")
    assert r.status == "MATCH" and "capitalization" in r.detail.lower()

def test_brand_curly_apostrophe_ocr_variant_is_match():
    assert check_brand("Stone's Throw", "STONE’S THROW").status == "MATCH"

def test_brand_close_but_not_equal_is_review():
    assert check_brand("Sunset Ale", "SUNSET ALES").status == "REVIEW"

def test_brand_absent_is_missing():
    assert check_brand("Sunset Ale", "MOONRISE LAGER 4.5% ALC/VOL").status == "MISSING"

def test_parse_abv_formats():
    assert parse_abv("5.9% ALC/VOL") == 5.9
    assert parse_abv("ALC. 5.9% BY VOL.") == 5.9
    assert parse_abv("ALCOHOL 13% BY VOLUME") == 13.0
    assert parse_abv("12 FL OZ") is None

def test_abv_match_and_mismatch():
    assert check_abv(5.9, "SUNSET ALE 5.9% ALC/VOL").status == "MATCH"
    r = check_abv(5.9, "SUNSET ALE 6.2% ALC/VOL")
    assert r.status == "MISMATCH" and "6.2" in r.detail

def test_abv_absent_is_missing():
    assert check_abv(5.9, "SUNSET ALE 12 FL OZ").status == "MISSING"


from app.checks import verify_label
from app.models import ApplicationData

def test_verify_label_all_match_is_pass():
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9),
                     "SUNSET ALE\n5.9% ALC/VOL\n" + GOOD)
    assert r.overall == "PASS"
    assert set(r.fields) == {"brand_name", "abv", "warning"}

def test_any_problem_is_needs_review_never_autoreject():
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9),
                     "SUNSET ALE\n6.2% ALC/VOL\n" + GOOD)
    assert r.overall == "NEEDS REVIEW"  # Dave (R10): tool advises, agent decides

def test_abv_omitted_from_application_skips_abv_check():
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=None),
                     "SUNSET ALE\n" + GOOD)
    assert "abv" not in r.fields and r.overall == "PASS"
