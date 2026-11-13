from sec_llm_project.sec_intel.retrieval.entity import EntityResolver

R = EntityResolver({
    "MO": "ALTRIA GROUP, INC.", "IT": "GARTNER INC", "CL": "COLGATE PALMOLIVE CO",
    "D": "DOMINION ENERGY, INC", "UNP": "UNION PACIFIC CORP", "INTC": "INTEL CORP",
    "GOOG": "Alphabet Inc.", "GOOGL": "Alphabet Inc.", "AVY": "AVERY DENNISON CORP",
    "MMM": "3M CO", "VRTX": "VERTEX PHARMACEUTICALS INC / MA", "JBHT": "HUNT J B TRANSPORT SERVICES INC",
})


def t(q):
    return R.resolve(q).tickers


def test_ticker_and_name():
    assert t("MO, Altria Group's cybersecurity risk mitigation") == ["MO"]
    assert t("Colgate-Palmolive (CL) Brazilian tax impact") == ["CL"]


def test_ambiguous_words_need_a_ticker_cue():
    assert t("Impact of IT risk mgmt on JBHT cap allocation") == ["JBHT"]
    assert t("Gartner (IT) research revenue") == ["IT"]
    assert t("MDLZ capital allocation for R&D & promos") == []  # R&D is not ticker D


def test_lowercase_words_are_not_company_names():
    assert t("labor costs of union coverage at JBHT") == ["JBHT"]
    assert t("VRTX vulnerability scanning, threat intel") == ["VRTX"]


def test_ranking_and_share_classes():
    assert t("presence of 3M in AVY's competitive landscape")[0] == "AVY"
    assert len(t("Alphabet's capex plans")) == 1


def test_unknown_company_resolves_to_nothing():
    assert t("What launch-failure risks does SpaceX disclose?") == []
