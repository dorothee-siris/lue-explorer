from __future__ import annotations


# Project years (inclusive)
YEAR_START = 2019
YEAR_END = 2023


FIELDS_TO_DOMAIN = {
# 1) PHYSICAL SCIENCES
"Chemical Engineering": "Physical Sciences",
"Chemistry": "Physical Sciences",
"Computer Science": "Physical Sciences",
"Earth and Planetary Sciences": "Physical Sciences",
"Energy": "Physical Sciences",
"Engineering": "Physical Sciences",
"Materials Science": "Physical Sciences",
"Mathematics": "Physical Sciences",
"Physics and Astronomy": "Physical Sciences",
# 2) LIFE SCIENCES
"Agricultural and Biological Sciences": "Life Sciences",
"Biochemistry, Genetics and Molecular Biology": "Life Sciences",
"Environmental Science": "Life Sciences",
"Immunology and Microbiology": "Life Sciences",
# 3) HEALTH SCIENCES
"Dentistry": "Health Sciences",
"Health Professions": "Health Sciences",
"Medicine": "Health Sciences",
"Neuroscience": "Health Sciences",
"Nursing": "Health Sciences",
"Pharmacology, Toxicology and Pharmaceutics": "Health Sciences",
"Veterinary": "Health Sciences",
# 4) SOCIAL SCIENCES
"Arts and Humanities": "Social Sciences",
"Business, Management and Accounting": "Social Sciences",
"Decision Sciences": "Social Sciences",
"Economics, Econometrics and Finance": "Social Sciences",
"Psychology": "Social Sciences",
"Social Sciences": "Social Sciences",
}


DOMAIN_NAMES = ["Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences", "Other"]


DOMAIN_COLORS = {
"Health Sciences": "#F85C32",
"Life Sciences": "#0CA750",
"Physical Sciences": "#8190FF",
"Social Sciences": "#FFCB3A",
"Other": "#7f7f7f",
}


# Links
ROR_URL = "https://ror.org/{ror}"
OPENALEX_INSTITUTION_QUERY = "https://api.openalex.org/institutions?filter=ror:{ror}" # API query as a stable fallback
OPENALEX_WORKS_FOR_ROR = "https://api.openalex.org/works?filter=institutions.ror:{ror}" # direct works listing