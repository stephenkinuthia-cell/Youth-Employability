DEGREE_LEVEL_CHOICES = [
    ("Bachelor", "Bachelor"),
    ("Master", "Master"),
    ("PhD", "PhD"),
]

FIELD_OF_STUDY_CHOICES = [
    ("Business & Finance", "Business & Finance"),
    ("Computer Science", "Computer Science"),
    ("Data Science & AI", "Data Science & AI"),
    ("Engineering", "Engineering"),
    ("Healthcare & Medicine", "Healthcare & Medicine"),
    ("Natural Sciences", "Natural Sciences"),
    ("Social Sciences", "Social Sciences"),
]

REGION_CHOICES = [
    ("Asia-Pacific", "Asia-Pacific"),
    ("Europe", "Europe"),
    ("Latin America", "Latin America"),
    ("Middle East & Africa", "Middle East & Africa"),
    ("North America", "North America"),
]

EXPERIENCE_LEVEL_CHOICES = [
    ("entry", "Entry level"),
    ("mid", "Mid level"),
    ("senior", "Senior level"),
]

SKILL_CHOICES = [
    ("Communication", "Communication"),
    ("Cloud Computing", "Cloud Computing"),
    ("Cybersecurity", "Cybersecurity"),
    ("Data Analysis", "Data Analysis"),
    ("Data Visualization", "Data Visualization"),
    ("Excel", "Excel"),
    ("Laboratory Skills", "Laboratory Skills"),
    ("Machine Learning", "Machine Learning"),
    ("NLP", "NLP"),
    ("Patient Care", "Patient Care"),
    ("Project Management", "Project Management"),
    ("Public Policy", "Public Policy"),
    ("Python", "Python"),
    ("Qualitative Research", "Qualitative Research"),
    ("R", "R"),
    ("Scientific Writing", "Scientific Writing"),
    ("SQL", "SQL"),
    ("Statistical Analysis", "Statistical Analysis"),
    ("Strategic Planning", "Strategic Planning"),
]

INDUSTRY_CHOICES = [
    ("Finance/Consulting", "Finance/Consulting"),
    ("Government/NGO", "Government/NGO"),
    ("Healthcare", "Healthcare"),
    ("Manufacturing/Construction", "Manufacturing/Construction"),
    ("Research/Academia", "Research/Academia"),
    ("Technology", "Technology"),
    ("Technology/Consulting", "Technology/Consulting"),
]

FIELD_TO_INDUSTRY = {
    "Business & Finance": "Finance/Consulting",
    "Computer Science": "Technology",
    "Data Science & AI": "Technology/Consulting",
    "Engineering": "Manufacturing/Construction",
    "Healthcare & Medicine": "Healthcare",
    "Natural Sciences": "Research/Academia",
    "Social Sciences": "Government/NGO",
}

FIELD_TO_ROLE = {
    "Business & Finance": "Financial Analyst",
    "Computer Science": "Software Engineer",
    "Data Science & AI": "Data Scientist",
    "Engineering": "Mechanical Engineer",
    "Healthcare & Medicine": "Public Health Specialist",
    "Natural Sciences": "Research Scientist",
    "Social Sciences": "Policy Analyst",
}

FIELD_TO_SKILLS = {
    "Business & Finance": ["Communication", "Excel", "Project Management", "SQL"],
    "Computer Science": ["Python", "Cloud Computing", "SQL", "Cybersecurity"],
    "Data Science & AI": ["Python", "Data Analysis", "Machine Learning", "Data Visualization"],
    "Engineering": ["Project Management", "Data Analysis", "Strategic Planning", "Communication"],
    "Healthcare & Medicine": ["Patient Care", "Communication", "Data Analysis", "Scientific Writing"],
    "Natural Sciences": ["Laboratory Skills", "Scientific Writing", "Data Analysis", "Statistical Analysis"],
    "Social Sciences": ["Communication", "Qualitative Research", "Public Policy", "Project Management"],
}
