from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from employabilityapp.platform_choices import FIELD_TO_INDUSTRY, FIELD_TO_ROLE, FIELD_TO_SKILLS


@dataclass
class PredictionResult:
    employability_score: float
    risk_category: str
    model_status: str
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[dict[str, Any]]
    career_paths: list[dict[str, Any]]
    report_next_steps: list[str]
    report_markdown: str


class PredictionService:
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.ml_root = self.repo_root / "employability-ai-system"
        self.reference_csv = self.repo_root / "model_ready_data.csv"
        self.training_features_csv = self.ml_root / "data" / "processed" / "final_features.csv"
        self.raw_reference_csv = self.repo_root / "global_graduate_employability_index.csv"

    def predict_profile(self, cleaned_data: dict[str, Any]) -> PredictionResult:
        feature_frame = self._build_feature_frame(cleaned_data)
        score, model_status = self._predict_score(cleaned_data, feature_frame)
        risk_category = self._risk_from_score(score)
        strengths, weaknesses = self._summarize_profile(cleaned_data)
        recommendations = self._build_recommendations(cleaned_data, feature_frame, score)
        career_paths = self._build_career_paths(cleaned_data, score)
        next_steps = [rec["title"] for rec in recommendations[:3]]
        report_markdown = self._build_report(cleaned_data, score, risk_category, strengths, weaknesses, recommendations, career_paths)

        return PredictionResult(
            employability_score=score,
            risk_category=risk_category,
            model_status=model_status,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            career_paths=career_paths,
            report_next_steps=next_steps,
            report_markdown=report_markdown,
        )

    def _predict_score(self, cleaned_data: dict[str, Any], feature_frame: pd.DataFrame) -> tuple[float, str]:
        heuristic_score = self._heuristic_score(cleaned_data)
        model = self._load_model()
        if model is not None:
            try:
                probability = float(model.predict_proba(feature_frame)[0][1])
                calibrated_score = (probability * 100 * 0.65) + (heuristic_score * 0.35)
                return round(calibrated_score, 2), "trained_model_loaded"
            except Exception:
                pass

        return heuristic_score, "demo_fallback_no_saved_model"

    def _heuristic_score(self, cleaned_data: dict[str, Any]) -> float:
        degree_bonus = {"Bachelor": 8, "Master": 12, "PhD": 15}.get(cleaned_data["degree_level"], 8)
        experience_bonus = {"entry": 8, "mid": 13, "senior": 18}.get(cleaned_data["experience_level"], 8)
        field_bonus = {
            "Computer Science": 12,
            "Data Science & AI": 14,
            "Engineering": 10,
            "Healthcare & Medicine": 9,
            "Business & Finance": 8,
            "Natural Sciences": 7,
            "Social Sciences": 7,
        }.get(cleaned_data["field_of_study"], 7)
        skill_bonus = min(len(cleaned_data["skills"]) * 5, 20)
        demand_bonus = cleaned_data["skill_demand_score"] * 0.18
        reputation_bonus = cleaned_data["employer_reputation_score"] * 0.14
        remote_bonus = cleaned_data["remote_work_availability"] * 0.06
        salary_bonus = min(cleaned_data["average_starting_salary"] / 8000, 12)

        score = 15 + degree_bonus + experience_bonus + field_bonus + skill_bonus
        score += demand_bonus + reputation_bonus + remote_bonus + salary_bonus
        return round(max(25, min(score, 96)), 2)

    def _build_feature_frame(self, cleaned_data: dict[str, Any]) -> pd.DataFrame:
        current_year = date.today().year
        reference_profile = self._select_reference_profile(cleaned_data)
        skills = list(cleaned_data["skills"])[:3]
        while len(skills) < 3:
            skills.append("Communication")

        years_since_graduation = max(current_year - cleaned_data["graduation_year"], 0)
        desired_industry = cleaned_data.get("desired_industry") or reference_profile.get("Top_Industry") or FIELD_TO_INDUSTRY[cleaned_data["field_of_study"]]
        job_role = reference_profile.get("Job_Role") or FIELD_TO_ROLE[cleaned_data["field_of_study"]]
        career_stage = "Early" if years_since_graduation <= 2 else "Mid" if years_since_graduation <= 5 else "Late"
        base_employment_6 = float(reference_profile.get("Employment_Rate_6_Months (%)", 68))
        employment_rate_6 = min(95, max(35, base_employment_6 + (len(cleaned_data["skills"]) - 3) * 2 + years_since_graduation * 1.5))
        country = reference_profile.get("Country") or self._default_country_for_region(cleaned_data["region"])

        raw_values = {
            "Country": country,
            "University_Name": reference_profile.get("University_Name", "Platform Candidate"),
            "Graduation_Year": cleaned_data["graduation_year"],
            "Employment_Rate_6_Months (%)": employment_rate_6,
            "Employment_Rate_12_Months (%)": 0,
            "Average_Starting_Salary_USD": cleaned_data["average_starting_salary"],
            "Job_Role": job_role,
            "Skill_1": skills[0],
            "Skill_2": skills[1],
            "Skill_3": skills[2],
            "Skill_Demand_Score (1–100)": cleaned_data["skill_demand_score"],
            "Remote_Work_Availability (%)": cleaned_data["remote_work_availability"],
            "Employer_Reputation_Score (1–100)": cleaned_data["employer_reputation_score"],
            "Region": cleaned_data["region"],
            "Degree_Level": cleaned_data["degree_level"],
            "Field_of_Study": cleaned_data["field_of_study"],
            "Top_Industry": desired_industry,
            "Year": current_year,
            "Demand_x_Reputation": cleaned_data["skill_demand_score"] * cleaned_data["employer_reputation_score"] / 100,
            "Demand_x_Remote": cleaned_data["skill_demand_score"] * cleaned_data["remote_work_availability"] / 100,
            "Reputation_x_Remote": cleaned_data["employer_reputation_score"] * cleaned_data["remote_work_availability"] / 100,
            "Unique_Skills_Count": len(set(cleaned_data["skills"])),
            "Years_Since_Graduation": years_since_graduation,
            "Career_Stage": career_stage,
            "Salary_Per_Year_Experience": cleaned_data["average_starting_salary"] if years_since_graduation == 0 else cleaned_data["average_starting_salary"] / years_since_graduation,
            "Is_US": 0,
        }

        aligned = {column: 0 for column in self._get_reference_columns()}
        for column, value in raw_values.items():
            if column in aligned:
                aligned[column] = value

        for prefix, value in {
            "Region": cleaned_data["region"],
            "Degree_Level": cleaned_data["degree_level"],
            "Field_of_Study": cleaned_data["field_of_study"],
            "Top_Industry": desired_industry,
            "Country": country,
            "Job_Role": job_role,
            "Career_Stage": career_stage,
        }.items():
            encoded_column = f"{prefix}_{value}"
            if encoded_column in aligned:
                aligned[encoded_column] = 1

        for index, skill in enumerate(skills, start=1):
            encoded_column = f"Skill_{index}_{skill}"
            if encoded_column in aligned:
                aligned[encoded_column] = 1

        frame = pd.DataFrame([aligned])
        model = self._load_model()
        if model is not None and hasattr(model, "feature_names_in_"):
            frame = frame.reindex(columns=list(model.feature_names_in_), fill_value=0)
        return frame

    def _build_recommendations(self, cleaned_data: dict[str, Any], feature_frame: pd.DataFrame, score: float) -> list[dict[str, Any]]:
        target_skills = FIELD_TO_SKILLS[cleaned_data["field_of_study"]]
        missing_skills = [skill for skill in target_skills if skill not in cleaned_data["skills"]]
        recommendations: list[dict[str, Any]] = []

        for skill in missing_skills[:2]:
            recommendations.append(
                {
                    "title": f"Build {skill}",
                    "category": "Skill",
                    "rationale": f"{skill} is commonly associated with stronger outcomes in {cleaned_data['field_of_study']}.",
                    "priority": "High",
                    "improvement_potential": 8.5,
                    "course_name": f"{skill} learning path",
                    "course_url": f"https://www.coursera.org/search?query={skill.replace(' ', '%20')}",
                }
            )

        if cleaned_data["skill_demand_score"] < 60:
            recommendations.append(
                {
                    "title": "Strengthen market-ready projects",
                    "category": "Portfolio",
                    "rationale": "The current profile suggests stronger portfolio evidence would improve job-market readiness.",
                    "priority": "High",
                    "improvement_potential": 7.0,
                    "course_name": "Project portfolio guide",
                    "course_url": "https://www.coursera.org/search?query=portfolio%20projects",
                }
            )

        if cleaned_data["employer_reputation_score"] < 60:
            recommendations.append(
                {
                    "title": "Pursue certifications and internships",
                    "category": "Experience",
                    "rationale": "Recognized credentials can offset lower employer reputation exposure.",
                    "priority": "Medium",
                    "improvement_potential": 6.0,
                    "course_name": "Career certificate programs",
                    "course_url": "https://www.coursera.org/career-academy",
                }
            )

        if score < 55 and "Communication" not in cleaned_data["skills"]:
            recommendations.append(
                {
                    "title": "Improve communication skills",
                    "category": "Soft Skill",
                    "rationale": "Communication is a strong employability multiplier across all fields.",
                    "priority": "High",
                    "improvement_potential": 9.0,
                    "course_name": "Communication foundations",
                    "course_url": "https://www.edx.org/learn/communication",
                }
            )

        return recommendations[:4]

    def _build_career_paths(self, cleaned_data: dict[str, Any], score: float) -> list[dict[str, Any]]:
        field = cleaned_data["field_of_study"]
        skills = set(cleaned_data["skills"])
        suggestions = {
            "Computer Science": ["Software Engineer", "Cloud Engineer", "Cybersecurity Analyst"],
            "Data Science & AI": ["Data Analyst", "Data Scientist", "Machine Learning Engineer"],
            "Engineering": ["Process Engineer", "Project Engineer", "Operations Analyst"],
            "Healthcare & Medicine": ["Clinical Data Analyst", "Public Health Specialist", "Healthcare Operations Officer"],
            "Business & Finance": ["Business Analyst", "Financial Analyst", "Operations Associate"],
            "Natural Sciences": ["Research Assistant", "Laboratory Analyst", "Quality Assurance Officer"],
            "Social Sciences": ["Policy Analyst", "Research Associate", "Program Officer"],
        }

        results = []
        for title in suggestions.get(field, ["Graduate Trainee", "Analyst", "Coordinator"]):
            boost = 4 if any(skill in title for skill in ["Data", "Research"]) and ("Data Analysis" in skills or "Qualitative Research" in skills) else 0
            results.append(
                {
                    "title": title,
                    "match_score": round(min(score + boost, 99), 1),
                    "summary": f"{title} aligns with {field}, {cleaned_data['experience_level']} experience, and the candidate's current skill mix.",
                }
            )
        return results[:3]

    def _summarize_profile(self, cleaned_data: dict[str, Any]) -> tuple[list[str], list[str]]:
        strengths = []
        weaknesses = []

        if len(cleaned_data["skills"]) >= 3:
            strengths.append("The profile already shows a healthy spread of core skills.")
        if cleaned_data["skill_demand_score"] >= 70:
            strengths.append("Selected skills map well to current market demand.")
        if cleaned_data["employer_reputation_score"] >= 65:
            strengths.append("The target opportunities appear connected to credible employers or sectors.")
        if cleaned_data["remote_work_availability"] >= 60:
            strengths.append("Remote-friendly roles increase the addressable job market.")

        if len(cleaned_data["skills"]) < 3:
            weaknesses.append("The candidate needs a broader skill portfolio to stay competitive.")
        if cleaned_data["skill_demand_score"] < 60:
            weaknesses.append("The current skills do not yet signal enough in-demand capability.")
        if cleaned_data["employer_reputation_score"] < 60:
            weaknesses.append("Employer or internship exposure could be strengthened.")
        if cleaned_data["remote_work_availability"] < 50:
            weaknesses.append("The profile may be competing in a narrower opportunity pool.")

        if not strengths:
            strengths.append("The profile has a usable baseline that can be improved with targeted upskilling.")
        if not weaknesses:
            weaknesses.append("The next gains will come from deeper specialization and visible project work.")

        return strengths, weaknesses

    def _build_report(
        self,
        cleaned_data: dict[str, Any],
        score: float,
        risk_category: str,
        strengths: list[str],
        weaknesses: list[str],
        recommendations: list[dict[str, Any]],
        career_paths: list[dict[str, Any]],
    ) -> str:
        recommendation_lines = "\n".join(f"- {item['title']}: {item['rationale']}" for item in recommendations[:3])
        career_lines = "\n".join(f"- {item['title']} ({item['match_score']}% match)" for item in career_paths[:3])
        strengths_lines = "\n".join(f"- {item}" for item in strengths)
        weaknesses_lines = "\n".join(f"- {item}" for item in weaknesses)

        return (
            f"# AI Employability Feedback Report\n\n"
            f"Candidate: {cleaned_data['full_name']}\n\n"
            f"Predicted employability score: {score:.2f}%\n"
            f"Risk category: {risk_category}\n\n"
            f"## Strengths\n{strengths_lines}\n\n"
            f"## Weaknesses\n{weaknesses_lines}\n\n"
            f"## Recommended next steps\n{recommendation_lines}\n\n"
            f"## Suggested career paths\n{career_lines}\n"
        )

    def _risk_from_score(self, score: float) -> str:
        if score >= 75:
            return "Low"
        if score >= 55:
            return "Medium"
        return "High"

    @lru_cache(maxsize=1)
    def _load_model(self) -> Any | None:
        try:
            self._bootstrap_ml_imports()
            from src.models.registry import load_best_model

            return load_best_model()
        except Exception:
            return None

    @lru_cache(maxsize=1)
    def _load_reference_dataset(self) -> pd.DataFrame:
        if self.reference_csv.exists():
            return pd.read_csv(self.reference_csv)
        return pd.DataFrame()

    @lru_cache(maxsize=1)
    def _get_reference_columns(self) -> list[str]:
        if self.training_features_csv.exists():
            dataset = pd.read_csv(self.training_features_csv, nrows=1)
            return [
                column
                for column in dataset.columns
                if column not in {"High_Employability", "Employment_Rate_12_Months (%)"}
            ]

        dataset = self._load_reference_dataset()
        if not dataset.empty:
            return [
                column
                for column in dataset.columns
                if column not in {"High_Employability", "Employment_Rate_12_Months (%)"}
            ]
        return [
            "Graduation_Year",
            "Employment_Rate_6_Months (%)",
            "Average_Starting_Salary_USD",
            "Skill_Demand_Score (1–100)",
            "Remote_Work_Availability (%)",
            "Employer_Reputation_Score (1–100)",
            "Year",
            "Unique_Skills_Count",
            "Years_Since_Graduation",
            "Demand_x_Reputation",
            "Demand_x_Remote",
            "Reputation_x_Remote",
            "Is_US",
        ]

    def _bootstrap_ml_imports(self) -> None:
        ml_path = str(self.ml_root)
        if ml_path not in sys.path:
            sys.path.insert(0, ml_path)

    @lru_cache(maxsize=1)
    def _load_raw_reference_dataset(self) -> pd.DataFrame:
        if self.raw_reference_csv.exists():
            return pd.read_csv(self.raw_reference_csv)
        return pd.DataFrame()

    def _select_reference_profile(self, cleaned_data: dict[str, Any]) -> dict[str, Any]:
        dataset = self._load_raw_reference_dataset()
        if dataset.empty:
            return {}

        candidates = dataset.copy()
        for column, value in [
            ("Field_of_Study", cleaned_data["field_of_study"]),
            ("Region", cleaned_data["region"]),
            ("Degree_Level", cleaned_data["degree_level"]),
        ]:
            filtered = candidates[candidates[column] == value]
            if not filtered.empty:
                candidates = filtered

        requested_skills = set(cleaned_data["skills"])
        if requested_skills:
            candidates = candidates.assign(
                skill_overlap=(
                    candidates["Skill_1"].isin(requested_skills).astype(int)
                    + candidates["Skill_2"].isin(requested_skills).astype(int)
                    + candidates["Skill_3"].isin(requested_skills).astype(int)
                ),
                year_distance=(candidates["Graduation_Year"] - cleaned_data["graduation_year"]).abs(),
            ).sort_values(["skill_overlap", "year_distance", "Employment_Rate_6_Months (%)"], ascending=[False, True, False])
        else:
            candidates = candidates.assign(
                year_distance=(candidates["Graduation_Year"] - cleaned_data["graduation_year"]).abs()
            ).sort_values(["year_distance", "Employment_Rate_6_Months (%)"], ascending=[True, False])

        return candidates.iloc[0].to_dict()

    def _default_country_for_region(self, region: str) -> str:
        return {
            "Asia-Pacific": "Singapore",
            "Europe": "Germany",
            "Latin America": "Brazil",
            "Middle East & Africa": "UAE",
            "North America": "USA",
        }.get(region, "USA")
