from django import forms

from employabilityapp.platform_choices import (
    DEGREE_LEVEL_CHOICES,
    EXPERIENCE_LEVEL_CHOICES,
    FIELD_OF_STUDY_CHOICES,
    INDUSTRY_CHOICES,
    REGION_CHOICES,
    SKILL_CHOICES,
)


class EmployabilityPredictionForm(forms.Form):
    full_name = forms.CharField(max_length=120)
    email = forms.EmailField()
    degree_level = forms.ChoiceField(choices=DEGREE_LEVEL_CHOICES)
    field_of_study = forms.ChoiceField(choices=FIELD_OF_STUDY_CHOICES)
    region = forms.ChoiceField(choices=REGION_CHOICES, initial="Middle East & Africa")
    graduation_year = forms.IntegerField(min_value=2010, max_value=2035, initial=2024)
    experience_level = forms.ChoiceField(choices=EXPERIENCE_LEVEL_CHOICES)
    desired_industry = forms.ChoiceField(choices=INDUSTRY_CHOICES, required=False)
    skills = forms.MultipleChoiceField(
        choices=SKILL_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        help_text="Select the skills the candidate already has.",
    )
    average_starting_salary = forms.IntegerField(min_value=0, initial=40000)
    skill_demand_score = forms.IntegerField(min_value=1, max_value=100, initial=65)
    employer_reputation_score = forms.IntegerField(min_value=1, max_value=100, initial=60)
    remote_work_availability = forms.IntegerField(min_value=1, max_value=100, initial=55)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            if isinstance(field.widget, forms.CheckboxSelectMultiple):
                continue
            field.widget.attrs.update({"class": "form-control"})
