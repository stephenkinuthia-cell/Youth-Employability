from django import forms
from django.conf import settings
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from employabilityapp.platform_choices import (
    DEGREE_LEVEL_CHOICES,
    EXPERIENCE_LEVEL_CHOICES,
    FIELD_OF_STUDY_CHOICES,
    INDUSTRY_CHOICES,
)


class StyledAuthenticationForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({"class": "form-control"})


class RegistrationForm(UserCreationForm):
    ROLE_CHOICES = [
        ("student", "User / Graduate"),
        ("employer", "Employer"),
        ("admin", "Administrator"),
    ]

    first_name = forms.CharField(max_length=150)
    last_name = forms.CharField(max_length=150)
    email = forms.EmailField()
    role = forms.ChoiceField(choices=ROLE_CHOICES)
    phone_number = forms.CharField(max_length=30)
    location = forms.CharField(max_length=120)
    bio = forms.CharField(widget=forms.Textarea(attrs={"rows": 3}), required=False)
    admin_code = forms.CharField(max_length=30, required=False, help_text="Required only for administrator accounts.")

    degree_level = forms.ChoiceField(choices=DEGREE_LEVEL_CHOICES, required=False)
    field_of_study = forms.ChoiceField(choices=FIELD_OF_STUDY_CHOICES, required=False)
    graduation_year = forms.IntegerField(min_value=2010, max_value=2035, required=False)
    experience_level = forms.ChoiceField(choices=EXPERIENCE_LEVEL_CHOICES, required=False)
    skills_text = forms.CharField(required=False, help_text="Comma-separated skills.")
    career_goal = forms.CharField(max_length=150, required=False)

    company_name = forms.CharField(max_length=150, required=False)
    industry = forms.ChoiceField(choices=INDUSTRY_CHOICES, required=False)
    website = forms.URLField(required=False)
    company_size = forms.CharField(max_length=50, required=False)
    contact_person = forms.CharField(max_length=120, required=False)
    job_title = forms.CharField(max_length=120, required=False)
    job_location = forms.CharField(max_length=120, required=False)
    job_description = forms.CharField(widget=forms.Textarea(attrs={"rows": 3}), required=False)
    job_required_skills = forms.CharField(required=False, help_text="Comma-separated skills for the first listed job.")
    minimum_score = forms.FloatField(min_value=0, max_value=100, required=False, initial=55)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "first_name", "last_name", "email", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            if isinstance(field.widget, forms.CheckboxInput):
                continue
            field.widget.attrs.update({"class": "form-control"})

    def clean_email(self):
        email = self.cleaned_data["email"]
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with that email already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        role = cleaned_data.get("role")

        if role == "admin":
            if cleaned_data.get("admin_code") != settings.ADMIN_SIGNUP_CODE:
                self.add_error("admin_code", "Invalid administrator sign-up code.")

        if role == "student":
            for field_name in ["degree_level", "field_of_study", "graduation_year", "experience_level"]:
                if not cleaned_data.get(field_name):
                    self.add_error(field_name, "This field is required for user accounts.")

        if role == "employer":
            for field_name in ["company_name", "industry", "contact_person", "job_title", "job_location", "job_description"]:
                if not cleaned_data.get(field_name):
                    self.add_error(field_name, "This field is required for employer accounts.")

        return cleaned_data
