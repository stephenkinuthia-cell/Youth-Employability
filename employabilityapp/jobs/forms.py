from django import forms


class JobPostingForm(forms.Form):
    title = forms.CharField(max_length=120)
    industry = forms.CharField(max_length=80)
    location = forms.CharField(max_length=120)
    description = forms.CharField(widget=forms.Textarea(attrs={"rows": 4}))
    required_skills = forms.CharField(help_text="Comma-separated skills")
    minimum_score = forms.FloatField(min_value=0, max_value=100, initial=55)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({"class": "form-control"})
