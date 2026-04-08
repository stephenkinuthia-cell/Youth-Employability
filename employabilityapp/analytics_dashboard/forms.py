from django import forms


class SupportMessageForm(forms.Form):
    subject = forms.CharField(max_length=150)
    contact_email = forms.EmailField()
    message = forms.CharField(widget=forms.Textarea(attrs={"rows": 5}))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({"class": "form-control"})
