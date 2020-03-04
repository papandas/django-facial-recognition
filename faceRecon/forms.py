from django import forms
from django.contrib.auth.models import User

class UserSelection(forms.Form):
    selected_user = forms.ModelChoiceField(label='Select User', queryset=User.objects.all(), required=True)