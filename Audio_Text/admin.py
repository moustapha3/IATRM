from django.contrib import admin
from .models import AudioModel

# Register your models here.
@admin.register(AudioModel)
class AudioModelAdmin(admin.ModelAdmin):
    list_display = ('id', 'audioname', 'audio')  # Champs à afficher dans l'admin
    search_fields = ('audioname',)  # Champ de recherche basé sur 'audioname'
