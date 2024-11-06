from django.contrib import admin
from django.urls import path, include
from Audio_Text import views


urlpatterns = [
    path('', views.home, name='home'),   # Définit la page d'accueil à l'URL racine
    path('home/', views.home, name='home'),
    path('TSS', views.translate_text, name='TSS'),
    path('RSS', views.summary_text, name='RSS'),
    path('DSS', views.doc_summary, name='DSS'),
    path('admin/', admin.site.urls),
]

