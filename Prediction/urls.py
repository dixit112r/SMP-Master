from django.urls import path
from . import views

app_name = 'smp'
urlpatterns = [
    path('', views.index, name="index"),
    path('google', views.google, name="google"),
    path('twitter', views.twitter, name="twitter"),
    path('apple', views.apple, name="apple"),
    path('microsoft', views.microsoft, name="microsoft"),
    path('getDataFromWeb', views.getDataFromWeb, name="getdataFromWeb"),
    path('createModel', views.createModel, name="createModel"),
]