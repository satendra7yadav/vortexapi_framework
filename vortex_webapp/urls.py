from django.urls import path
from vortex_webapp import views

urlpatterns = [
    path("vortex_app/train/", views.train, name="train"),
    path("vortex_app/home/", views.home, name="home"),
    path("vortex_app/predict/", views.predict, name="predict")
]