"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from python.views import *

urlpatterns = [
 
    path('', mainpage, name='mainpage'),
    path('features/',features),
    # add other URL patterns for your project here

    path('admin/', admin.site.urls),
    path('mainpage/',mainpage),
    path('mainpage/features/',features),
#Identification
    path('train/',train),
    path('features/train/',train),
    path('train/identify/',identify_face),
    path('features/train/identify/',identify_face),
    path('mainpage/features/train/',train),
    path('mainpage/features/train/identify/',identify_face),
#record
    path('record/',record),
    path('mainpage/features/record/',record),
    path('features/record/',record),
    
#Rectangle
    path('motiondetection/',motion_detection),
    path('mainpage/features/motiondetection/',motion_detection),
    path('features/motiondetection/',motion_detection),
#In_out
    path('inout/',in_out),
    path('mainpage/features/inout/',in_out),
    path('features/inout/',in_out),
#alarm
    path('alarm/',motion_detector_view),
    path('mainpage/features/alarm/',motion_detector_view),
    path('features/alarm/',motion_detector_view),
#StructuralSimilarity
    path('similarity/',motion_detection_view),
    path('mainpage/features/similarity/',motion_detection_view),
    path('features/similarity/',motion_detection_view),
]

