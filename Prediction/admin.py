from django.contrib import admin
from .models import Google, Twitter, Apple, Microsoft

# Register your models here.
@admin.register(Google)
class GoogleAdmin(admin.ModelAdmin):
    list_per_page = 100


@admin.register(Twitter)
class TwitterAdmin(admin.ModelAdmin):
    list_per_page = 100


@admin.register(Apple)
class AppleAdmin(admin.ModelAdmin):
    list_per_page = 100


@admin.register(Microsoft)
class MicrosoftAdmin(admin.ModelAdmin):
    list_per_page = 100