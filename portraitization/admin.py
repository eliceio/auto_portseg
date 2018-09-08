from django.contrib import admin

from .models import RawImage, Portrait


class RawImageAdmin(admin.ModelAdmin):
    model = RawImage

    list_display = ['image_file', ]


class PortraitAdmin(admin.ModelAdmin):
    model = Portrait

    list_display = ['raw_image', 'image_file', ]


admin.site.register(RawImage, RawImageAdmin)
admin.site.register(Portrait, PortraitAdmin)
