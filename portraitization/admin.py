from django.contrib import admin

from .models import RawImage, CropImage, Portrait


class RawImageAdmin(admin.ModelAdmin):
    model = RawImage

    list_display = ['image_file', ]


class CropImageAdmin(admin.ModelAdmin):
    model = CropImage

    list_display = ['raw_image', 'image_file', ]


class PortraitAdmin(admin.ModelAdmin):
    model = Portrait

    list_display = ['raw_image', 'crop_image', 'image_file', ]


admin.site.register(RawImage, RawImageAdmin)
admin.site.register(CropImage, CropImageAdmin)
admin.site.register(Portrait, PortraitAdmin)
