from django.db import models

from .utils import PathAndRename

raw_image_upload = PathAndRename("raw_images")
portrait_upload = PathAndRename("portraits")


class TimeStampedModel(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class RawImage(TimeStampedModel):
    image_file = models.ImageField(upload_to=raw_image_upload,
                                   height_field="height_field",
                                   width_field="width_field")
    height_field = models.IntegerField(null=True, default=0)
    width_field = models.IntegerField(null=True, default=0)


class Portrait(TimeStampedModel):
    raw_image = models.ForeignKey(RawImage, on_delete=models.CASCADE)
    image_file = models.ImageField(upload_to=portrait_upload,
                                   height_field="height_field",
                                   width_field="width_field")
    height_field = models.IntegerField(null=True, default=0)
    width_field = models.IntegerField(null=True, default=0)
