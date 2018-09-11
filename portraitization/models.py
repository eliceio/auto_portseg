import datetime
import os

from django.db import models
from django.utils.deconstruct import deconstructible


@deconstructible
class PathAndRename(object):

    def __init__(self, sub_path):
        self.path = sub_path

    def __call__(self, instance, filename):
        now = datetime.datetime.now()
        name = now.strftime("%Y%m%d%H%M%S_") + filename

        return os.path.join(self.path, name)


raw_image_upload = PathAndRename("raw_images")
# crop_image_upload = PathAndRename("crop_images")
# portrait_upload = PathAndRename("portraits")


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


class CropImage(TimeStampedModel):
    raw_image = models.ForeignKey(RawImage, on_delete=models.CASCADE)
    image_file = models.ImageField(upload_to='not_used',
                                   height_field="height_field",
                                   width_field="width_field")
    height_field = models.IntegerField(null=True, default=0)
    width_field = models.IntegerField(null=True, default=0)


class Portrait(TimeStampedModel):
    raw_image = models.ForeignKey(RawImage, on_delete=models.CASCADE)
    image_file = models.ImageField(upload_to='not_used',
                                   height_field="height_field",
                                   width_field="width_field")
    height_field = models.IntegerField(null=True, default=0)
    width_field = models.IntegerField(null=True, default=0)
