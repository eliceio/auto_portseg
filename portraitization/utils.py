from .models import RawImage, Portrait


def handle_image_file(image_file):
    raw_image = RawImage.objects.create(image_file=image_file)
    raw_image.save()

    portrait_image = None

    return portrait_image
