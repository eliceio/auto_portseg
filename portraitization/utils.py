import datetime
import os

from django.utils.deconstruct import deconstructible


@deconstructible
class PathAndRename(object):

    def __init__(self, sub_path):
        self.path = sub_path

    def __call__(self, instance, filename):
        now = datetime.datetime.now()

        filepath = self.path + now.strftime("/%Y/%m/%d/%H/%M")

        name = filename.split('.')[0]
        ext = filename.split('.')[-1]
        filename = '{}.{}'.format(str(now.second) + "_" + name, ext)

        return os.path.join(filepath, filename)
