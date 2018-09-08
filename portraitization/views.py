from django.shortcuts import render

from .forms import UploadImageForm
from .utils import handle_image_file


def portraitization(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            portrait_image = handle_image_file(image_file=request.FILES['image_file'])
            return render(request, "pages/portraitization/portrait_result.html",
                          {'portrait_image': portrait_image})
    else:
        form = UploadImageForm()
    return render(request, 'pages/portraitization/upload_image.html', {'form': form})
