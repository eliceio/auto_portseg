from django.shortcuts import render
from .forms import UploadImageForm


def portraitization(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return render(request, "pages/portraitization/portrait_result.html",
                          {})
    else:
        form = UploadImageForm()
    return render(request, 'pages/portraitization/upload_image.html', {'form': form})
