"""
WSGI config for auto_portseg project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/howto/deployment/wsgi/
"""

import os
import socket

from django.core.wsgi import get_wsgi_application

ip = socket.gethostbyname(socket.gethostname())

if ip == '172.31.30.149':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'auto_portseg.settings_products')
else:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'auto_portseg.settings')

application = get_wsgi_application()
