from flask import Flask
from config import Config
from apps.iris_dashboard import generate_dashboard

import dash
import dash_core_components as dcc
import dash_html_components as html

server = Flask(__name__)
server.config.from_object(Config)

app = generate_dashboard(server)

from apps import routes
