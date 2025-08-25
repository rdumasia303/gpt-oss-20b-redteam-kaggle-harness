from flask import Flask
import time


def create_app():
    app = Flask(__name__)

    # Config
    app.config['FINDINGS_DIR'] = 'findings'
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    # Inject current time into templates
    @app.context_processor
    def inject_now():
        return {"now": time.strftime('%Y-%m-%d %H:%M:%S')}

    # Blueprints / routes
    from .views import bp
    app.register_blueprint(bp)

    return app
