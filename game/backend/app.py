from flask import Flask
# from app.routes import api_routes  # Import your API routes
#from config.config import Config  # Import your configuration

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register your API routes
    app.register_blueprint(api_routes)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
