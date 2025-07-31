from os import cpu_count

timeout = 500 
workers = 2 * cpu_count() + 1
name = "web"
bind = "0.0.0.0:5000"  #localhost:port
worker_class = "gthread"  #uvicorn workers for asgi, but flask is wsgi
accesslog="gunicorn_access.log"
wsgi_app = "web:app"
capture_output = False  #captures prints in errorlog
loglevel = "info"  #captures prints in errorlog