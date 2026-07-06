"""Starlette middlewares layered on top of the FastAPI app.

Each middleware here guards a specific policy (global daily cap, kill
switch) so ``server.py`` stays a wiring file rather than a policy file.
"""
