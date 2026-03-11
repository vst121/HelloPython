"""
docker_fastapi_demo.py
Comprehensive demonstration of Dockerizing a Python FastAPI project.

Docker is a platform for developing, shipping, and running applications in containers.
Containers are lightweight and contain everything needed to run the software including
the runtime, libraries, and dependencies.

Why Docker for Python Projects?
- Reproducible environments across different machines
- Easy deployment to production
- Isolation from system dependencies
- Consistent development workflow
- Scalability

Why FastAPI?
- Modern, high-performance Python web framework
- Automatic API documentation (Swagger UI, ReDoc)
- Built-in data validation with Pydantic
- Async/await support out of the box
- Type hints support

Key Topics Covered:
- Understanding Docker images and containers
- Creating a Dockerfile for FastAPI applications
- Best practices for Python Docker images
- Using docker-compose for multi-container applications
- Building, running, and managing containers
- Production deployment with Gunicorn and Uvicorn

This demo includes a FastAPI application that demonstrates:
- Basic FastAPI application structure
- Environment variable configuration
- Pydantic models for data validation
- Automatic API documentation
- Health checks
- Multi-stage builds for production optimization
"""

import subprocess
import os
import sys

# ============================================================================
# SIMPLE FASTAPI APPLICATION (The app we'll Dockerize)
# ============================================================================

# This is the application code that will run inside the Docker container.
# In a real project, this would be in a separate file like main.py

APP_CODE = '''
"""
FastAPI application for Docker demonstration.
FastAPI is a modern, fast (high-performance), web framework for building APIs
with Python 3.7+ based on standard Python type hints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import uvicorn

# Configuration from environment variables
APP_NAME = os.environ.get("APP_NAME", "FastAPIDockerDemo")
APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="A demonstration FastAPI application containerized with Docker",
    debug=DEBUG_MODE
)


# Pydantic models for data validation
class Item(BaseModel):
    """Model for an item in our data store."""
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    price: float = Field(..., gt=0)
    tax: Optional[float] = None


class ItemCreate(BaseModel):
    """Model for creating a new item."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    price: float = Field(..., gt=0)
    tax: Optional[float] = None


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    app: str


class AppInfo(BaseModel):
    """Model for application info response."""
    message: str
    version: str
    debug: bool


# In-memory data store (replace with database in production)
items_db: dict[int, Item] = {}


@app.get("/", response_model=AppInfo)
async def root():
    """Root endpoint returning basic app info."""
    return AppInfo(
        message=f"Welcome to {APP_NAME}!",
        version=APP_VERSION,
        debug=DEBUG_MODE
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration."""
    return HealthResponse(
        status="healthy",
        app=APP_NAME
    )


@app.get("/api/items", response_model=List[Item])
async def get_items(skip: int = 0, limit: int = 100):
    """Get all items with pagination."""
    items = list(items_db.values())
    return items[skip : skip + limit]


@app.get("/api/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    """Get a specific item by ID."""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]


@app.post("/api/items", response_model=Item, status_code=201)
async def create_item(item: ItemCreate):
    """Create a new item."""
    # Generate new ID
    new_id = max(items_db.keys(), default=0) + 1
    
    # Create full item
    new_item = Item(
        id=new_id,
        name=item.name,
        description=item.description,
        price=item.price,
        tax=item.tax
    )
    
    items_db[new_id] = new_item
    return new_item


@app.put("/api/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: ItemCreate):
    """Update an existing item."""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    
    updated_item = Item(
        id=item_id,
        name=item.name,
        description=item.description,
        price=item.price,
        tax=item.tax
    )
    
    items_db[item_id] = updated_item
    return updated_item


@app.delete("/api/items/{item_id}")
async def delete_item(item_id: int):
    """Delete an item."""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    
    del items_db[item_id]
    return {"message": "Item deleted successfully"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=DEBUG_MODE,
        log_level="info"
    )
'''

# ============================================================================
# DOCKERFILE - The recipe for building our Docker image
# ============================================================================

DOCKERFILE = '''# Use official Python runtime as base image
# We use a specific tag for reproducible builds
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE prevents Python from creating .pyc files
# PYTHONUNBUFFERED ensures output is streamed immediately
# UVICORN_WEB_CONCURRENCY optimizes worker count
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    UVICORN_WEB_CONCURRENCY=4

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8000

# Define health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# ============================================================================
# REQUIREMENTS.TXT - Python dependencies
# ============================================================================

REQUIREMENTS_TXT = '''fastapi==0.109.0
uvicorn[standard]==0.27.0
gunicorn==21.2.0
pydantic==2.5.3
pydantic-settings==2.1.0
'''

# ============================================================================
# DOCKER-COMPOSE.YML - For running multi-container applications
# ============================================================================

DOCKER_COMPOSE = '''version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_NAME=FastAPIDocker
      - APP_VERSION=1.0.0
      - DEBUG_MODE=false
    volumes:
      - ./main.py:/app/main.py:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 3s
      retries: 3

  # Example of adding a PostgreSQL database
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=fastapi
      - POSTGRES_PASSWORD=fastapi
      - POSTGRES_DB=fastapi_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Example of adding Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
'''

# ============================================================================
# MULTI-STAGE DOCKERFILE - For production-optimized images
# ============================================================================

DOCKERFILE_MULTISTAGE = '''# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN pip install --no-cache-dir build

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production stage
FROM python:3.11-slim

WORKDIR /app

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy application code
COPY --chown=appuser:appgroup main.py .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run with Gunicorn + Uvicorn workers for production
# Gunicorn handles worker management, Uvicorn runs ASGI workers
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4"]
'''

# ============================================================================
# DOCKER COMMANDS AND USAGE
# ============================================================================

def print_dockerfile_sample():
    """Display the Dockerfile content."""
    print("=" * 70)
    print("DOCKERFILE - Recipe for building the Docker image")
    print("=" * 70)
    print(DOCKERFILE)


def print_requirements_sample():
    """Display the requirements.txt content."""
    print("\n" + "=" * 70)
    print("REQUIREMENTS.TXT - Python dependencies")
    print("=" * 70)
    print(REQUIREMENTS_TXT)


def print_docker_compose_sample():
    """Display the docker-compose.yml content."""
    print("\n" + "=" * 70)
    print("DOCKER-COMPOSE.YML - Multi-container configuration")
    print("=" * 70)
    print(DOCKER_COMPOSE)


def print_multistage_dockerfile():
    """Display the multi-stage Dockerfile."""
    print("\n" + "=" * 70)
    print("MULTI-STAGE DOCKERFILE - Production optimized")
    print("=" * 70)
    print(DOCKERFILE_MULTISTAGE)


def print_docker_commands():
    """Display common Docker commands."""
    print("\n" + "=" * 70)
    print("COMMON DOCKER COMMANDS")
    print("=" * 70)
    commands = """
# Build the Docker image
docker build -t my-fastapi-app .

# Run the container
docker run -p 8000:8000 my-fastapi-app

# Run in detached mode (background)
docker run -d -p 8000:8000 --name my-api my-fastapi-app

# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# Stop a container
docker stop my-api

# Remove a container
docker remove my-api

# View logs
docker logs my-api

# Follow logs in real-time
docker logs -f my-api

# Interactive shell inside container
docker exec -it my-api /bin/bash

# Using docker-compose
docker-compose up -d          # Build and run
docker-compose down          # Stop and remove
docker-compose build         # Rebuild
docker-compose logs -f       # Follow logs

# Access FastAPI docs (automatically generated)
# Once running, visit:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - OpenAPI JSON: http://localhost:8000/openapi.json
"""
    print(commands)


def print_best_practices():
    """Display Docker best practices for Python/FastAPI."""
    print("\n" + "=" * 70)
    print("DOCKER BEST PRACTICES FOR FASTAPI")
    print("=" * 70)
    practices = """
1. USE SPECIFIC PYTHON VERSIONS
   - Use python:3.11-slim instead of python:latest
   - This ensures reproducible builds
   - Consider python:3.11-alpine for even smaller images

2. USE SMALL BASE IMAGES
   - Prefer slim or alpine variants
   - Example: python:3.11-slim instead of python:3.11
   - Alpine images are smaller but may have compatibility issues

3. LAYER CACHING OPTIMIZATION
   - Copy requirements.txt first, then install
   - Copy application code last
   - This caches dependency installation

4. USE NON-ROOT USERS
   - Create and use a non-root user for security
   - Prevents privilege escalation attacks
   - Add USER directive at the end of Dockerfile

5. MINIMIZE LAYERS
   - Combine RUN commands when possible
   - Remove unnecessary files in same layer

6. USE MULTI-STAGE BUILDS
   - Build in one stage, run in another
   - Reduces final image size significantly
   - Excludes build dependencies from production image

7. ADD HEALTH CHECKS
   - FastAPI has /health endpoint built-in
   - Helps orchestration systems monitor your app
   - Used by Kubernetes, Docker Compose

8. USE ENVIRONMENT VARIABLES
   - Configuration without rebuilding image
   - Use pydantic-settings for type-safe config
   - Secrets should use Docker secrets or env files

9. INCLUDE .DOCKERIGNORE
   - Exclude unnecessary files from build context
   - Example: __pycache__, .git, *.pyc, venv/

10. USE GUNICORN WITH UVICORN FOR PRODUCTION
    - Uvicorn's dev server is not for production
    - Gunicorn provides worker processes
    - Use UvicornWorker for ASGI support
"""
    print(practices)


def print_dockerignore_sample():
    """Display .dockerignore content."""
    print("\n" + "=" * 70)
    print(".DOCKERIGNORE - Exclude files from build context")
    print("=" * 70)
    dockerignore = """
# Git
.git
.gitignore

# Python
__pycache__
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.venv

# Testing
.pytest_cache
.coverage
htmlcov/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Documentation
*.md
docs/

# Docker
Dockerfile
docker-compose.yml
.dockerignore

# CI/CD
.github/
.gitlab-ci.yml
"""
    print(dockerignore)


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def demonstrate_dockerization():
    """Main demonstration of Dockerizing a FastAPI project."""
    print("\n" + "=" * 70)
    print("DOCKERIZING A FASTAPI PROJECT - Complete Guide")
    print("=" * 70)
    print("""
This demonstration shows how to containerize a FastAPI application
using Docker. We'll cover the essential files and best practices.

The sample application is a REST API that demonstrates:
- CRUD operations with Pydantic models
- Automatic API documentation
- Health check endpoint
- Environment variable configuration
- Async/await support

FastAPI Advantages:
- Automatic Swagger UI at /docs
- Automatic ReDoc documentation at /redoc
- Built-in data validation with Pydantic
- Native async support for high performance
- OpenAPI schema generation

Let's walk through each component...
""")

    # Show the sample FastAPI application
    print("\n" + "=" * 70)
    print("SAMPLE FASTAPI APPLICATION (main.py)")
    print("=" * 70)
    print(APP_CODE)

    # Show Dockerfile
    print_dockerfile_sample()

    # Show requirements.txt
    print_requirements_sample()

    # Show .dockerignore
    print_dockerignore_sample()

    # Show docker-compose
    print_docker_compose_sample()

    # Show multi-stage Dockerfile
    print_multistage_dockerfile()

    # Show commands
    print_docker_commands()

    # Show best practices
    print_best_practices()

    print("\n" + "=" * 70)
    print("QUICK START - How to Dockerize Your FastAPI Project")
    print("=" * 70)
    print("""
Step 1: Create requirements.txt
    pip freeze > requirements.txt

Step 2: Create your FastAPI application (main.py)
    Write your FastAPI application with Pydantic models

Step 3: Create Dockerfile
    Copy the Dockerfile template from this demo

Step 4: Create .dockerignore
    Exclude unnecessary files from build

Step 5: Build the image
    docker build -t my-fastapi-app .

Step 6: Run the container
    docker run -p 8000:8000 my-fastapi-app

Step 7: Access the API
    - API: http://localhost:8000
    - Swagger Docs: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - Health Check: http://localhost:8000/health
""")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("""
You now have all the information needed to Dockerize a FastAPI project.
The key files needed are:
1. main.py - Your FastAPI application
2. requirements.txt - Python dependencies
3. Dockerfile - Image build instructions
4. .dockerignore - Files to exclude

For production, consider:
- Using docker-compose for multi-container apps (database, cache)
- Multi-stage builds for smaller images
- Non-root users for security
- Health checks for monitoring
- Gunicorn with Uvicorn workers for production
- PostgreSQL or other databases for persistence
- Redis for caching

FastAPI automatically generates interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
""")


if __name__ == "__main__":
    demonstrate_dockerization()
