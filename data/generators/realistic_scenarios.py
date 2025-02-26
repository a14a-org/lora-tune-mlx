"""Generator for realistic coding scenarios."""

import random
from typing import Dict, List, Any, Optional

from data.utils.openai_variations import generate_message_variations, DEFAULT_MODEL

class RealisticScenarioGenerator:
    """Generates realistic coding scenarios with natural language variations."""
    
    def __init__(self, use_openai_variations: bool = True, model: str = DEFAULT_MODEL):
        """Initialize the generator.
        
        Args:
            use_openai_variations: Whether to use OpenAI API for generating message variations
            model: OpenAI model to use for variations (defaults to cost-effective model)
        """
        self.use_openai_variations = use_openai_variations
        self.model = model
        self.scenario_types = [
            self._generate_feature_implementation_scenario,
            self._generate_code_cleanup_scenario,
            self._generate_performance_optimization_scenario,
            self._generate_config_management_scenario,
            self._generate_error_handling_scenario
        ]
    
    def generate(self) -> Dict[str, Any]:
        """Generate a realistic coding scenario."""
        scenario_func = random.choice(self.scenario_types)
        scenario = scenario_func()
        
        if self.use_openai_variations:
            # Generate variations for user messages
            messages = scenario["messages"]
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    variations = generate_message_variations(
                        msg["content"],
                        num_variations=1,  # Get one variation to replace the original
                        style="technical",  # Use technical style for coding scenarios
                        model=self.model  # Use the specified model (defaults to cost-effective one)
                    )
                    if variations and variations[0] != msg["content"]:
                        messages[i]["content"] = variations[0]
        
        return scenario

    def _generate_feature_implementation_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about implementing a new feature with proper research."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "We need to add rate limiting to our API endpoints. Can you implement that?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help implement rate limiting. First, let me search the codebase to understand our current API setup.",
                    "function_call": {
                        "name": "codebase_search",
                        "arguments": {
                            "query": "API middleware setup route configuration",
                            "explanation": "Understanding current API setup to determine best place for rate limiting"
                        }
                    }
                }
                # Additional messages would follow showing implementation steps
            ]
        }

    def _generate_code_cleanup_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about cleaning up and refactoring code."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "We have a lot of duplicate code in our utility functions. Can you help refactor it?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help clean up the duplicate code. Let's first find all utility functions in the codebase.",
                    "function_call": {
                        "name": "grep_search",
                        "arguments": {
                            "query": "def.*util",
                            "explanation": "Searching for utility function definitions to identify duplication"
                        }
                    }
                }
                # Additional messages would follow showing refactoring steps
            ]
        }

    def _generate_performance_optimization_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about optimizing application performance."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "Our database queries are taking too long. Can you help optimize them?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help optimize the database queries. Let's first locate all database query code.",
                    "function_call": {
                        "name": "codebase_search",
                        "arguments": {
                            "query": "database query SELECT JOIN WHERE",
                            "explanation": "Finding database queries to analyze for optimization"
                        }
                    }
                }
                # Additional messages would follow showing optimization steps
            ]
        }

    def _generate_config_management_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about improving configuration management."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "We need to improve our configuration management to handle different environments better."
                },
                {
                    "role": "assistant",
                    "content": "I'll help set up a robust configuration system. First, let me check current config handling.",
                    "function_call": {
                        "name": "codebase_search",
                        "arguments": {
                            "query": "config settings environment variables",
                            "explanation": "Looking for existing configuration management code"
                        }
                    }
                }
                # Additional messages would follow showing implementation steps
            ]
        }

    def _generate_error_handling_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about improving error handling."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "Our error handling is inconsistent. Can you help standardize it?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help standardize error handling. First, let me search for current error handling patterns.",
                    "function_call": {
                        "name": "grep_search",
                        "arguments": {
                            "query": "raise|except|try|error|Exception",
                            "explanation": "Finding current error handling patterns"
                        }
                    }
                }
                # Additional messages would follow showing implementation steps
            ]
        }

    def _generate_logging_setup_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about setting up logging in an application."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "We need to add error logging to our application. Can you help set that up?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help you set up error logging. First, let me check if there's any existing logging configuration in the codebase.",
                    "function_call": {
                        "name": "codebase_search",
                        "arguments": {
                            "query": "logging configuration setup",
                            "explanation": "Searching for any existing logging setup in the codebase"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "No relevant results found."
                },
                {
                    "role": "assistant",
                    "content": "I don't see any existing logging configuration. Let me check the project structure to determine the best place to add logging.",
                    "function_call": {
                        "name": "list_dir",
                        "arguments": {
                            "relative_workspace_path": ".",
                            "explanation": "Checking project structure to determine best location for logging setup"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Contents of directory: ['src', 'tests', 'utils']"
                },
                {
                    "role": "assistant",
                    "content": "I'll create a comprehensive logging configuration in the utils directory.",
                    "function_call": {
                        "name": "edit_file",
                        "arguments": {
                            "target_file": "utils/logging_config.py",
                            "instructions": "Create logging configuration file with both file and console handlers",
                            "code_edit": """import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict
import json
from logging.handlers import RotatingFileHandler

class JsonFormatter(logging.Formatter):
    \"\"\"Custom formatter that outputs logs in JSON format.\"\"\"
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Union[str, int]] = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    json_format: bool = False
) -> None:
    \"\"\"
    Set up logging configuration with both console and file handlers.
    
    Args:
        log_file: Optional path to log file. If None, only console logging is enabled.
        level: Logging level to use. Defaults to INFO.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of backup files to keep.
        json_format: Whether to output logs in JSON format.
    \"\"\"
    # Create formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    \"\"\"
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger, typically __name__
        
    Returns:
        Configured logger instance
    \"\"\"
    return logging.getLogger(name)"""
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "{\"success\": true, \"changes\": [{\"type\": \"create\", \"path\": \"utils/logging_config.py\"}]}"
                },
                {
                    "role": "assistant",
                    "content": "Now let's create an example of how to use this logging configuration. I'll check for the main application file.",
                    "function_call": {
                        "name": "file_search",
                        "arguments": {
                            "query": "main.py app.py",
                            "explanation": "Looking for the main application file to add logging example"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Found: src/main.py"
                },
                {
                    "role": "assistant",
                    "content": "I'll add logging to the main application file.",
                    "function_call": {
                        "name": "edit_file",
                        "arguments": {
                            "target_file": "src/main.py",
                            "instructions": "Add logging configuration and example usage",
                            "code_edit": """from utils.logging_config import setup_logging, get_logger
import os
from pathlib import Path

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
setup_logging(
    log_file=str(log_dir / "app.log"),
    level=logging.INFO,
    json_format=True
)

logger = get_logger(__name__)

def main():
    logger.info("Application starting")
    try:
        # Your main application code here
        logger.debug("Initializing components")
        
        # Example of different log levels
        logger.info("Application initialized successfully")
        logger.warning("Resource usage above 80%")
        
    except Exception as e:
        logger.error("Fatal error in main application", exc_info=True)
        raise

if __name__ == "__main__":
    main()"""
                        }
                    }
                }
            ]
        }

    def _generate_debugging_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about debugging a performance issue."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "Our API endpoints are responding slowly. Can you help investigate?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help investigate the performance issue. First, let's look for API-related code.",
                    "function_call": {
                        "name": "codebase_search",
                        "arguments": {
                            "query": "API endpoint route handler definition",
                            "explanation": "Looking for API endpoint implementations to investigate performance"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Found matches in: src/api/routes.py"
                },
                {
                    "role": "assistant",
                    "content": "Let me examine the API routes implementation.",
                    "function_call": {
                        "name": "read_file",
                        "arguments": {
                            "relative_workspace_path": "src/api/routes.py",
                            "start_line_one_indexed": 1,
                            "end_line_one_indexed_inclusive": 100,
                            "should_read_entire_file": False,
                            "explanation": "Reading API route implementations to identify performance bottlenecks"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Content of src/api/routes.py:\n@app.route('/data')\ndef get_data():\n    results = db.session.query(Data).all()\n    return jsonify([r.to_dict() for r in results])"
                },
                {
                    "role": "assistant",
                    "content": "I see a potential performance issue. The endpoint is fetching all records without pagination. Let me add pagination and query optimization.",
                    "function_call": {
                        "name": "edit_file",
                        "arguments": {
                            "target_file": "src/api/routes.py",
                            "instructions": "Add pagination and query optimization to the data endpoint",
                            "code_edit": """from flask import request, jsonify
from sqlalchemy.orm import joinedload

@app.route('/data')
def get_data():
    \"\"\"
    Get paginated data with optimized query.
    
    Query parameters:
        page: Page number (default: 1)
        per_page: Items per page (default: 20)
    \"\"\"
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Add index hint and eager loading if needed
    query = db.session.query(Data)\\
        .options(joinedload(Data.related_data))\\
        .order_by(Data.id)
    
    # Add pagination
    paginated = query.paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    
    return jsonify({
        'data': [item.to_dict() for item in paginated.items],
        'total': paginated.total,
        'pages': paginated.pages,
        'current_page': page
    })"""
                        }
                    }
                }
            ]
        }

    def _generate_test_coverage_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about improving test coverage."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "Our test coverage is low for the authentication module. Can you help add more tests?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help improve test coverage. First, let me examine the authentication module.",
                    "function_call": {
                        "name": "read_file",
                        "arguments": {
                            "relative_workspace_path": "src/auth/authentication.py",
                            "start_line_one_indexed": 1,
                            "end_line_one_indexed_inclusive": 100,
                            "should_read_entire_file": False,
                            "explanation": "Reading authentication module to understand what needs testing"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Content shows authentication functions: verify_password(), create_token(), validate_token()"
                },
                {
                    "role": "assistant",
                    "content": "Now let me check existing tests to avoid duplication.",
                    "function_call": {
                        "name": "read_file",
                        "arguments": {
                            "relative_workspace_path": "tests/test_authentication.py",
                            "start_line_one_indexed": 1,
                            "end_line_one_indexed_inclusive": 100,
                            "should_read_entire_file": False,
                            "explanation": "Checking existing tests to identify coverage gaps"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Only basic happy path tests found"
                },
                {
                    "role": "assistant",
                    "content": "I'll add comprehensive tests including edge cases and error conditions.",
                    "function_call": {
                        "name": "edit_file",
                        "arguments": {
                            "target_file": "tests/test_authentication.py",
                            "instructions": "Add comprehensive test cases for authentication module",
                            "code_edit": """import pytest
from datetime import datetime, timedelta
from src.auth.authentication import verify_password, create_token, validate_token
from src.auth.exceptions import AuthenticationError, TokenError

@pytest.fixture
def valid_password():
    return "SecurePass123!"

@pytest.fixture
def valid_token():
    return create_token({"user_id": 123, "role": "user"})

class TestPasswordVerification:
    def test_valid_password(self, valid_password):
        \"\"\"Test successful password verification\"\"\"
        hashed = hash_password(valid_password)
        assert verify_password(valid_password, hashed) is True
    
    def test_invalid_password(self, valid_password):
        \"\"\"Test failed password verification\"\"\"
        hashed = hash_password(valid_password)
        assert verify_password("WrongPass123!", hashed) is False
    
    def test_empty_password(self):
        \"\"\"Test empty password handling\"\"\"
        with pytest.raises(ValueError):
            verify_password("", "some_hash")
    
    def test_none_password(self):
        \"\"\"Test None password handling\"\"\"
        with pytest.raises(ValueError):
            verify_password(None, "some_hash")

class TestTokenCreation:
    def test_valid_token_creation(self):
        \"\"\"Test successful token creation\"\"\"
        payload = {"user_id": 123, "role": "user"}
        token = create_token(payload)
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_token_with_custom_expiry(self):
        \"\"\"Test token creation with custom expiration\"\"\"
        payload = {"user_id": 123}
        expiry = timedelta(hours=2)
        token = create_token(payload, expiry)
        decoded = decode_token(token)
        assert decoded["exp"] - decoded["iat"] == int(expiry.total_seconds())
    
    def test_invalid_payload(self):
        \"\"\"Test token creation with invalid payload\"\"\"
        with pytest.raises(ValueError):
            create_token(None)
        
        with pytest.raises(ValueError):
            create_token({"user_id": object()})  # Non-serializable type

class TestTokenValidation:
    def test_valid_token(self, valid_token):
        \"\"\"Test successful token validation\"\"\"
        payload = validate_token(valid_token)
        assert isinstance(payload, dict)
        assert "user_id" in payload
        assert "role" in payload
    
    def test_expired_token(self):
        \"\"\"Test expired token handling\"\"\"
        expired_token = create_token(
            {"user_id": 123},
            expiry=timedelta(seconds=-1)
        )
        with pytest.raises(TokenError) as exc_info:
            validate_token(expired_token)
        assert "expired" in str(exc_info.value).lower()
    
    def test_invalid_token_format(self):
        \"\"\"Test malformed token handling\"\"\"
        invalid_tokens = [
            "not.a.token",
            "invalid",
            "",
            None
        ]
        for token in invalid_tokens:
            with pytest.raises(TokenError):
                validate_token(token)
    
    def test_tampered_token(self, valid_token):
        \"\"\"Test tampered token handling\"\"\"
        tampered = valid_token[:-1] + ("1" if valid_token[-1] == "0" else "0")
        with pytest.raises(TokenError) as exc_info:
            validate_token(tampered)
        assert "invalid" in str(exc_info.value).lower()"""
                        }
                    }
                }
            ]
        }

    def _generate_api_documentation_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about improving API documentation."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "We need to improve our API documentation. Can you help with that?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help improve the API documentation. First, let me check the current documentation.",
                    "function_call": {
                        "name": "list_dir",
                        "arguments": {
                            "relative_workspace_path": ".",
                            "explanation": "Checking project structure to identify API documentation"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Contents of directory: ['docs']"
                },
                {
                    "role": "assistant",
                    "content": "I'll improve the API documentation. Let's first update the existing documentation.",
                    "function_call": {
                        "name": "edit_file",
                        "arguments": {
                            "target_file": "docs/api_documentation.md",
                            "instructions": "Update API documentation",
                            "code_edit": """# API Documentation

## Endpoints

### GET /data

Returns paginated data with optimized query.

#### Query Parameters
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)

#### Example
```
curl -X GET "http://localhost:5000/data?page=1&per_page=20"
```

### POST /data

Creates a new data entry.

#### Request Body
```
{
  "title": "New Data Entry",
  "content": "This is the content of the new data entry."
}
```

### PUT /data/{id}

Updates an existing data entry.

#### Path Parameters
- `id`: ID of the data entry to update

#### Request Body
```
{
  "title": "Updated Data Entry",
  "content": "This is the updated content of the data entry."
}
```

### DELETE /data/{id}

Deletes an existing data entry.

#### Path Parameters
- `id`: ID of the data entry to delete

#### Example
```
curl -X DELETE "http://localhost:5000/data/1"
```"""
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Documentation updated successfully"
                }
            ]
        }

    def _generate_code_migration_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about migrating code to a new framework/version."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "We need to migrate our code from Flask to FastAPI. Can you help with that?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help migrate the code. First, let me analyze the current Flask routes.",
                    "function_call": {
                        "name": "grep_search",
                        "arguments": {
                            "query": "@app.route",
                            "include_pattern": "*.py",
                            "explanation": "Finding all Flask route definitions"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Found routes in src/api/routes.py and src/api/auth.py"
                },
                {
                    "role": "assistant",
                    "content": "Let me examine one of the route files to understand the current implementation.",
                    "function_call": {
                        "name": "read_file",
                        "arguments": {
                            "relative_workspace_path": "src/api/routes.py",
                            "start_line_one_indexed": 1,
                            "end_line_one_indexed_inclusive": 100,
                            "should_read_entire_file": False,
                            "explanation": "Reading Flask routes to plan migration to FastAPI"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Content shows Flask routes with request parsing and response formatting"
                },
                {
                    "role": "assistant",
                    "content": "I'll create a new FastAPI version of this route file with proper type hints and validation.",
                    "function_call": {
                        "name": "edit_file",
                        "arguments": {
                            "target_file": "src/api/routes_fastapi.py",
                            "instructions": "Create FastAPI version of the routes with proper typing",
                            "code_edit": """from fastapi import FastAPI, HTTPException, Query, Depends
from typing import List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .database import get_db
from .models import Data
from .auth import get_current_user

app = FastAPI(title="API Service", version="1.0.0")

class DataResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class DataCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1)

@app.get("/data", response_model=List[DataResponse])
async def get_data(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> List[DataResponse]:
    \"\"\"
    Get paginated data entries.
    
    Args:
        page: Page number (min: 1)
        per_page: Items per page (min: 1, max: 100)
        db: Database session
        current_user: Authenticated user
        
    Returns:
        List of data entries
        
    Raises:
        HTTPException: If page is invalid or no data found
    \"\"\"
    skip = (page - 1) * per_page
    
    query = db.query(Data)\\
        .options(joinedload(Data.related_data))\\
        .order_by(Data.created_at.desc())
    
    total = query.count()
    if skip >= total:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page} does not exist"
        )
    
    items = query.offset(skip).limit(per_page).all()
    return items

@app.post("/data", response_model=DataResponse)
async def create_data(
    data: DataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> DataResponse:
    \"\"\"
    Create a new data entry.
    
    Args:
        data: Data to create
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Created data entry
        
    Raises:
        HTTPException: If data creation fails
    \"\"\"
    db_item = Data(
        title=data.title,
        content=data.content,
        user_id=current_user.id
    )
    
    try:
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return db_item
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to create data entry"
        ) from e"""
                        }
                    }
                }
            ]
        }

    def _generate_monitoring_setup_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about setting up application monitoring."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "We need to add proper monitoring to our application. Can you help set that up?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help set up comprehensive monitoring. First, let me check the current application structure.",
                    "function_call": {
                        "name": "list_dir",
                        "arguments": {
                            "relative_workspace_path": "src",
                            "explanation": "Understanding application structure to plan monitoring setup"
                        }
                    }
                },
                {
                    "role": "system",
                    "content": "Found: api/, core/, utils/"
                },
                {
                    "role": "assistant",
                    "content": "I'll create a monitoring module with Prometheus metrics and structured logging.",
                    "function_call": {
                        "name": "edit_file",
                        "arguments": {
                            "target_file": "src/monitoring/metrics.py",
                            "instructions": "Create monitoring module with Prometheus metrics",
                            "code_edit": """from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Optional, Callable
from functools import wraps
import time
import logging
from contextvars import ContextVar

# Initialize logger
logger = logging.getLogger(__name__)

# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

db_pool_size = Gauge(
    'db_pool_size',
    'Database connection pool size'
)

app_info = Info('app_info', 'Application information')

# Request tracking
request_id_var: ContextVar[str] = ContextVar('request_id')

def track_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float
) -> None:
    \"\"\"
    Track HTTP request metrics.
    
    Args:
        method: HTTP method
        endpoint: Request endpoint
        status_code: Response status code
        duration: Request duration in seconds
    \"\"\"
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status=status_code
    ).inc()
    
    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)

def monitor_endpoint(func: Callable) -> Callable:
    \"\"\"
    Decorator to monitor endpoint performance and errors.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with monitoring
    \"\"\"
    @wraps(func)
    async def wrapper(*args, **kwargs):
        method = kwargs.get('method', 'UNKNOWN')
        endpoint = func.__name__
        
        start_time = time.time()
        active_requests.inc()
        
        try:
            result = await func(*args, **kwargs)
            status_code = getattr(result, 'status_code', 200)
            return result
        except Exception as e:
            status_code = getattr(e, 'status_code', 500)
            logger.exception(
                "Request failed",
                extra={
                    'method': method,
                    'endpoint': endpoint,
                    'error': str(e),
                    'request_id': request_id_var.get(None)
                }
            )
            raise
        finally:
            duration = time.time() - start_time
            track_request(method, endpoint, status_code, duration)
            active_requests.dec()
    
    return wrapper

def update_db_metrics(pool_size: int) -> None:
    \"\"\"Update database connection pool metrics.\"\"\"
    db_pool_size.set(pool_size)

def set_app_info(
    version: str,
    environment: str,
    git_commit: Optional[str] = None
) -> None:
    \"\"\"
    Set application information metrics.
    
    Args:
        version: Application version
        environment: Deployment environment
        git_commit: Optional git commit hash
    \"\"\"
    info = {
        'version': version,
        'environment': environment
    }
    if git_commit:
        info['git_commit'] = git_commit
    
    app_info.info(info)"""
                        }
                    }
                },
                {
                    "role": "assistant",
                    "content": "Now let's add the monitoring middleware to our FastAPI application.",
                    "function_call": {
                        "name": "edit_file",
                        "arguments": {
                            "target_file": "src/api/middleware.py",
                            "instructions": "Add monitoring middleware",
                            "code_edit": """import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ..monitoring.metrics import (
    track_request,
    request_id_var,
    active_requests
)

class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        
        # Add request ID to headers
        request.state.request_id = request_id
        
        # Track request
        method = request.method
        endpoint = request.url.path
        start_time = time.time()
        active_requests.inc()
        
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Re-raise exception after tracking
            raise e
        finally:
            duration = time.time() - start_time
            status_code = getattr(response, 'status_code', 500)
            
            track_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )
            active_requests.dec()

async def metrics_endpoint() -> Response:
    \"\"\"Endpoint to expose Prometheus metrics.\"\"\"
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )"""
                        }
                    }
                }
            ]
        }

    def _generate_error_handling_scenario(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario about improving error handling."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "Our error handling is inconsistent. Can you help standardize it?"
                },
                {
                    "role": "assistant",
                    "content": "I'll help standardize error handling. First, let me search for current error handling patterns.",
                    "function_call": {
                        "name": "grep_search",
                        "arguments": {
                            "query": "raise|except|try|error|Exception",
                            "explanation": "Finding current error handling patterns"
                        }
                    }
                }
                # Additional messages would follow showing implementation steps
            ]
        }

    # Additional scenario methods would follow... 

    # Additional scenario methods would follow... 