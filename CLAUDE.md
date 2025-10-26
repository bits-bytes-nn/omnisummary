# 🚀 Development Guidelines

## 1. Guiding Philosophy: Simplicity First

All development decisions must adhere to the following principles above all else.

* **Solve the immediate problem with the most straightforward approach.**
* **Build working code first, then optimize and scale only when necessary.**
* **Design general solutions, not temporary workarounds.** Implement solutions that handle common cases robustly, not just the immediate edge case.
* **Avoid premature abstractions and complex design patterns.**
* **Aggressively remove any feature or dependency that does not add clear value.**
* **Prefer boring, proven solutions over new and exciting technologies.**

***

## 2. Code Implementation Rules

### ✅ Minimal, Working Code

* **Implement only essential features.** Write only the code necessary to solve the problem.
* **Minimize comments and docstrings.** Express intent through clear function and variable names. Avoid writing comments and docstrings unless absolutely necessary.
* **Build incrementally and validate frequently.** Develop in small units and test them immediately.
* **Avoid over-engineering.** Always seek the simplest, most optimal solution.

### 🐍 Python Code Standards

#### 1. Type Hints & Naming

* Use descriptive names and follow modern type hinting syntax.

```python
# :white_check_mark: GOOD
def extract_document_metadata(documents: list[dict[str, str]],
                              include_timestamps: bool = True) -> dict[str, str | int]:
    pass

# :x: BAD
from typing import Dict, List
def process_data(*args, **kwargs) -> Dict[str, List[str]]:
    pass
```

#### 2. Function Design

* **Single Responsibility:** Keep functions under 20-30 lines, ensuring each performs a single task.
* **Explicit Parameters:** Use explicit arguments instead of `*args` and `**kwargs`.
* **Immutability:** Favor pure functions that return new objects over mutating existing ones.
* **Data Validation:** Use Pydantic models at module boundaries to validate data.

#### 3. Logging

* Use `%`-formatting for performance.

```python
# :white_check_mark: GOOD
logger.info("Processing document %s with %d pages", document_id, page_count)

# :x: BAD
logger.info(f"Processing document {document_id} with {page_count} pages")
```

***

## 3. Architecture & Tooling

### 🛠️ Required Libraries

* **Data Models:** `Pydantic` (instead of dataclasses)
* **File System:** `pathlib` (instead of os.path)
* **Package Management:** `uv` (instead of pip) - Update `pyproject.toml` when adding or removing dependencies
* **Core Principle:** Choose tools because they solve problems efficiently, not because they are new.

### 🔗 LangChain Integration

* Structure all LLM interactions using **LangChain Expression Language (LCEL)**.
* Store prompts as strings within `.py` files for version control.
* Build modular, reusable chains with LCEL syntax.
* Use explicit Output Parsers when structured responses are needed.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# A minimal chain structure using LCEL
prompt = ChatPromptTemplate.from_template("Analyze this document: {document}")
chain = prompt | llm | StrOutputParser()
```

***

## 4. Reliability & Testing

### 🛡️ Exception Handling

* **Fail-Fast:** Detect and propagate errors as soon as they occur.
* **Custom Exceptions:** Create specific, custom exceptions for different error types.
* **Meaningful Recovery:** Handle only the exceptions you can meaningfully recover from; let others propagate.
* **Graceful Degradation:** Design systems so that a partial failure does not cause a total system outage.

### 🧪 Testing Approach

* **Quality over Quantity:** Focus on the quality of tests, not the quantity. Avoid test proliferation.
* **Test Critical Paths First:** Prioritize testing core business logic and critical paths. Handle edge cases second.
* **Focus on Unit Tests:** Concentrate on unit tests for pure logic with no external dependencies. Use integration tests selectively.
* **Structured Directories:** Organize test code according to the following structure:

```text
tests/
├── unit/           # Core business logic tests
├── integration/    # Tests for AWS and other external services
└── fixtures/       # Test data and mock objects
```

***

## 5. Project Structure

### 📂 Organization Principles

* **Group by Feature:** Organize code by feature domain, not by technical layer (e.g., `controllers`, `services`).
* **Separate Layers:** Clearly separate the presentation, business logic, and data access layers.
* **Dependency Injection:** Use dependency injection to improve testability.
* **Start Simple:** Begin with a simple, flat structure and refactor into a more complex one only as the project demands it.

### ⚡️ Performance Considerations

* **Async Patterns:** Actively use `async` for I/O-bound operations.
* **Optimize Bottlenecks:** Optimize only based on measured bottlenecks, not on assumptions.
* **Design for Horizontal Scaling:** Consider horizontal scaling patterns to improve performance when needed.
* **Event-Driven Architecture:** Evaluate an event-driven architecture to decouple components.
