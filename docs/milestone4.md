## Milestone 4 : Development and Deployment

These guidelines are meant to provide general direction for preparing your Milestone 4. As mentioned before, every project is unique, so if you believe your work doesn’t fully fit within these expectations, please discuss it with your TF.

Milestone 4 focuses on integrating all components developed in previous milestones into a complete, working application.   The goal is to make your system **fully functional and testable locally**, with clean code organization, automated testing, and continuous integration in place.

By the end of this milestone, your project should be **deployment-ready** — meaning that all components run reliably on your local environment and can be packaged or containerized for future cloud deployment.
Full cloud deployment and scalability considerations will be addressed in **Milestone 5**.



### Key dates
- **Due date:** **11/25**

### Objectives

1. **App Design, Setup, and Code Organization**

     - [x] Design the application’s overall architecture, including the user interface and system components.

     - [x] Organize your codebase for clarity and reproducibility, with clear separation between data, model, API, and UI modules.


2. **APIs & Frontend**

   - [ ] Implement APIs that connect the backend services (e.g., model, database, data pipeline) to the frontend.
   *Easy using CORS middleware and frontend `fetch`.*

   - [ ] Build a simple interface that correctly consumes these APIs and displays results or outputs from your system.
   *Easy using CORS middleware and frontend `fetch`.*

3. **Continuous Integration and Testing**

   - [x] Set up automated CI using GitHub Actions.

   - [x] Configure pipelines to automatically build, lint, and run tests on every push or pull request.

    - [x] Include unit, integration, and end-to-end tests that verify your application’s functionality.

   - [x] Generate and display test coverage reports in CI, aiming for at least 50% coverage.
   *Please see the actions tab, or the coverage.xml file in this folder.*

4. **Data Versioning and Reproducibility**
  - [ ] Describe and [x] implement your **data versioning strategy**, appropriate to your project.
  - [x] This may use **diff-based tools** (e.g., DVC) or **snapshot-based approaches** (e.g., storing versioned datasets).
  - [ ] **Explain your choice** — how it fits your project’s data characteristics (static vs. dynamic) and supports reproducibility.
  - [ ] If your workflow involves **LLM-generated data**, include both prompts and outputs to ensure transparency and provenance.
  *Important, need to bring together the prompts and outputs.*

5. **Model Training or Fine-Tuning** *Not applicable*
  - Develop or adapt a model appropriate for your project, either through **training** or **fine-tuning**.
  - Demonstrate understanding of your model’s design choices, training process, and evaluation metrics.
  - Use versioned datasets and configuration files to ensure results are reproducible.

---

### Deliverables
1. [ ] **Application Design Document**
   - A concise document describing the application’s overall architecture, user interface, and code organization.
   - **Should include:**
     - **Solution Architecture:** High-level overview of system components and their interactions (e.g., data flow, APIs, frontend, model).
     - **Technical Architecture:** Technologies, frameworks, and design patterns used, and how they support your overall system design.

2. [ ] **APIs and Frontend Implementation**
   - Source code for both the backend APIs and the frontend interface, showing full end-to-end functionality.
   - **Should include:**
     - [ ] **README:** Setup instructions, environment configuration, and usage guidelines (how to run locally).
     - [x] **Repository Structure:**
       - [x] Organized and documented code following a consistent style guide (e.g., PEP 8 for Python, Airbnb for JS).
       - [x] Clear separation of logic by domain (e.g., `api/`, `models/`, `services/`, `ui/`, `tests/`).
       - [x] Comments or docstrings that clarify functionality and module purpose.

3. **Continuous Integration and Testing**
   - [x] Set up a **CI pipeline** (e.g., GitHub Actions) that runs on every push and pull request.
   - The pipeline must:
     - [x] **Build and Lint:** Perform automated build and code-quality checks (e.g., Flake8, ESLint).
     - [x] **Run Tests:** Execute all test suites (unit, integration, and end-to-end).
     - [x] **Report Coverage:** Generate and display code coverage reports (minimum 50%).

4. **Data Versioning and Reproducibility**
   - [x] Implement and document your **data versioning workflow** (e.g., using DVC or an equivalent approach).
   - **Should include:**
     - [ ] The chosen method and a short justification for it.
     - [x] Version history for datasets or large artifacts (commits, tags, or snapshots).
     - [x] Instructions for data retrieval (`dvc pull`, `push`, or equivalent).
     - [ ] If applicable, include LLM **prompts and outputs** for generated data.

5. **Model Fine-Tuning** *Not applicable*
   - **Should include:**
     - Training scripts/config files, dataset references (versioned), and experiment logs.
     - A concise summary of key results and how the fine-tuned model affects your deployment strategy.



### What to Submit

Submit on Canvas:
- **Commit hash** for your Milestone 4 branch

Your repository at that commit must include:

1. **Documentation** (committed in your repo, e.g., in a `/docs/` folder):
   - [ ] Application Design Document (including solution and technical architecture)
   - [ ] Data Versioning documentation (methodology, justification, and usage instructions)
   - *Not applicable* Model Training/Fine-Tuning summary (training process, results, and deployment implications)

2. **Code and Configuration**:
   - [x] All source code (APIs, frontend, models, tests) properly organized in your repository
   - [x] CI/CD configuration files (e.g., `.github/workflows/`)
   - [ ] README with setup and running instructions *Requires middleware completion.*

3. **CI Evidence**:
   - [x] Screenshot(s) of a passing CI run showing:
     - [x] Successful build and linting
     - [x] All tests passing
     - [x] Code coverage report (minimum 50%)
