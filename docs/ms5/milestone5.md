## Objectives
1. Production Deployment & Infrastructure Automation
- [X] Deploy your full application to a cloud Kubernetes cluster (GCP or AWS).
- [X] Demonstrate reliability and basic scalability (e.g., scaling pods up/down under load).
- [X] Automate infrastructure provisioning and deployment using Pulumi.
- [N/A] Integrate ML workflow elements (data preprocessing, training, evaluation, retraining triggers) into your deployed system.

2. CI/CD for Production
- [ ] Extend your GitHub Actions pipelines to support deployment.
- [ ] Include unit tests and integration tests.
- [ ] Achieve minimum 60% test coverage, and clearly document what remains untested.
- [ ] Ensure merges to main trigger an automated build-and-deploy pipeline.

3. Public Communication and Presentation Materials
- [ ] Prepare a polished 6-minute video describing and demonstrating your project.
- [ ] Write a 600–800 word Medium blog post explaining your project to a general audience.
- [ ] Create engaging booth materials for the showcase (e.g., diagrams, QR codes).

4. Showcase Preparation
- [ ] Ensure your application is publicly accessible, stable, and easy to demo.
- [ ] Prepare to explain your architecture, decisions, and business value to visitors.

## Deliverables
### 1. Technical Implementation
- [X] Kubernetes Deployment:
    - [X] Deploy the application to a Kubernetes cluster.
    - [X] Demonstrate basic scaling behavior by varying the load and showing how the cluster responds (e.g., scaling replicas/pods).
- [X] Pulumi Infrastructure Code:
    - [X] Use Pulumi to automate the provisioning and deployment of your infrastructure (e.g., Kubernetes cluster, networking, storage, configurations, etc.) and application.
- [ ] CI/CD Pipeline Implementation (GitHub Actions):
    - [ ] Set up a CI/CD pipeline using GitHub Actions. The pipeline should:
        - [X] Have a unit test suite for each service/container.
        - [ ] Run integration tests on the code base .
        - [ ] Deploy updates to the Kubernetes cluster upon merging changes into the main branch.
        - [ ] Achieve at least 60% line coverage. Document which functions and modules are not covered by tests.
- [X] Machine Learning Workflow:
    - Demonstrate a production-ready ML workflow, including:
        - Data preprocessing, model training, and evaluation steps integrated into the pipeline.
        - Automated retraining and deployment triggered by new data or updates to the codebase.
        - Validation checks to ensure only models meeting performance thresholds are deployed.

### 2. Documentation
- [ ] GitHub Repository:
    - [ ] Include a well-structured and modular codebase.
    - [ ] Provide a comprehensive README file with the following sections:
        - [ ] Prerequisites and setup instructions.
        - [ ] Deployment instructions.
        - [ ] Usage details and examples.
        - [ ] Known issues and limitations.
    - [ ] Submit the `main` branch for this deliverable.

### 3. Presentation Materials
- [ ] Video Presentation:
    - [ ] Record a 6-minute video covering:
        - Problem statement and the proposed solution.
        - Technical architecture and key components.
        - Live demo of the application in action.
        - Challenges faced and solutions implemented.
    - [ ] Submit the video in MP4 format or YouTube link with a minimum resolution of 720p.
- [ ] Blog Post:
    - Write a 600–800 word Medium blog post summarizing your project for a general audience. The post should highlight the problem, solution, technical approach, and impact.
    - Include visuals or diagrams where appropriate.
- [ ] Self and Peer Review Forms:
    - Complete self-assessment and peer evaluation forms to provide feedback on team contributions.

### 4. Showcase (Dec 10th)
- Event Format:
    - Each team will have 45 minutes during the live showcase to present their project.
    - Participants will visit your booth to interact with your application and learn about your implementation.
    - Monitors will be provided to most teams. Additional equipment or materials must be arranged by the team.
- App Requirements:
    - The app must be fully functional and hosted on GCP or AWS, accessible via a public URL.
    - Include a QR code linking to your application so visitors can easily access and explore it.
    - Be prepared to explain your problem, solution, technical implementation, and business value to participants.
- Best of Show Award:
    - A committee will evaluate all projects during the showcase to select the Best of Show. Evaluation criteria include:
    - Innovation and impact.
    - Technical complexity and robustness.
    - Clarity of presentation and engagement with participants.
