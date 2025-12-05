import os
import pulumi
import pulumi_docker_build as docker_build
from pulumi_gcp import artifactregistry
from pulumi import CustomTimeouts
import datetime

# 🔧 Get project info
project = pulumi.Config("gcp").require("project")
location = os.environ["GCP_REGION"]

# 🕒 Timestamp for tagging
timestamp_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
repository_name = "datadetox-repository"
registry_url = f"us-central1-docker.pkg.dev/{project}/{repository_name}"

repository = artifactregistry.Repository.get(
    repository_name,
    f"projects/{project}/locations/{location}/repositories/{repository_name}",
)

# Docker Build + Push -> Backend
image_config = {
    "image_name": "datadetox-backend",
    "context_path": "../../backend",
    "dockerfile": "Dockerfile",
}
backend_image = docker_build.Image(
    f"build-{image_config['image_name']}",
    tags=[
        pulumi.Output.concat(
            registry_url, "/", image_config["image_name"], ":", timestamp_tag
        )
    ],
    context=docker_build.BuildContextArgs(location=image_config["context_path"]),
    dockerfile={
        "location": f"{image_config['context_path']}/{image_config['dockerfile']}"
    },
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
    opts=pulumi.ResourceOptions(
        custom_timeouts=CustomTimeouts(create="30m"),
        retain_on_delete=True,
        depends_on=[repository],
    ),
)
# Export references to stack
pulumi.export("datadetox-backend-ref", backend_image.ref)
pulumi.export("datadetox-backend-tags", backend_image.tags)

# Docker Build + Push -> Frontend
image_config = {
    "image_name": "datadetox-frontend",
    "context_path": "../../frontend",
    "dockerfile": "Dockerfile",
}
frontend_image = docker_build.Image(
    f"build-{image_config['image_name']}",
    tags=[
        pulumi.Output.concat(
            registry_url, "/", image_config["image_name"], ":", timestamp_tag
        )
    ],
    context=docker_build.BuildContextArgs(location=image_config["context_path"]),
    dockerfile={
        "location": f"{image_config['context_path']}/{image_config['dockerfile']}"
    },
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
    opts=pulumi.ResourceOptions(
        custom_timeouts=CustomTimeouts(create="30m"),
        retain_on_delete=True,
        depends_on=[repository],
    ),
)
pulumi.export("datadetox-frontend-ref", frontend_image.ref)
pulumi.export("datadetox-frontend-tags", frontend_image.tags)

# Docker Build + Push -> Model Lineage
image_config = {
    "image_name": "datadetox-model-lineage",
    "context_path": "../../model-lineage",
    "dockerfile": "Dockerfile",
}
model_lineage_image = docker_build.Image(
    f"build-{image_config['image_name']}",
    tags=[
        pulumi.Output.concat(
            registry_url, "/", image_config["image_name"], ":", timestamp_tag
        )
    ],
    context=docker_build.BuildContextArgs(location=image_config["context_path"]),
    dockerfile={
        "location": f"{image_config['context_path']}/{image_config['dockerfile']}"
    },
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
    opts=pulumi.ResourceOptions(
        custom_timeouts=CustomTimeouts(create="30m"),
        retain_on_delete=True,
        depends_on=[repository],
    ),
)
# Export references to stack
pulumi.export("datadetox-model-lineage-ref", model_lineage_image.ref)
pulumi.export("datadetox-model-lineage-tags", model_lineage_image.tags)

# Pull and push Neo4j to Artifact Registry
image_config = {
    "image_name": "datadetox-neo4j-mirror",
    "context_path": "../../neo4j",
    "dockerfile": "Dockerfile",
}
neo4j_image = docker_build.Image(
    f"build-{image_config['image_name']}",
    tags=[
        pulumi.Output.concat(
            registry_url, "/", image_config["image_name"], ":", timestamp_tag
        )
    ],
    context=docker_build.BuildContextArgs(location=image_config["context_path"]),
    dockerfile={
        "location": f"{image_config['context_path']}/{image_config['dockerfile']}"
    },
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
    opts=pulumi.ResourceOptions(
        custom_timeouts=CustomTimeouts(create="30m"),
        retain_on_delete=True,
        depends_on=[repository],
    ),
)

pulumi.export("datadetox-neo4j-mirror-ref", neo4j_image.ref)
pulumi.export("datadetox-neo4j-mirror-tags", neo4j_image.tags)
