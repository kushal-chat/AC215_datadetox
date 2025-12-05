import os
import pulumi
import pulumi_kubernetes as k8s
from dotenv import load_dotenv

load_dotenv("/app/.env")


def setup_containers(project, namespace, k8s_provider, ksa_name):
    # Get image references from deploy_images stack in GCS bucket
    images_stack = pulumi.StackReference("organization/datadetox-deployment/dev")
    # Get the image tags (these are arrays, so we take the first element)
    backend_tag = images_stack.get_output("datadetox-backend-tags")
    frontend_tag = images_stack.get_output("datadetox-frontend-tags")
    model_lineage_tag = images_stack.get_output("datadetox-model-lineage-tags")
    neo4j_mirror_tag = images_stack.get_output("datadetox-neo4j-mirror-tags")

    # Get environment variables from .env file and set as secrets
    openai_api_key = pulumi.Output.secret(os.getenv("OPENAI_API_KEY", ""))
    hf_token = pulumi.Output.secret(os.getenv("HF_TOKEN", ""))
    neo4j_user = os.getenv(
        "NEO4J_USER", "neo4j"
    )  # Username can be plain (not sensitive)
    neo4j_password = pulumi.Output.secret(os.getenv("NEO4J_PASSWORD", "password"))

    # Combine neo4j_user and password securely
    neo4j_auth = pulumi.Output.all(
        pulumi.Output.from_input(neo4j_user), neo4j_password
    ).apply(lambda args: f"{args[0]}/{args[1]}")

    # General persistent storage for application data (5Gi)
    persistent_pvc = k8s.core.v1.PersistentVolumeClaim(
        "persistent-pvc",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="persistent-pvc",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(
            access_modes=["ReadWriteOnce"],  # Single pod read/write access
            resources=k8s.core.v1.VolumeResourceRequirementsArgs(
                requests={"storage": "5Gi"},  # Request 5GB of persistent storage
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    # Dedicated storage for Neo4j database (20Gi)
    neo4j_pvc = k8s.core.v1.PersistentVolumeClaim(
        "neo4j-pvc",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="neo4j-pvc",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(
            access_modes=["ReadWriteOnce"],  # Single pod read/write access
            resources=k8s.core.v1.VolumeResourceRequirementsArgs(
                requests={"storage": "20Gi"},  # Request 20GB for Neo4j database
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    # --- Frontend Deployment ---
    # Creates pods running the frontend container on port 3000
    # ram 1.7 gb
    frontend_deployment = k8s.apps.v1.Deployment(
        "frontend",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="frontend",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "frontend"},  # Select pods with this label
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "frontend"},  # Label assigned to pods
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="frontend",
                            image=frontend_tag.apply(
                                lambda tags: tags[0]
                            ),  # Container image for frontend
                            image_pull_policy="IfNotPresent",  # Use cached image if available
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=3000,  # Frontend app listens on port 3000
                                    protocol="TCP",
                                )
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "250m", "memory": "2Gi"},
                                limits={"cpu": "500m", "memory": "3Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    # --- Frontend Service ---
    frontend_service = k8s.core.v1.Service(
        "frontend-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="frontend",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",  # Internal only - not exposed outside cluster
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=3000,  # Service port
                    target_port=3000,  # Container port to forward to
                    protocol="TCP",
                )
            ],
            selector={"run": "frontend"},  # Route traffic to pods with this label
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[frontend_deployment]
        ),
    )

    # --- Neo4j Deployment ---
    neo4j_deployment = k8s.apps.v1.Deployment(
        "neo4j",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="neo4j",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "neo4j"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "neo4j"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="neo4j",
                            image=neo4j_mirror_tag.apply(
                                lambda tags: tags[0]
                            ),  # Container image for Neo4j
                            image_pull_policy="IfNotPresent",  # Use cached image if available
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=7474,
                                    protocol="TCP",
                                ),  # HTTP protocol for Neo4j
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=7687,
                                    protocol="TCP",
                                ),  # Bolt protocol for Neo4j
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_AUTH", value=neo4j_auth
                                ),  # Neo4j authentication
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_dbms_memory_heap_maxSize", value="2G"
                                ),  # Set Neo4j heap size to 2GB
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_dbms_memory_pagecache_size", value="1G"
                                ),  # Set Neo4j page cache size to 1GB
                            ],
                            volume_mounts=[
                                k8s.core.v1.VolumeMountArgs(
                                    name="neo4j-storage",
                                    mount_path="/data",
                                ),
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "500m", "memory": "2Gi"},
                                limits={"cpu": "2000m", "memory": "4Gi"},
                            ),
                        ),
                    ],
                    volumes=[
                        k8s.core.v1.VolumeArgs(
                            name="neo4j-storage",
                            persistent_volume_claim=k8s.core.v1.PersistentVolumeClaimVolumeSourceArgs(
                                claim_name=neo4j_pvc.metadata.name,  # Mount the 20Gi PVC
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[namespace, neo4j_pvc]
        ),
    )

    # --- Neo4j Service ---
    neo4j_service = k8s.core.v1.Service(
        "neo4j-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="neo4j",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",  # Internal only
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=7474, target_port=7474, protocol="TCP", name="http"
                ),
                k8s.core.v1.ServicePortArgs(
                    port=7687, target_port=7687, protocol="TCP", name="bolt"
                ),
            ],
            selector={"run": "neo4j"},
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[neo4j_deployment]
        ),
    )

    # --- Model Lineage Deployment ---
    model_lineage_job = k8s.batch.v1.Job(
        "model-lineage",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="model-lineage",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.batch.v1.JobSpecArgs(
            backoff_limit=3,  # Retry up to 4 times on failure
            template=k8s.core.v1.PodTemplateSpecArgs(
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,  # Use Workload Identity for GCP access
                    restart_policy="Never",  # Don't restart pod on completion
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="model-lineage",
                            image=model_lineage_tag.apply(lambda tags: tags[0]),
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_PROJECT", value=project
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_URI",
                                    value="bolt://neo4j:7687",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_USER", value=neo4j_user
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_PASSWORD", value=neo4j_password
                                ),
                                k8s.core.v1.EnvVarArgs(name="HF_TOKEN", value=hf_token),
                            ],
                            args=[
                                "uv",
                                "run",
                                "python",
                                "lineage_scraper.py",
                                "--full",
                                "--limit",
                                "500",
                            ],
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider,
            depends_on=[neo4j_service],
        ),
    )

    # --- Backend Deployment ---
    backend_deployment = k8s.apps.v1.Deployment(
        "backend",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="backend",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "backend"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "backend"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,  # Use KSA for Workload Identity (GCP access)
                    security_context=k8s.core.v1.PodSecurityContextArgs(
                        fs_group=1000,
                    ),
                    volumes=[
                        k8s.core.v1.VolumeArgs(
                            name="persistent-vol",
                            persistent_volume_claim=k8s.core.v1.PersistentVolumeClaimVolumeSourceArgs(
                                claim_name=persistent_pvc.metadata.name,  # Persistent storage
                            ),
                        )
                    ],
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="backend",
                            image=backend_tag.apply(
                                lambda tags: tags[0]
                            ),  # Backend container image
                            image_pull_policy="IfNotPresent",
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=8000,  # Backend server port
                                    protocol="TCP",
                                )
                            ],
                            volume_mounts=[
                                k8s.core.v1.VolumeMountArgs(
                                    name="persistent-vol",
                                    mount_path="/persistent",  # Temporary file storage
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_URI",
                                    value="bolt://neo4j:7687",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_USER", value=neo4j_user
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="NEO4J_PASSWORD", value=neo4j_password
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="OPENAI_API_KEY", value=openai_api_key
                                ),
                                k8s.core.v1.EnvVarArgs(name="HF_TOKEN", value=hf_token),
                            ],
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[model_lineage_job]
        ),
    )

    # --- Backend Service ---
    backend_service = k8s.core.v1.Service(
        "backend-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="backend",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",  # Internal only
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=8000,
                    target_port=8000,
                    protocol="TCP",
                )
            ],
            selector={"run": "backend"},
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[backend_deployment]
        ),
    )

    return frontend_service, backend_service
