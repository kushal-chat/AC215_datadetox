import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { GitBranch, ExternalLink, Download, Heart } from "lucide-react";
import Tree from "react-d3-tree";
import { useMemo } from "react";

// Mock data for testing without backend
const MOCK_NEO4J_DATA = {
  nodes: {
    nodes: [
      {
        model_id: "meta-llama/Llama-2-7b-hf",
        downloads: 5000000,
        likes: 1500,
        pipeline_tag: "text-generation",
        url: "https://huggingface.co/meta-llama/Llama-2-7b-hf",
        tags: ["llama", "meta", "base-model"]
      },
      {
        model_id: "NousResearch/Llama-2-7b-chat-hf",
        downloads: 2000000,
        likes: 800,
        pipeline_tag: "text-generation",
        url: "https://huggingface.co/NousResearch/Llama-2-7b-chat-hf",
        tags: ["llama", "chat", "fine-tuned"]
      },
      {
        model_id: "timdettmers/guanaco-7b",
        downloads: 1500000,
        likes: 600,
        pipeline_tag: "text-generation",
        url: "https://huggingface.co/timdettmers/guanaco-7b",
        tags: ["llama", "qlora", "fine-tuned"]
      },
      {
        model_id: "vineetsharma/qlora-adapter-7b",
        downloads: 500000,
        likes: 200,
        pipeline_tag: "text-generation",
        url: "https://huggingface.co/vineetsharma/qlora-adapter-7b",
        tags: ["adapter", "qlora"]
      },
      {
        model_id: "custom/llama2-medical",
        downloads: 300000,
        likes: 150,
        pipeline_tag: "text-generation",
        url: "https://huggingface.co/custom/llama2-medical",
        tags: ["medical", "domain-specific"]
      }
    ]
  },
  relationships: {
    relationships: [
      {
        source: { model_id: "meta-llama/Llama-2-7b-hf", downloads: 5000000, likes: 1500, url: "https://huggingface.co/meta-llama/Llama-2-7b-hf" },
        relationship: "BASED_ON",
        target: { model_id: "NousResearch/Llama-2-7b-chat-hf", downloads: 2000000, likes: 800, url: "https://huggingface.co/NousResearch/Llama-2-7b-chat-hf" }
      },
      {
        source: { model_id: "meta-llama/Llama-2-7b-hf", downloads: 5000000, likes: 1500, url: "https://huggingface.co/meta-llama/Llama-2-7b-hf" },
        relationship: "FINE_TUNED",
        target: { model_id: "timdettmers/guanaco-7b", downloads: 1500000, likes: 600, url: "https://huggingface.co/timdettmers/guanaco-7b" }
      },
      {
        source: { model_id: "timdettmers/guanaco-7b", downloads: 1500000, likes: 600, url: "https://huggingface.co/timdettmers/guanaco-7b" },
        relationship: "BASED_ON",
        target: { model_id: "vineetsharma/qlora-adapter-7b", downloads: 500000, likes: 200, url: "https://huggingface.co/vineetsharma/qlora-adapter-7b" }
      },
      {
        source: { model_id: "NousResearch/Llama-2-7b-chat-hf", downloads: 2000000, likes: 800, url: "https://huggingface.co/NousResearch/Llama-2-7b-chat-hf" },
        relationship: "FINE_TUNED",
        target: { model_id: "custom/llama2-medical", downloads: 300000, likes: 150, url: "https://huggingface.co/custom/llama2-medical" }
      }
    ]
  },
  queried_model_id: "meta-llama/Llama-2-7b-hf"
};

const USE_MOCK_DATA = false; // Set to false when using real backend

interface Neo4jNode {
  model_id?: string;  // For models
  dataset_id?: string;  // For datasets
  downloads?: number;
  pipeline_tag?: string | null;
  created_at?: string;
  library_name?: string;
  url?: string;
  likes?: number;
  tags?: string[];
  training_datasets?: {  // Datasets extracted from arxiv papers
    arxiv_url: string | null;
    datasets: Array<{
      name: string;
      url?: string;
      description?: string;
    }>;
  };
}

interface Neo4jRelationship {
  source: Neo4jNode;
  relationship: string;
  target: Neo4jNode;
}

interface Neo4jData {
  nodes: {
    nodes: Neo4jNode[];
  };
  relationships: {
    relationships: Neo4jRelationship[];
  };
  queried_model_id?: string;  // The model ID that was queried
}

interface ModelTreeProps {
  neo4jData: Neo4jData | null;
}

interface DatasetInfo {
  id: string;
  url?: string;
  downloads?: number;
  likes?: number;
  relationship?: string;
  source?: 'neo4j' | 'arxiv';  // Where the dataset info came from
  arxiv_url?: string;  // Link to arxiv paper (for arxiv-sourced datasets)
  description?: string;  // Context from paper
}

interface TreeNode {
  name: string;
  attributes?: {
    downloads?: number;
    likes?: number;
    pipeline_tag?: string | null;
    url?: string;
    relationship?: string; // Relationship type from parent
    datasetsJson?: string; // JSON stringified array of datasets
  };
  children?: TreeNode[];
}

// Helper function to get the ID from either a model or dataset
const getNodeId = (node: Neo4jNode): string => {
  return node.model_id || node.dataset_id || 'unknown';
};

// Build hierarchical tree from Neo4j nodes and relationships
// Relationship semantics: source -> RELATIONSHIP -> target means "source is RELATIONSHIP from target"
// Example: modelA -> FINETUNED -> modelB means "modelA is finetuned from modelB" (B is parent)
const buildTreeFromRelationships = (
  nodes: Neo4jNode[],
  relationships: Neo4jRelationship[],
  queriedModelId?: string
): { trees: TreeNode[], hasDatasets: boolean } => {
  if (nodes.length === 0) return { trees: [], hasDatasets: false };

  // Separate models and datasets
  const models = nodes.filter(n => n.model_id);
  const datasets = nodes.filter(n => n.dataset_id && !n.model_id);

  // Create a map of ID to node data (models only for tree structure)
  const nodeMap = new Map<string, Neo4jNode>();
  models.forEach(node => nodeMap.set(getNodeId(node), node));

  // Map of model ID to its associated datasets
  const modelDatasets = new Map<string, DatasetInfo[]>();

  // Build adjacency lists for models only
  // parentMap: child -> Set<parent> (upstream dependencies)
  // childrenMap: parent -> Map<child, relationship> (downstream dependents)
  const parentMap = new Map<string, Map<string, string>>(); // child -> Map<parent, relationship>
  const childrenMap = new Map<string, Map<string, string>>(); // parent -> Map<child, relationship>

  relationships.forEach(rel => {
    const sourceId = getNodeId(rel.source);
    const targetId = getNodeId(rel.target);
    const relType = rel.relationship;

    // Check if this is a model-dataset relationship
    const sourceIsModel = rel.source.model_id;
    const targetIsDataset = rel.target.dataset_id && !rel.target.model_id;

    if (sourceIsModel && targetIsDataset) {
      // Model -> Dataset relationship (e.g., TRAINED_ON)
      // Attach dataset to model
      if (!modelDatasets.has(sourceId)) {
        modelDatasets.set(sourceId, []);
      }
      modelDatasets.get(sourceId)!.push({
        id: targetId,
        url: rel.target.url,
        downloads: rel.target.downloads,
        likes: rel.target.likes,
        relationship: relType,
        source: 'neo4j',  // Mark as coming from neo4j relationships
      });
      return; // Don't add to tree structure
    }

    // Model-to-model relationships only
    if (!sourceIsModel || targetIsDataset) return;

    // source -> RELATIONSHIP -> target means target is the parent
    // Store parent relationship
    if (!parentMap.has(sourceId)) {
      parentMap.set(sourceId, new Map());
    }
    parentMap.get(sourceId)!.set(targetId, relType);

    // Store child relationship
    if (!childrenMap.has(targetId)) {
      childrenMap.set(targetId, new Map());
    }
    childrenMap.get(targetId)!.set(sourceId, relType);
  });

  // Merge arxiv-extracted training datasets into modelDatasets map
  models.forEach(node => {
    const modelId = getNodeId(node);

    if (node.training_datasets && node.training_datasets.datasets.length > 0) {
      if (!modelDatasets.has(modelId)) {
        modelDatasets.set(modelId, []);
      }

      // Add each arxiv-extracted dataset
      node.training_datasets.datasets.forEach(dataset => {
        modelDatasets.get(modelId)!.push({
          id: dataset.name,
          url: dataset.url,
          relationship: 'TRAINED_ON',  // Default relationship for training datasets
          source: 'arxiv',  // Mark as coming from arxiv extraction
          arxiv_url: node.training_datasets!.arxiv_url || undefined,
          description: dataset.description,
        });
      });
    }
  });

  // Find the root: the node with no parents (highest in the hierarchy)
  // Only consider models for root candidates
  const rootCandidates = models.filter(node => {
    const nodeId = getNodeId(node);
    const hasNoParent = !parentMap.has(nodeId) || parentMap.get(nodeId)!.size === 0;
    return hasNoParent;
  });

  // Always use the true root of the hierarchy (not the queried model)
  // The queried model is only used for highlighting purposes
  let rootNode: Neo4jNode | undefined;
  if (rootCandidates.length > 0) {
    // Prefer models over datasets when selecting root
    // Datasets without parents are typically training data, not the top of the model hierarchy
    const modelRoots = rootCandidates.filter(n => n.model_id); // Models have model_id
    const datasetRoots = rootCandidates.filter(n => n.dataset_id && !n.model_id); // Pure datasets

    if (modelRoots.length > 0) {
      // Use the model with the most children (most influential in the tree)
      rootNode = modelRoots.sort((a, b) => {
        const aChildren = childrenMap.get(getNodeId(a))?.size || 0;
        const bChildren = childrenMap.get(getNodeId(b))?.size || 0;
        return bChildren - aChildren;
      })[0];
    } else {
      // If only datasets available, use the first one
      rootNode = datasetRoots[0];
    }
  } else if (nodes.length > 0) {
    // Fallback to first node if no clear root
    rootNode = nodes[0];
  }

  if (!rootNode) return { trees: [], hasDatasets: false };

  // Check if any model has datasets
  const hasAnyDatasets = modelDatasets.size > 0;

  // Build tree recursively from root downwards
  const buildNode = (modelId: string, relationship?: string, visited = new Set<string>()): TreeNode | null => {
    if (visited.has(modelId)) return null; // Prevent cycles
    visited.add(modelId);

    const nodeData = nodeMap.get(modelId);
    if (!nodeData) return null;

    // Get datasets associated with this model
    const datasets = modelDatasets.get(modelId) || [];

    const treeNode: TreeNode = {
      name: getNodeId(nodeData),
      attributes: {
        downloads: nodeData.downloads,
        likes: nodeData.likes,
        pipeline_tag: nodeData.pipeline_tag,
        url: nodeData.url,
        relationship: relationship, // Relationship from parent to this node
        datasetsJson: datasets.length > 0 ? JSON.stringify(datasets) : undefined,
      },
    };

    // Add children (models that depend on this one)
    const children = childrenMap.get(modelId);
    if (children && children.size > 0) {
      treeNode.children = Array.from(children.entries())
        .map(([childId, relType]) => buildNode(childId, relType, new Set(visited)))
        .filter((child): child is TreeNode => child !== null);
    }

    return treeNode;
  };

  const tree = buildNode(getNodeId(rootNode));
  return {
    trees: tree ? [tree] : [],
    hasDatasets: hasAnyDatasets
  };
};

// Custom node rendering for the tree
const renderCustomNode = ({ nodeDatum, toggleNode, queriedModelId }: any) => {
  const attrs = nodeDatum.attributes || {};
  const modelName = nodeDatum.name;
  const relationship = attrs.relationship;
  const isQueriedModel = modelName === queriedModelId;

  // Parse datasets if available
  const datasets: DatasetInfo[] = attrs.datasetsJson ? JSON.parse(attrs.datasetsJson) : [];

  // Check if node has children (to determine dataset placement)
  const hasChildren = nodeDatum.children && nodeDatum.children.length > 0;

  const handleClick = () => {
    if (attrs.url) {
      window.open(attrs.url, '_blank', 'noopener,noreferrer');
    }
  };

  // Split model name into lines if too long
  const maxCharsPerLine = 20;
  const nameParts = (modelName || 'Unknown').split('/');
  const lines: string[] = [];

  nameParts.forEach((part, idx) => {
    if (part.length > maxCharsPerLine) {
      // Split long parts into chunks
      for (let i = 0; i < part.length; i += maxCharsPerLine) {
        lines.push(part.slice(i, i + maxCharsPerLine));
      }
    } else {
      // Add slash before non-first parts
      lines.push((idx > 0 ? '/' : '') + part);
    }
  });

  // Calculate card height with proper spacing
  const baseHeight = 45;
  const nameHeight = lines.length * 14;
  const pipelineHeight = attrs.pipeline_tag ? 14 : 0; // Padding for pipeline tag
  const statsHeight = 18;
  const cardHeight = baseHeight + nameHeight + pipelineHeight + statsHeight;
  const cardY = -cardHeight / 2;

  return (
    <g onClick={handleClick} style={{ cursor: attrs.url ? 'pointer' : 'default' }}>
      {/* Relationship label - positioned above the node (on the incoming edge) */}
      {relationship && (
        <g transform="translate(0, -90)" className="relationship-label">
          <rect
            x={-55}
            y={-12}
            width={110}
            height={24}
            rx={6}
            fill="#0f172a"
            stroke="#60a5fa"
            strokeWidth={1.5}
          />
          <text
            fill="#60a5fa"
            strokeWidth="0"
            x="0"
            y="4"
            fontSize="11"
            fontWeight="600"
            textAnchor="middle"
            style={{ pointerEvents: 'none' }}
          >
            {relationship.replace(/_/g, ' ')}
          </text>
        </g>
      )}

      {/* Background card - highlight if queried model */}
      <rect
        x={-75}
        y={cardY}
        width={150}
        height={cardHeight}
        rx={8}
        fill={isQueriedModel ? "#1e3a5f" : "#1e293b"}
        stroke={isQueriedModel ? "#fbbf24" : "#3b82f6"}
        strokeWidth={isQueriedModel ? 3 : 2}
        className="node-card"
        style={{ transition: 'all 0.2s' }}
      />

      {/* Glow effect for queried model or dataset indicator */}
      {isQueriedModel && (
        <rect
          x={-75}
          y={cardY}
          width={150}
          height={cardHeight}
          rx={8}
          fill="none"
          stroke="#fbbf24"
          strokeWidth={1}
          opacity={0.5}
          style={{ filter: 'blur(4px)' }}
        />
      )}

      {/* Floating datasets - positioned to avoid overlap */}
      {datasets.map((dataset, idx) => {
        // Place datasets to the right if node has children (to avoid overlap)
        // Otherwise place below
        const datasetWidth = 130;
        const datasetHeight = 75;
        const datasetX = hasChildren
          ? 85  // To the right of the model card
          : -65; // Center below the model card
        const datasetY = hasChildren
          ? cardY + (idx * 85)  // Stack vertically on the right
          : cardY + cardHeight + 20 + (idx * 85); // Stack below

        const datasetNameParts = dataset.id.split('/');
        const datasetShortName = datasetNameParts[datasetNameParts.length - 1] || dataset.id;

        return (
          <g key={dataset.id}>
            {/* Connection line from model to dataset */}
            <line
              x1={hasChildren ? 75 : 0} // Right edge if has children, center bottom otherwise
              y1={hasChildren ? cardY + cardHeight / 2 : cardY + cardHeight}
              x2={datasetX + (hasChildren ? 0 : datasetWidth / 2)}
              y2={hasChildren ? datasetY + datasetHeight / 2 : datasetY}
              stroke="#64748b"
              strokeWidth={1.5}
              strokeDasharray="4 2"
              opacity={0.5}
            />

            {/* Dataset card */}
            <rect
              x={datasetX}
              y={datasetY}
              width={datasetWidth}
              height={datasetHeight}
              rx={6}
              fill={dataset.source === 'arxiv' ? "#faf5ff" : "white"}
              stroke={dataset.source === 'arxiv' ? "#8b5cf6" : "#64748b"}
              strokeWidth={1.5}
              onClick={() => dataset.url && window.open(dataset.url, '_blank', 'noopener,noreferrer')}
              style={{ cursor: dataset.url ? 'pointer' : 'default', transition: 'all 0.2s' }}
              className="dataset-card"
            />

            {/* "Dataset" label in top left corner */}
            <text
              fill={dataset.source === 'arxiv' ? "#8b5cf6" : "#64748b"}
              strokeWidth="0"
              x={datasetX + 6}
              y={datasetY + 12}
              fontSize="7"
              fontWeight="600"
              style={{ pointerEvents: 'none' }}
            >
              {dataset.source === 'arxiv' ? 'Dataset (arXiv)' : 'Dataset'}
            </text>

            {/* Dataset name */}
            <text
              fill="#334155"
              strokeWidth="0"
              x={datasetX + datasetWidth / 2}
              y={datasetY + 32}
              fontSize="9"
              fontWeight="600"
              textAnchor="middle"
              style={{ pointerEvents: 'none' }}
            >
              {datasetShortName.length > 15 ? datasetShortName.substring(0, 15) + '...' : datasetShortName}
            </text>

            {/* Relationship label */}
            {dataset.relationship && (
              <text
                fill="#64748b"
                strokeWidth="0"
                x={datasetX + datasetWidth / 2}
                y={datasetY + 45}
                fontSize="7"
                textAnchor="middle"
                opacity={0.8}
                style={{ pointerEvents: 'none' }}
              >
                {dataset.relationship.replace(/_/g, ' ')}
              </text>
            )}

            {/* Stats */}
            {(dataset.downloads !== undefined || dataset.likes !== undefined) && (
              <g transform={`translate(${datasetX + datasetWidth / 2}, ${datasetY + 58})`}>
                {dataset.downloads !== undefined && (
                  <text fill="#64748b" strokeWidth="0" x="-25" y="0" fontSize="7" style={{ pointerEvents: 'none' }}>
                    ↓ {(dataset.downloads / 1000).toFixed(0)}k
                  </text>
                )}
                {dataset.likes !== undefined && (
                  <text fill="#64748b" strokeWidth="0" x="10" y="0" fontSize="7" style={{ pointerEvents: 'none' }}>
                    ♥ {dataset.likes}
                  </text>
                )}
              </g>
            )}

            {/* arXiv paper icon (if available) */}
            {dataset.arxiv_url && (
              <g
                transform={`translate(${datasetX + datasetWidth - 15}, ${datasetY + 28})`}
                onClick={(e) => {
                  e.stopPropagation();
                  window.open(dataset.arxiv_url, '_blank', 'noopener,noreferrer');
                }}
                style={{ cursor: 'pointer' }}
              >
                <circle r="7" fill="#8b5cf6" opacity="0.9" />
                <text fill="white" strokeWidth="0" x="0" y="2.5" fontSize="7" textAnchor="middle" fontWeight="bold" style={{ pointerEvents: 'none' }}>
                  📄
                </text>
              </g>
            )}

            {/* External link icon for dataset */}
            {dataset.url && (
              <g
                transform={`translate(${datasetX + datasetWidth - 15}, ${datasetY + 12})`}
                onClick={(e) => {
                  e.stopPropagation();
                  window.open(dataset.url, '_blank', 'noopener,noreferrer');
                }}
                style={{ cursor: 'pointer' }}
              >
                <circle r="7" fill={dataset.source === 'arxiv' ? "#8b5cf6" : "#64748b"} opacity="0.9" />
                <text fill="white" strokeWidth="0" x="0" y="2.5" fontSize="7" textAnchor="middle" style={{ pointerEvents: 'none' }}>
                  ↗
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* Model name (multi-line with full name) */}
      <g>
        {lines.map((line, idx) => (
          <text
            key={idx}
            fill={isQueriedModel ? "#fbbf24" : "#f1f5f9"}
            strokeWidth="0"
            x="0"
            y={cardY + 20 + idx * 14}
            fontSize="11"
            fontWeight={isQueriedModel ? "bold" : "600"}
            textAnchor="middle"
            style={{ pointerEvents: 'none' }}
          >
            {line}
          </text>
        ))}
      </g>

      {/* Pipeline tag */}
      {attrs.pipeline_tag && (
        <text
          fill="#94a3b8"
          strokeWidth="0"
          x="0"
          y={cardY + 20 + lines.length * 14 + 8}
          fontSize="9"
          textAnchor="middle"
          style={{ pointerEvents: 'none' }}
        >
          {attrs.pipeline_tag}
        </text>
      )}

      {/* Stats row - centered with equal spacing */}
      <g transform={`translate(0, ${cardY + cardHeight - 15})`}>
        {(() => {
          const downloadsText = attrs.downloads !== undefined
            ? attrs.downloads < 100000
              ? attrs.downloads.toLocaleString()
              : `${(attrs.downloads / 1000000).toFixed(1)}M`
            : null;
          const likesText = attrs.likes !== undefined ? attrs.likes.toString() : null;

          if (downloadsText && likesText) {
            // Equal spacing on both sides - gap of 12px between them
            const gap = 12;
            return (
              <>
                <text fill="#94a3b8" strokeWidth="0" x={-gap / 2} y="0" fontSize="9" textAnchor="end" style={{ pointerEvents: 'none' }}>
                  ↓ {downloadsText}
                </text>
                <text fill="#f472b6" strokeWidth="0" x={gap / 2} y="0" fontSize="9" textAnchor="start" style={{ pointerEvents: 'none' }}>
                  ♥ {likesText}
                </text>
              </>
            );
          } else if (downloadsText) {
            return (
              <text fill="#94a3b8" strokeWidth="0" x="0" y="0" fontSize="9" textAnchor="middle" style={{ pointerEvents: 'none' }}>
                ↓ {downloadsText}
              </text>
            );
          } else if (likesText) {
            return (
              <text fill="#f472b6" strokeWidth="0" x="0" y="0" fontSize="9" textAnchor="middle" style={{ pointerEvents: 'none' }}>
                ♥ {likesText}
              </text>
            );
          }
          return null;
        })()}
      </g>

      {/* External link icon - moved down to avoid edge overlap */}
      {attrs.url && (
        <g transform={`translate(60, ${cardY + cardHeight - 15})`}>
          <circle r="9" fill={isQueriedModel ? "#fbbf24" : "#3b82f6"} opacity="0.9" />
          <text fill={isQueriedModel ? "#1e293b" : "white"} strokeWidth="0" x="0" y="3.5" fontSize="9" textAnchor="middle" style={{ pointerEvents: 'none' }}>
            ↗
          </text>
        </g>
      )}

      {/* Star icon for queried model */}
      {isQueriedModel && (
        <g transform={`translate(-65, ${cardY + 8})`}>
          <text fill="#fbbf24" strokeWidth="0" x="0" y="4" fontSize="14" textAnchor="middle" style={{ pointerEvents: 'none' }}>
            ⭐
          </text>
        </g>
      )}
    </g>
  );
};


const Neo4jModelNode = ({ node }: { node: Neo4jNode }) => {
  return (
    <div className="animate-fade-in py-2 px-3 hover:bg-muted/50 rounded-lg transition-colors border border-border mb-2">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-secondary flex-shrink-0" />
            <a
              href={node.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium hover:text-secondary transition-colors"
            >
              {getNodeId(node)}
            </a>
            <ExternalLink className="w-3 h-3 text-muted-foreground" />
          </div>

          {node.pipeline_tag && (
            <Badge variant="outline" className="text-xs mt-2">
              {node.pipeline_tag}
            </Badge>
          )}

          <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
            {node.downloads !== undefined && (
              <div className="flex items-center gap-1">
                <Download className="w-3 h-3" />
                <span>{(node.downloads || 0).toLocaleString()}</span>
              </div>
            )}
            {node.likes !== undefined && (
              <div className="flex items-center gap-1">
                <Heart className="w-3 h-3" />
                <span>{node.likes}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const ModelTree = ({ neo4jData }: ModelTreeProps) => {
  // Use mock data if enabled and no real data provided
  const effectiveData = USE_MOCK_DATA && (!neo4jData || neo4jData.nodes.nodes.length === 0)
    ? MOCK_NEO4J_DATA
    : neo4jData;

  if (!effectiveData || !effectiveData.nodes?.nodes || effectiveData.nodes.nodes.length === 0) {
    return (
      <Card className="bg-card border-border shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-secondary" />
            Model Lineage Tree
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <p className="text-sm">No model lineage data yet.</p>
            <p className="text-xs mt-2">Ask about a specific model to see its relationships!</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const nodes = effectiveData.nodes.nodes;
  const relationships = effectiveData.relationships?.relationships || [];

  // Get the queried model ID from the backend response
  const queriedModelId = effectiveData.queried_model_id || nodes[0]?.model_id;

  // Build tree structure from relationships
  const { treeData, hasDatasets } = useMemo(() => {
    const result = buildTreeFromRelationships(nodes, relationships, queriedModelId);
    const trees = result.trees;

    if (trees.length === 1) {
      return { treeData: trees[0], hasDatasets: result.hasDatasets };
    } else if (trees.length > 1) {
      // Multiple trees - shouldn't happen with the new logic, but handle it
      return { treeData: trees[0], hasDatasets: result.hasDatasets };
    }

    // No relationships - return first node
    return {
      treeData: {
        name: nodes[0] ? getNodeId(nodes[0]) : "No models",
        attributes: {
          downloads: nodes[0]?.downloads,
          likes: nodes[0]?.likes,
          pipeline_tag: nodes[0]?.pipeline_tag,
          url: nodes[0]?.url,
        },
      },
      hasDatasets: result.hasDatasets
    };
  }, [nodes, relationships, queriedModelId]);

  const hasRelationships = relationships.length > 0;

  // Wrapper to pass queriedModelId to renderCustomNode
  const renderNodeWithContext = (nodeProps: any) =>
    renderCustomNode({ ...nodeProps, queriedModelId });

  return (
    <Card className="bg-card border-border shadow-md h-full flex flex-col">
      <CardContent className="p-0 flex-1 flex flex-col">
        {hasRelationships ? (
          <div className="w-full flex-1 flex flex-col border-t border-border bg-slate-950 relative overflow-hidden">
            <style>{`
              .rd3t-link {
                stroke: #3b82f6 !important;
                stroke-width: 2.5 !important;
                fill: none !important;
              }
              .rd3t-node:hover .node-card {
                filter: brightness(1.2);
                transition: filter 0.3s ease;
              }
              .rd3t-node:hover .dataset-card {
                filter: brightness(1.3);
                transition: filter 0.3s ease;
              }
              .relationship-label {
                opacity: 0.8;
                transition: opacity 0.3s ease;
                pointer-events: none;
              }
            `}</style>

            {/* Tree visualization - takes up most space */}
            <div className="flex-1 relative">
              <Tree
                data={treeData}
                orientation="vertical"
                pathFunc="step"
                translate={{ x: 400, y: 60 }}
                nodeSize={hasDatasets ? { x: 200, y: 180 } : { x: 180, y: 160 }}
                renderCustomNodeElement={renderNodeWithContext}
                separation={hasDatasets ? { siblings: 1.1, nonSiblings: 1.2 } : { siblings: 1.0, nonSiblings: 1.1 }}
                zoomable
                collapsible={false}
                initialDepth={undefined}
              />
              <div className="absolute bottom-3 right-3 text-xs text-muted-foreground bg-slate-900/90 px-2 py-1.5 rounded text-[11px]">
                💡 Click nodes • Scroll zoom • Drag pan • Hover edges
              </div>
            </div>

            {/* Compact relationships footer */}
            <div className="border-t border-slate-800 bg-slate-900/50 px-4 py-2">
              <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap">
                <span className="font-semibold text-slate-400">{relationships.length} relationships:</span>
                {relationships.slice(0, 5).map((rel: Neo4jRelationship, index: number) => {
                  const sourceId = getNodeId(rel.source);
                  const targetId = getNodeId(rel.target);
                  const sourceName = sourceId.split('/').pop() || 'Unknown';
                  const targetName = targetId.split('/').pop() || 'Unknown';
                  return (
                    <div key={index} className="flex items-center gap-1.5">
                      <span className="font-mono text-[10px] truncate max-w-[100px]">{sourceName}</span>
                      <Badge variant="secondary" className="text-[9px] px-1 py-0 h-4">
                        {rel.relationship}
                      </Badge>
                      <span className="font-mono text-[10px] truncate max-w-[100px]">{targetName}</span>
                    </div>
                  );
                })}
                {relationships.length > 5 && (
                  <span className="text-[10px] text-slate-500">+{relationships.length - 5} more</span>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-2 p-4">
            {nodes.map((node: Neo4jNode, index: number) => (
              <Neo4jModelNode key={index} node={node} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ModelTree;
