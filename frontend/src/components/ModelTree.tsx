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
  model_id: string;
  downloads?: number;
  pipeline_tag?: string | null;
  created_at?: string;
  library_name?: string;
  url?: string;
  likes?: number;
  tags?: string[];
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

interface TreeNode {
  name: string;
  attributes?: {
    downloads?: number;
    likes?: number;
    pipeline_tag?: string | null;
    url?: string;
    relationship?: string; // Relationship type from parent
  };
  children?: TreeNode[];
}

// Build hierarchical tree from Neo4j nodes and relationships
// Relationship semantics: source -> RELATIONSHIP -> target means "source is RELATIONSHIP from target"
// Example: modelA -> FINETUNED -> modelB means "modelA is finetuned from modelB" (B is parent)
const buildTreeFromRelationships = (
  nodes: Neo4jNode[],
  relationships: Neo4jRelationship[],
  queriedModelId?: string
): TreeNode[] => {
  if (nodes.length === 0) return [];

  // Create a map of model_id to node data
  const nodeMap = new Map<string, Neo4jNode>();
  nodes.forEach(node => nodeMap.set(node.model_id, node));

  // Build adjacency lists
  // parentMap: child -> Set<parent> (upstream dependencies)
  // childrenMap: parent -> Map<child, relationship> (downstream dependents)
  const parentMap = new Map<string, Map<string, string>>(); // child -> Map<parent, relationship>
  const childrenMap = new Map<string, Map<string, string>>(); // parent -> Map<child, relationship>

  relationships.forEach(rel => {
    const sourceId = rel.source.model_id;
    const targetId = rel.target.model_id;
    const relType = rel.relationship;

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

  // Find the root: the node with no parents (highest in the hierarchy)
  const rootCandidates = nodes.filter(node => {
    const hasNoParent = !parentMap.has(node.model_id) || parentMap.get(node.model_id)!.size === 0;
    return hasNoParent;
  });

  // Always use the true root of the hierarchy (not the queried model)
  // The queried model is only used for highlighting purposes
  let rootNode: Neo4jNode | undefined;
  if (rootCandidates.length > 0) {
    // Use the first root candidate (highest level parent)
    rootNode = rootCandidates[0];
  } else if (nodes.length > 0) {
    // Fallback to first node if no clear root
    rootNode = nodes[0];
  }

  if (!rootNode) return [];

  // Build tree recursively from root downwards
  const buildNode = (modelId: string, relationship?: string, visited = new Set<string>()): TreeNode | null => {
    if (visited.has(modelId)) return null; // Prevent cycles
    visited.add(modelId);

    const nodeData = nodeMap.get(modelId);
    if (!nodeData) return null;

    const treeNode: TreeNode = {
      name: nodeData.model_id,
      attributes: {
        downloads: nodeData.downloads,
        likes: nodeData.likes,
        pipeline_tag: nodeData.pipeline_tag,
        url: nodeData.url,
        relationship: relationship, // Relationship from parent to this node
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

  const tree = buildNode(rootNode.model_id);
  return tree ? [tree] : [];
};

// Custom node rendering for the tree
const renderCustomNode = ({ nodeDatum, toggleNode, queriedModelId }: any) => {
  const attrs = nodeDatum.attributes || {};
  const modelName = nodeDatum.name;
  const relationship = attrs.relationship;
  const isQueriedModel = modelName === queriedModelId;

  const handleClick = () => {
    if (attrs.url) {
      window.open(attrs.url, '_blank', 'noopener,noreferrer');
    }
  };

  // Split model name into lines if too long
  const maxCharsPerLine = 20;
  const nameParts = modelName.split('/');
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

      {/* Glow effect for queried model */}
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

      {/* Stats row */}
      <g transform={`translate(0, ${cardY + cardHeight - 15})`}>
        {attrs.downloads !== undefined && (
          <>
            <text fill="#94a3b8" strokeWidth="0" x="-30" y="0" fontSize="9" textAnchor="start" style={{ pointerEvents: 'none' }}>
              ↓ {(attrs.downloads / 1000000).toFixed(1)}M
            </text>
          </>
        )}
        {attrs.likes !== undefined && (
          <text fill="#f472b6" strokeWidth="0" x="18" y="0" fontSize="9" textAnchor="start" style={{ pointerEvents: 'none' }}>
            ♥ {attrs.likes}
          </text>
        )}
      </g>

      {/* External link icon */}
      {attrs.url && (
        <g transform={`translate(60, ${cardY + 8})`}>
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
              {node.model_id}
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
  const treeData = useMemo(() => {
    const trees = buildTreeFromRelationships(nodes, relationships, queriedModelId);

    if (trees.length === 1) {
      return trees[0];
    } else if (trees.length > 1) {
      // Multiple trees - shouldn't happen with the new logic, but handle it
      return trees[0];
    }

    // No relationships - return first node
    return {
      name: nodes[0]?.model_id || "No models",
      attributes: {
        downloads: nodes[0]?.downloads,
        likes: nodes[0]?.likes,
        pipeline_tag: nodes[0]?.pipeline_tag,
        url: nodes[0]?.url,
      },
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
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
              }
              .rd3t-link:hover {
                stroke: #60a5fa !important;
                stroke-width: 4 !important;
                filter: drop-shadow(0 0 12px rgba(59, 130, 246, 0.9));
              }
              .rd3t-node:hover .node-card {
                filter: brightness(1.2);
                transition: filter 0.3s ease;
              }
              .relationship-label {
                opacity: 0;
                transition: opacity 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                pointer-events: none;
              }
              /* Show relationship label when hovering over the edge */
              .rd3t-link:hover ~ .rd3t-node .relationship-label,
              g:hover > .rd3t-link ~ .rd3t-node .relationship-label {
                opacity: 1 !important;
              }
            `}</style>

            {/* Tree visualization - takes up most space */}
            <div className="flex-1 relative">
              <Tree
                data={treeData}
                orientation="vertical"
                pathFunc="step"
                translate={{ x: 400, y: 60 }}
                nodeSize={{ x: 180, y: 160 }}
                renderCustomNodeElement={renderNodeWithContext}
                separation={{ siblings: 1.2, nonSiblings: 1.5 }}
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
                {relationships.slice(0, 5).map((rel: Neo4jRelationship, index: number) => (
                  <div key={index} className="flex items-center gap-1.5">
                    <span className="font-mono text-[10px] truncate max-w-[100px]">{rel.source.model_id.split('/').pop()}</span>
                    <Badge variant="secondary" className="text-[9px] px-1 py-0 h-4">
                      {rel.relationship}
                    </Badge>
                    <span className="font-mono text-[10px] truncate max-w-[100px]">{rel.target.model_id.split('/').pop()}</span>
                  </div>
                ))}
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
