import { useState, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  Panel,
  useReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Card, CardContent } from '@/components/ui/card';
import { GitBranch } from 'lucide-react';
import ModelNode, { ModelNodeData, DatasetInfo } from './ModelNode';
import { useGraphLayout } from '@/hooks/useGraphLayout';

interface Neo4jNode {
  model_id?: string;
  dataset_id?: string;
  downloads?: number;
  pipeline_tag?: string | null;
  created_at?: string;
  library_name?: string;
  url?: string;
  likes?: number;
  tags?: string[];
  training_datasets?: {
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
  queried_model_id?: string;
}

interface ModelTreeProps {
  neo4jData: Neo4jData | null;
}

const nodeTypes = {
  model: ModelNode,
};

// Helper to get node ID
const getNodeId = (node: Neo4jNode): string => {
  return node.model_id || node.dataset_id || 'unknown';
};

// Build datasets for each model
const buildModelDatasets = (
  nodes: Neo4jNode[],
  relationships: Neo4jRelationship[]
): Map<string, DatasetInfo[]> => {
  const modelDatasets = new Map<string, DatasetInfo[]>();

  // Process neo4j dataset relationships
  relationships.forEach(rel => {
    const sourceId = getNodeId(rel.source);
    const targetId = getNodeId(rel.target);
    const sourceIsModel = rel.source.model_id;
    const targetIsDataset = rel.target.dataset_id && !rel.target.model_id;

    if (sourceIsModel && targetIsDataset) {
      if (!modelDatasets.has(sourceId)) {
        modelDatasets.set(sourceId, []);
      }
      modelDatasets.get(sourceId)!.push({
        id: targetId,
        url: rel.target.url,
        downloads: rel.target.downloads,
        likes: rel.target.likes,
        relationship: rel.relationship,
        source: 'neo4j',
      });
    }
  });

  // Merge arxiv-extracted datasets
  nodes.forEach(node => {
    const modelId = getNodeId(node);
    if (node.training_datasets && node.training_datasets.datasets.length > 0) {
      if (!modelDatasets.has(modelId)) {
        modelDatasets.set(modelId, []);
      }
      node.training_datasets.datasets.forEach(dataset => {
        modelDatasets.get(modelId)!.push({
          id: dataset.name,
          url: dataset.url,
          relationship: 'TRAINED_ON',
          source: 'arxiv',
          arxiv_url: node.training_datasets!.arxiv_url || undefined,
          description: dataset.description,
        });
      });
    }
  });

  return modelDatasets;
};

const ModelTreeFlowInner = ({ neo4jData }: ModelTreeProps) => {
  const { fitView } = useReactFlow();
  const { getLayoutedElements } = useGraphLayout();
  const [nodes, setNodes] = useState<Node<ModelNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    if (!neo4jData || !neo4jData.nodes?.nodes || neo4jData.nodes.nodes.length === 0) {
      setNodes([]);
      setEdges([]);
      return;
    }

    const models = neo4jData.nodes.nodes.filter(n => n.model_id);
    const relationships = neo4jData.relationships?.relationships || [];
    const queriedModelId = neo4jData.queried_model_id || models[0]?.model_id;

    // Build dataset information
    const modelDatasets = buildModelDatasets(models, relationships);

    // Convert to react-flow nodes
    const flowNodes: Node<ModelNodeData>[] = models.map(node => {
      const modelId = getNodeId(node);
      const datasets = modelDatasets.get(modelId) || [];
      // Calculate node height based on number of datasets (up to 3 shown)
      const datasetCount = Math.min(datasets.length, 3);
      const baseHeight = 200;
      const datasetHeight = datasetCount * 35; // ~35px per dataset line
      const totalHeight = baseHeight + datasetHeight;

      return {
        id: modelId,
        type: 'model',
        data: {
          model_id: modelId,
          downloads: node.downloads,
          likes: node.likes,
          pipeline_tag: node.pipeline_tag,
          url: node.url,
          isQueried: modelId === queriedModelId,
          datasets: datasets,
        },
        position: { x: 0, y: 0 }, // Will be set by layout
        width: 250,
        height: totalHeight,
      };
    });

    // Convert to react-flow edges
    const flowEdges: Edge[] = [];
    const processedEdges = new Set<string>();

    relationships.forEach(rel => {
      const sourceId = getNodeId(rel.source);
      const targetId = getNodeId(rel.target);
      const sourceIsModel = rel.source.model_id;
      const targetIsModel = rel.target.model_id;

      // Only model-to-model relationships
      if (sourceIsModel && targetIsModel) {
        const edgeId = `${sourceId}-${targetId}`;
        if (!processedEdges.has(edgeId)) {
          flowEdges.push({
            id: edgeId,
            source: sourceId,
            target: targetId,
            label: rel.relationship.replace(/_/g, ' '),
            type: 'smoothstep',
            style: { stroke: '#3b82f6', strokeWidth: 2 },
            labelStyle: { fill: '#94a3b8', fontSize: 11, fontWeight: 600 },
            labelBgStyle: { fill: '#1e293b', opacity: 0.8 },
          });
          processedEdges.add(edgeId);
        }
      }
    });

    // Apply automatic layout
    console.log('Applying layout to', flowNodes.length, 'nodes and', flowEdges.length, 'edges');
    getLayoutedElements(flowNodes, flowEdges, {
      direction: 'UP',  // Upstream at top, downstream at bottom
      nodeSpacing: 100,
      layerSpacing: 120,
    }).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
      console.log('Layout complete. First node position:', layoutedNodes[0]?.position);
      setNodes(layoutedNodes);
      setEdges(layoutedEdges);
      // Fit view after layout
      setTimeout(() => fitView({ padding: 0.2 }), 100);
    }).catch(error => {
      console.error('Layout failed:', error);
      // Use simple fallback layout
      const fallbackNodes = flowNodes.map((node, i) => ({
        ...node,
        position: { x: (i % 3) * 400, y: Math.floor(i / 3) * 300 },
      }));
      setNodes(fallbackNodes);
      setEdges(flowEdges);
    });
  }, [neo4jData, getLayoutedElements, fitView]);

  const totalNodes = nodes.length;
  const totalEdges = edges.length;
  const datasetsCount = nodes.reduce((sum, node) => sum + (node.data.datasets?.length || 0), 0);

  if (nodes.length === 0) {
    return (
      <Card className="bg-card border-border shadow-md h-full flex flex-col">
        <CardContent className="p-8 flex-1 flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">No model lineage data yet.</p>
            <p className="text-xs mt-2">Ask about a specific model to see its relationships!</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-card border-border shadow-md h-full flex flex-col">
      <CardContent className="p-0 flex-1 flex flex-col relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.1}
          maxZoom={2}
          defaultEdgeOptions={{
            style: { strokeWidth: 2, stroke: '#3b82f6' },
          }}
        >
          <Background color="#3b82f6" gap={16} />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              if (node.data.isQueried) return '#fbbf24';
              return '#3b82f6';
            }}
            maskColor="rgba(0, 0, 0, 0.8)"
          />
          <Panel position="bottom-right" className="bg-slate-900/90 px-3 py-2 rounded-lg text-xs text-slate-300 space-y-1">
            <div>Models: {totalNodes}</div>
            <div>Relationships: {totalEdges}</div>
            <div>Datasets: {datasetsCount}</div>
          </Panel>
        </ReactFlow>
      </CardContent>
    </Card>
  );
};

const ModelTreeNew = (props: ModelTreeProps) => {
  return (
    <ReactFlowProvider>
      <ModelTreeFlowInner {...props} />
    </ReactFlowProvider>
  );
};

export default ModelTreeNew;
