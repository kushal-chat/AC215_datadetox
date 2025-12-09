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
import ModelNode, { ModelNodeData, DatasetInfo, DatasetRiskLevel } from './ModelNode';
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

interface DatasetRiskEntry {
  model_id: string;
  arxiv_url?: string;
  datasets: Array<{
    name: string;
    risk_level?: DatasetRiskLevel;
    indicators?: string[];
  }>;
}

export interface DatasetRiskContext {
  models?: DatasetRiskEntry[];
}

interface ModelTreeProps {
  neo4jData: Neo4jData | null;
  datasetRisk?: DatasetRiskContext | null;
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
  relationships: Neo4jRelationship[],
  datasetRisk?: DatasetRiskContext | null,
): Map<string, DatasetInfo[]> => {
  const modelDatasets = new Map<string, DatasetInfo[]>();
  const riskLookup = new Map<string, Map<string, { risk_level?: DatasetRiskLevel; indicators?: string[] }>>();

  datasetRisk?.models?.forEach(model => {
    const key = model.model_id.toLowerCase();
    const datasetMap = new Map<string, { risk_level?: DatasetRiskLevel; indicators?: string[] }>();
    model.datasets.forEach(ds => {
      datasetMap.set(ds.name.toLowerCase(), {
        risk_level: ds.risk_level || 'unknown',
        indicators: ds.indicators,
      });
    });
    riskLookup.set(key, datasetMap);
  });

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
    const modelRiskMap = riskLookup.get(modelId.toLowerCase());
    if (node.training_datasets && node.training_datasets.datasets.length > 0) {
      if (!modelDatasets.has(modelId)) {
        modelDatasets.set(modelId, []);
      }
      node.training_datasets.datasets.forEach(dataset => {
        const riskMeta = modelRiskMap?.get(dataset.name.toLowerCase());
        modelDatasets.get(modelId)!.push({
          id: dataset.name,
          url: dataset.url,
          relationship: 'TRAINED_ON',
          source: 'arxiv',
          arxiv_url: node.training_datasets!.arxiv_url || undefined,
          description: dataset.description,
          riskLevel: riskMeta?.risk_level || 'unknown',
          indicators: riskMeta?.indicators,
        });
      });
    }
  });

  return modelDatasets;
};

const ModelTreeFlowInner = ({ neo4jData, datasetRisk }: ModelTreeProps) => {
  const { fitView } = useReactFlow();
  const { getLayoutedElements } = useGraphLayout();
  const [nodes, setNodes] = useState<Node<ModelNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    console.log("ModelTree useEffect triggered with neo4jData:", neo4jData, "datasetRisk:", datasetRisk);
    if (!neo4jData || !neo4jData.nodes?.nodes || neo4jData.nodes.nodes.length === 0) {
      console.log("No neo4jData or empty nodes, clearing tree");
      setNodes([]);
      setEdges([]);
      return;
    }

    const relationships = neo4jData.relationships?.relationships || [];

    // Only keep models that participate in at least one model↔model relationship
    const relatedModelIds = new Set<string>();
    relationships.forEach(rel => {
      if (rel.source.model_id && rel.target.model_id) {
        relatedModelIds.add(rel.source.model_id);
        relatedModelIds.add(rel.target.model_id);
      }
    });

    let models = neo4jData.nodes.nodes.filter(n => n.model_id && relatedModelIds.has(n.model_id));
    // Fallback: if no relationships, show all models so the graph isn't empty
    if (models.length === 0) {
      models = neo4jData.nodes.nodes.filter(n => n.model_id);
    }
    const modelIdSet = new Set(models.map(m => m.model_id!));

    console.log('Available model IDs:', Array.from(modelIdSet));
    console.log('Total models before filtering:', neo4jData.nodes.nodes.length);
    console.log('Total models after filtering:', models.length);
    console.log('Queried model ID:', neo4jData.queried_model_id);
    const queriedModelId = neo4jData.queried_model_id || models[0]?.model_id;

    // Build dataset information
    const modelDatasets = buildModelDatasets(models, relationships, datasetRisk);

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
        width: 280,  // Increased from 250 to 280 for better spacing
        height: totalHeight,
      };
    });

    // Convert to react-flow edges
    const flowEdges: Edge[] = [];
    const processedEdges = new Set<string>();

    console.log('Processing relationships:', relationships.length, 'Total models in set:', modelIdSet.size);
    console.log('Available node IDs for edge creation:', Array.from(modelIdSet));

    relationships.forEach(rel => {
      const sourceId = getNodeId(rel.source);
      const targetId = getNodeId(rel.target);
      const sourceIsModel = rel.source.model_id;
      const targetIsModel = rel.target.model_id;

      console.log('Relationship:', {
        source: sourceId,
        target: targetId,
        relationship: rel.relationship,
        sourceIsModel,
        targetIsModel,
        sourceInSet: modelIdSet.has(sourceId),
        targetInSet: modelIdSet.has(targetId)
      });

      // STRICT VALIDATION: Only create edges if BOTH nodes exist in our filtered set
      if (sourceIsModel && targetIsModel) {
        const sourceExists = modelIdSet.has(sourceId);
        const targetExists = modelIdSet.has(targetId);

        if (sourceExists && targetExists) {
          // Use a safer edge ID format to avoid conflicts with model IDs containing hyphens
          const edgeId = `edge_${sourceId}_to_${targetId}`;
          if (!processedEdges.has(edgeId)) {
            const relationshipLabel = rel.relationship.replace(/_/g, ' ').toLowerCase();
            console.log('✅ Creating edge:', edgeId, 'from', sourceId, 'to', targetId, 'with label:', relationshipLabel);
            const edgeObject = {
              id: edgeId,
              source: sourceId,
              target: targetId,
              label: relationshipLabel,
              type: 'default',
              animated: false,
              style: {
                stroke: '#facc15',
                strokeWidth: 3.5
              },
              labelStyle: {
                fill: '#1f2937',
                fontSize: 13,
                fontWeight: 700,
                fontFamily: 'monospace',
              },
              labelShowBg: true,
              labelBgStyle: {
                fill: '#fef3c7',
              },
              labelBgPadding: [8, 4] as [number, number],
              labelBgBorderRadius: 4,
            };
            console.log('Edge object with label:', edgeObject);
            flowEdges.push(edgeObject);
            processedEdges.add(edgeId);
          }
        } else {
          console.warn('⚠️  Skipping edge - missing nodes:', {
            edge: `${sourceId} -> ${targetId}`,
            sourceExists,
            targetExists,
            relationship: rel.relationship
          });
        }
      }
    });
    console.log('Created', flowEdges.length, 'edges');
    console.log('Edge details:', flowEdges.map(e => ({ id: e.id, source: e.source, target: e.target })));

    // Create a set of all node IDs for validation
    const nodeIdSet = new Set(flowNodes.map(n => n.id));
    console.log('Flow nodes created:', flowNodes.length);
    console.log('Node IDs in flowNodes:', Array.from(nodeIdSet));

    // Filter out any edges that reference non-existent nodes (defensive check)
    const validEdges = flowEdges.filter(edge => {
      const isValid = nodeIdSet.has(edge.source) && nodeIdSet.has(edge.target);
      if (!isValid) {
        console.error('❌ Filtering out invalid edge:', {
          edgeId: edge.id,
          source: edge.source,
          target: edge.target,
          sourceExists: nodeIdSet.has(edge.source),
          targetExists: nodeIdSet.has(edge.target),
          label: edge.label
        });
        console.error('Available nodes:', Array.from(nodeIdSet));
      }
      return isValid;
    });

    console.log(`✅ Valid edges: ${validEdges.length} out of ${flowEdges.length}`);
    if (validEdges.length !== flowEdges.length) {
      console.warn(`⚠️  Filtered out ${flowEdges.length - validEdges.length} invalid edges`);
    }

    if (validEdges.length === 0 && flowEdges.length > 0) {
      console.error('🚨 NO VALID EDGES! All edges were filtered out.');
      console.error('This means none of the relationship sources/targets match the node IDs.');
      console.error('Check the "Skipping edge" warnings above to see why edges were not created.');
    }

    // Adaptive spacing: scale out mildly as the graph grows to avoid overlaps
    const nodeCount = flowNodes.length || 1;
    const densityScale = Math.min(1.4, Math.max(1, nodeCount / 8)); // tighter packing
    const nodeSpacing = 110 * densityScale;
    const layerSpacing = 130 * densityScale;

    // Apply automatic layout
    console.log('Applying layout to', flowNodes.length, 'nodes and', validEdges.length, 'edges', 'spacing', nodeSpacing, layerSpacing);
    getLayoutedElements(flowNodes, validEdges, {
      direction: 'UP',  // Upstream at top, downstream at bottom
      nodeSpacing,
      layerSpacing,
      preventOverlap: true,
      edgeRouting: 'ORTHOGONAL',
    }).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
      console.log('Layout complete. First node position:', layoutedNodes[0]?.position);

      // Final validation: ensure all edges reference existing nodes
      const finalNodeIds = new Set(layoutedNodes.map(n => n.id));
      const finalValidEdges = layoutedEdges.filter(edge => {
        const isValid = finalNodeIds.has(edge.source) && finalNodeIds.has(edge.target);
        if (!isValid) {
          console.error('🚨 FINAL VALIDATION: Filtering out invalid edge after layout:', {
            edge: edge.id,
            source: edge.source,
            target: edge.target
          });
        }
        return isValid;
      });

      console.log(`Final edge count: ${finalValidEdges.length} (filtered ${layoutedEdges.length - finalValidEdges.length})`);

      // ULTRA-DEFENSIVE: Triple-check edges one more time right before setting
      const finalNodeIdList = layoutedNodes.map(n => n.id);
      console.log('🔍 Final nodes being set to React Flow:', finalNodeIdList);

      const ultraSafeEdges = finalValidEdges.filter(edge => {
        const valid = finalNodeIdList.includes(edge.source) && finalNodeIdList.includes(edge.target);
        if (!valid) {
          console.error('🛑 ULTRA-DEFENSIVE FILTER caught invalid edge:', {
            edge: edge.id,
            source: edge.source,
            target: edge.target,
            sourceExists: finalNodeIdList.includes(edge.source),
            targetExists: finalNodeIdList.includes(edge.target)
          });
        }
        return valid;
      });

      console.log('🔍 Final edges being set to React Flow:', ultraSafeEdges.map(e => `${e.source} -> ${e.target}`));

      if (ultraSafeEdges.length !== finalValidEdges.length) {
        console.error(`🚨 CAUGHT ${finalValidEdges.length - ultraSafeEdges.length} invalid edges in ultra-defensive filter!`);
      }

      setNodes(layoutedNodes);
      setEdges(ultraSafeEdges);
      // Fit view after layout
      setTimeout(() => fitView({ padding: 0.2 }), 100);
    }).catch(error => {
      console.error('Layout failed:', error);
      // Use simple fallback layout
      const fallbackNodes = flowNodes.map((node, i) => ({
        ...node,
        position: { x: (i % 3) * 400, y: Math.floor(i / 3) * 300 },
      }));

      // ULTRA-DEFENSIVE: Validate edges in fallback path too
      const fallbackNodeIds = fallbackNodes.map(n => n.id);
      const safeFallbackEdges = validEdges.filter(edge => {
        const valid = fallbackNodeIds.includes(edge.source) && fallbackNodeIds.includes(edge.target);
        if (!valid) {
          console.error('🛑 FALLBACK FILTER caught invalid edge:', {
            edge: edge.id,
            source: edge.source,
            target: edge.target
          });
        }
        return valid;
      });

      console.log('📍 Using fallback layout with', fallbackNodes.length, 'nodes and', safeFallbackEdges.length, 'edges');
      setNodes(fallbackNodes);
      setEdges(safeFallbackEdges);
    });
  }, [neo4jData, datasetRisk, getLayoutedElements, fitView]);

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
    <Card className="bg-slate-900 border border-slate-700 shadow-md h-full flex flex-col text-slate-100">
      <CardContent className="p-0 flex-1 flex flex-col relative">
        <style>{`
          .edge-yellow path {
            stroke: #facc15 !important;
            stroke-width: 3.5px !important;
            opacity: 1 !important;
          }
          /* Fallback to ensure all edge paths stay visible */
          .react-flow__edge-path {
            stroke: #facc15 !important;
            stroke-width: 3.5px !important;
            opacity: 1 !important;
          }
          /* Ensure edge labels are visible */
          .react-flow__edge-text {
            fill: #1f2937 !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            font-family: monospace !important;
          }
          .react-flow__edge-textbg {
            fill: #fef3c7 !important;
            fill-opacity: 1 !important;
          }
        `}</style>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.1}
          maxZoom={2}
          style={{ backgroundColor: '#0f172a' }}
          defaultEdgeOptions={{
            style: { strokeWidth: 3.5, stroke: '#facc15' },
            labelShowBg: true,
            labelBgStyle: { fill: '#fef3c7' },
          }}
          edgesUpdatable={false}
          edgesFocusable={true}
        >
          <Background color="#334155" gap={16} />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              if (node.data.isQueried) return '#fbbf24';
              return '#3b82f6';
            }}
            maskColor="rgba(0, 0, 0, 0.8)"
          />
          <Panel position="bottom-right" className="bg-slate-800/90 px-3 py-2 rounded-lg text-xs text-slate-200 border border-slate-700 shadow">
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
