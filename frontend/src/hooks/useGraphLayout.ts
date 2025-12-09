import { useCallback } from 'react';
import ELK from 'elkjs/lib/elk.bundled.js';
import { Node, Edge } from '@xyflow/react';

const elk = new ELK();

/**
 * Hook for automatic graph layout using ELK (Eclipse Layout Kernel).
 * Provides collision-free, hierarchical layout for model trees.
 */
export const useGraphLayout = () => {
  const getLayoutedElements = useCallback(async <T extends Record<string, unknown>>(
    nodes: Node<T>[],
    edges: Edge[],
    options: {
      direction?: 'DOWN' | 'UP' | 'LEFT' | 'RIGHT';
      nodeSpacing?: number;
      layerSpacing?: number;
    } = {}
  ) => {
    const {
      direction = 'DOWN',
      nodeSpacing = 300,  // Increased from 80 to 300 for better horizontal spacing
      layerSpacing = 250,  // Increased from 120 to 250 for better vertical spacing
    } = options;

    const graph = {
      id: 'root',
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': direction,
        'elk.spacing.nodeNode': nodeSpacing.toString(),
        'elk.layered.spacing.nodeNodeBetweenLayers': layerSpacing.toString(),
        'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
        'elk.edgeRouting': 'ORTHOGONAL',
        'elk.padding': '[top=50,left=50,bottom=50,right=50]',
      },
      children: nodes.map(node => ({
        id: node.id,
        width: node.width || 280,  // Match the node width
        height: node.height || 250,
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        sources: [edge.source],
        targets: [edge.target],
      })),
    };

    try {
      const layouted = await elk.layout(graph);

      const childMap = new Map(
        (layouted.children || []).map(child => [child.id, child])
      );

      // Apply positions back to nodes
      const layoutedNodes: Node<T>[] = nodes.map(node => {
        const elkNode = childMap.get(node.id);
        return {
          ...node,
          position: {
            x: elkNode?.x ?? 0,
            y: elkNode?.y ?? 0,
          },
        };
      });

      return { nodes: layoutedNodes, edges };
    } catch (error) {
      console.error('Error calculating layout:', error);
      // Return original nodes/edges if layout fails
      return { nodes, edges };
    }
  }, []);

  return { getLayoutedElements };
};
