import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import ModelTree from './ModelTree';

// Mock react-d3-tree
vi.mock('react-d3-tree', () => ({
  default: ({ data }: any) => (
    <div data-testid="d3-tree" data-tree={JSON.stringify(data)}>
      Tree Visualization
    </div>
  ),
}));

describe('ModelTree', () => {
  const mockNeo4jData = {
    nodes: {
      nodes: [
        {
          model_id: 'test/model1',
          downloads: 1000,
          pipeline_tag: 'text-generation',
          url: 'https://huggingface.co/test/model1',
        },
        {
          model_id: 'test/model2',
          downloads: 500,
          pipeline_tag: 'text-classification',
          url: 'https://huggingface.co/test/model2',
        },
      ],
    },
    relationships: {
      relationships: [
        {
          source: {
            model_id: 'test/model1',
            downloads: 1000,
          },
          relationship: 'BASED_ON',
          target: {
            model_id: 'test/model2',
            downloads: 500,
          },
        },
      ],
    },
    queried_model_id: 'test/model1',
  };

  it('should render with neo4j data', () => {
    render(<ModelTree neo4jData={mockNeo4jData} />);

    expect(screen.getByTestId('d3-tree')).toBeInTheDocument();
  });

  it('should render without neo4j data', () => {
    render(<ModelTree neo4jData={null} />);

    // Should show placeholder or empty state
    expect(screen.queryByTestId('d3-tree')).not.toBeInTheDocument();
  });

  it('should handle empty nodes', () => {
    const emptyData = {
      nodes: { nodes: [] },
      relationships: { relationships: [] },
    };

    render(<ModelTree neo4jData={emptyData} />);

    // Should handle empty data gracefully
    expect(screen.queryByTestId('d3-tree')).not.toBeInTheDocument();
  });

  it('should handle missing queried_model_id', () => {
    const dataWithoutQueried = {
      ...mockNeo4jData,
      queried_model_id: undefined,
    };

    render(<ModelTree neo4jData={dataWithoutQueried} />);

    expect(screen.getByTestId('d3-tree')).toBeInTheDocument();
  });
});
