// Mock data for testing

export const mockNeo4jData = {
  nodes: {
    nodes: [
      {
        model_id: 'test/model1',
        downloads: 1000,
        pipeline_tag: 'text-generation',
        library_name: 'transformers',
        url: 'https://huggingface.co/test/model1',
        likes: 50,
        tags: ['nlp', 'llm'],
        created_at: '2024-01-01',
      },
      {
        model_id: 'test/model2',
        downloads: 500,
        pipeline_tag: 'text-classification',
        library_name: 'transformers',
        url: 'https://huggingface.co/test/model2',
        likes: 25,
        tags: ['nlp'],
        created_at: '2024-01-02',
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

export const mockApiResponse = {
  result: 'This is a test response from the API',
  neo4j_data: mockNeo4jData,
};

export const mockErrorResponse = {
  error: 'An error occurred',
  message: 'Failed to process request',
};
