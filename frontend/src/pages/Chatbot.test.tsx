import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Chatbot from './Chatbot';
import { BrowserRouter } from 'react-router-dom';

// Mock fetch globally
global.fetch = vi.fn();

// Mock toast
vi.mock('sonner', () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}));

// Mock react-d3-tree
vi.mock('react-d3-tree', () => ({
  default: () => <div data-testid="model-tree">Model Tree</div>,
}));

// Helper function to create a mock ReadableStream for streaming responses
const createMockStream = (text: string, metadata?: any) => {
  const encoder = new TextEncoder();
  const metadataStr = metadata
    ? `<METADATA_START>${JSON.stringify(metadata)}<METADATA_END>`
    : '';
  const chunks = [
    encoder.encode(text),
    ...(metadataStr ? [encoder.encode(metadataStr)] : [])
  ];
  let index = 0;

  return new ReadableStream({
    start(controller) {
      // Enqueue chunks asynchronously to simulate real streaming
      // Use a small delay to ensure React has time to process updates
      const enqueueNext = () => {
        if (index < chunks.length) {
          controller.enqueue(chunks[index++]);
          if (index < chunks.length) {
            // Small delay between chunks to allow React to process
            setTimeout(enqueueNext, 10);
          } else {
            // Close after last chunk
            setTimeout(() => controller.close(), 10);
          }
        }
      };
      enqueueNext();
    },
  });
};

const renderChatbot = () => {
  return render(
    <BrowserRouter>
      <Chatbot />
    </BrowserRouter>
  );
};

describe('Chatbot', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should render initial welcome message', () => {
    renderChatbot();

    expect(
      screen.getByText(/Hello! I'm DataDetox AI/i)
    ).toBeInTheDocument();
  });

  it('should render input field and send button', () => {
    renderChatbot();

    const input = screen.getByPlaceholderText(/Ask about HuggingFace/i);
    expect(input).toBeInTheDocument();

    // Find send button by finding button with Send icon
    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThan(0);
  });

  it('should add user message when sending', async () => {
    const user = userEvent.setup();

    (global.fetch as any).mockResolvedValue({
      ok: true,
      body: createMockStream('Response', {
        result: 'Response',
        neo4j_data: null,
      }),
    });

    renderChatbot();

    const input = screen.getByPlaceholderText(/Ask about HuggingFace/i);

    await user.clear(input);
    await user.type(input, 'my test query');
    await user.keyboard('{Enter}');

    // User message should appear immediately (not in input)
    const messages = screen.getAllByText('my test query');
    // Should have at least one message (may also be in input briefly)
    expect(messages.length).toBeGreaterThan(0);
  });

  it('should not send empty message', async () => {
    const user = userEvent.setup();
    renderChatbot();

    const input = screen.getByPlaceholderText(/Ask about HuggingFace/i);
    await user.type(input, '   '); // Only spaces
    await user.keyboard('{Enter}');

    // Should still only have the initial welcome message
    const messages = screen.getAllByText(/Hello! I'm DataDetox AI/i);
    expect(messages.length).toBe(1);
  });

  it('should show thinking indicator when loading', async () => {
    const user = userEvent.setup();

    // Mock a delayed response
    (global.fetch as any).mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                ok: true,
                body: createMockStream('Test response', {
                  result: 'Test response',
                  neo4j_data: null,
                }),
              }),
            100
          )
        )
    );

    renderChatbot();

    const input = screen.getByPlaceholderText(/Ask about HuggingFace/i);

    await user.type(input, 'test query');
    await user.keyboard('{Enter}');

    // Should show thinking indicator
    await waitFor(() => {
      expect(screen.getByText(/🤔 Analyzing your query/i)).toBeInTheDocument();
    }, { timeout: 3000 });
  });

  it('should handle API error', async () => {
    const user = userEvent.setup();
    const { toast } = await import('sonner');

    (global.fetch as any).mockRejectedValue(new Error('Network error'));

    renderChatbot();

    const input = screen.getByPlaceholderText(/Ask about HuggingFace/i);

    await user.type(input, 'test query');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalled();
    }, { timeout: 3000 });
  });

  it('should handle API response with error status', async () => {
    const user = userEvent.setup();
    const { toast } = await import('sonner');

    (global.fetch as any).mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ error: 'Server error' }),
    });

    renderChatbot();

    const input = screen.getByPlaceholderText(/Ask about HuggingFace/i);

    await user.type(input, 'test query');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalled();
    }, { timeout: 3000 });
  });

  it('should clear input after sending', async () => {
    const user = userEvent.setup();

    (global.fetch as any).mockResolvedValue({
      ok: true,
      body: createMockStream('Response', {
        result: 'Response',
        neo4j_data: null,
      }),
    });

    renderChatbot();

    const input = screen.getByPlaceholderText(/Ask about HuggingFace/i) as HTMLInputElement;

    await user.type(input, 'test query');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(input.value).toBe('');
    }, { timeout: 3000 });
  });

  it('should disable input while loading', async () => {
    const user = userEvent.setup();

    (global.fetch as any).mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                ok: true,
                body: createMockStream('Response', {
                  result: 'Response',
                  neo4j_data: null,
                }),
              }),
            100
          )
        )
    );

    renderChatbot();

    const input = screen.getByPlaceholderText(/Ask about HuggingFace/i) as HTMLInputElement;

    await user.type(input, 'test query');
    await user.keyboard('{Enter}');

    // Input should be disabled while loading
    await waitFor(() => {
      expect(input).toBeDisabled();
    }, { timeout: 3000 });
  });
});
