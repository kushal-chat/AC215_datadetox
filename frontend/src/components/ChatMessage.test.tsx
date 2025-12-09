import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import ChatMessage from './ChatMessage';

describe('ChatMessage', () => {
  it('should render user message', () => {
    render(
      <ChatMessage
        message="Hello, this is a user message"
        isUser={true}
        timestamp="10:00 AM"
      />
    );

    expect(screen.getByText('Hello, this is a user message')).toBeInTheDocument();
  });

  it('should render bot message', () => {
    render(
      <ChatMessage
        message="Hello, this is a bot message"
        isUser={false}
        timestamp="10:01 AM"
      />
    );

    expect(screen.getByText('Hello, this is a bot message')).toBeInTheDocument();
  });

  it('should render thinking indicator', () => {
    render(
      <ChatMessage
        message="🤔 Analyzing your query..."
        isUser={false}
        isThinking={true}
      />
    );

    expect(screen.getByText('🤔 Analyzing your query...')).toBeInTheDocument();
  });

  it('should render markdown content', () => {
    render(
      <ChatMessage
        message="This is **bold** text"
        isUser={false}
      />
    );

    // Check that the message container exists
    const messageContainer = screen.getByText(/This is/i);
    expect(messageContainer).toBeInTheDocument();
  });

  it('should render links in markdown', () => {
    render(
      <ChatMessage
        message="Check out [this link](https://example.com)"
        isUser={false}
      />
    );

    const link = screen.getByRole('link', { name: 'this link' });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', 'https://example.com');
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('should render metadata if provided', () => {
    const metadata = {
      searchTerms: 'test query',
      arxivId: '1234.5678',
      stageTimes: {
        stage1: 100,
        stage2: 200,
        stage3: 300,
        total: 600,
      },
    };

    render(
      <ChatMessage
        message="Test message"
        isUser={false}
        metadata={metadata}
      />
    );

    expect(screen.getByText('Test message')).toBeInTheDocument();
  });

  it('should render timestamp if provided', () => {
    render(
      <ChatMessage
        message="Test message"
        isUser={false}
        timestamp="10:30 AM"
      />
    );

    expect(screen.getByText('10:30 AM')).toBeInTheDocument();
  });
});
