import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp?: string;
  metadata?: {
    searchTerms?: string;
    arxivId?: string;
    stageTimes?: {
      stage1?: number;
      stage2?: number;
      stage3?: number;
      total?: number;
    };
  };
}

const ChatMessage = ({ message, isUser, timestamp, metadata }: ChatMessageProps) => {
  return (
    <div className={cn("flex gap-3 mb-4 animate-fade-in", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-accent flex items-center justify-center flex-shrink-0">
          <Bot className="w-5 h-5 text-primary-foreground" />
        </div>
      )}

      <div className={cn(
        "max-w-[70%] rounded-2xl px-4 py-3 shadow-sm",
        isUser
          ? "bg-gradient-accent text-primary-foreground"
          : "bg-card border border-border"
      )}>
        {isUser ? (
          <p className="text-sm leading-relaxed text-primary-foreground">
            {message}
          </p>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ node, ...props }) => <a {...props} className="text-blue-500 hover:underline" target="_blank" rel="noopener noreferrer" />,
                h3: ({ node, ...props }) => <h3 {...props} className="text-base font-bold mt-3 mb-2" />,
                ul: ({ node, ...props }) => <ul {...props} className="list-disc pl-4 space-y-1" />,
                li: ({ node, ...props }) => <li {...props} className="text-sm" />,
                p: ({ node, ...props }) => <p {...props} className="text-sm mb-2" />,
                strong: ({ node, ...props }) => <strong {...props} className="font-semibold" />,
              }}
            >
              {message}
            </ReactMarkdown>
          </div>
        )}

        {metadata && (
          <div className="mt-3 pt-2 border-t border-border/50 text-xs text-muted-foreground space-y-1">
            {metadata.searchTerms && (
              <div>🔍 Search terms: <span className="font-mono">{metadata.searchTerms}</span></div>
            )}
            {metadata.arxivId && (
              <div>📄 arXiv Paper: <span className="font-mono">{metadata.arxivId}</span></div>
            )}
            {metadata.stageTimes && (
              <div className="flex gap-3 flex-wrap">
                {metadata.stageTimes.stage1 && <span>⚡ Stage 1: {metadata.stageTimes.stage1}s</span>}
                {metadata.stageTimes.stage2 && <span>⚡ Stage 2: {metadata.stageTimes.stage2}s</span>}
                {metadata.stageTimes.stage3 && <span>📚 Stage 3 (Paper): {metadata.stageTimes.stage3}s</span>}
                {metadata.stageTimes.total && <span className="font-semibold">⏱️ Total: {metadata.stageTimes.total}s</span>}
              </div>
            )}
          </div>
        )}

        {timestamp && (
          <p className={cn(
            "text-xs mt-2",
            isUser ? "text-primary-foreground/70" : "text-muted-foreground"
          )}>
            {timestamp}
          </p>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 text-primary-foreground" />
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
