import { useState, useRef, useEffect } from "react";
import Navbar from "@/components/Navbar";
import ChatMessage from "@/components/ChatMessage";
import ModelTree from "@/components/ModelTree";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Send, Sparkles } from "lucide-react";
import { toast } from "sonner";

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
}

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string;
  isThinking?: boolean;
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
  neo4jData?: Neo4jData;
}

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "Hello! I'm DataDetox AI. Ask me about models and datasets on HuggingFace Hub!",
      isUser: false,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [latestNeo4jData, setLatestNeo4jData] = useState<Neo4jData | null>(null);
  const [leftWidth, setLeftWidth] = useState(50); // Percentage
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    const query = inputValue;
    setInputValue("");
    setIsLoading(true);

    // Add thinking indicator
    const thinkingId = (Date.now() + 1).toString();
    const thinkingMessage: Message = {
      id: thinkingId,
      text: "🤔 Analyzing your query...",
      isUser: false,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isThinking: true,
    };
    setMessages((prev) => [...prev, thinkingMessage]);

    const startTime = Date.now();

    try {
      // Call the backend API
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${apiUrl}/flow/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query_val: query }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);

      // Remove thinking message
      setMessages((prev) => prev.filter(msg => msg.id !== thinkingId));

      const aiMessage: Message = {
        id: (Date.now() + 2).toString(),
        text: data.result || "I couldn't find information about that.",
        isUser: false,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        metadata: {
          searchTerms: query,
          stageTimes: {
            total: parseFloat(totalTime),
          },
        },
        neo4jData: data.neo4j_data, // Attach Neo4j graph data if available
      };
      setMessages((prev) => [...prev, aiMessage]);

      // Update the latest Neo4j data for the ModelTree visualization
      if (data.neo4j_data) {
        setLatestNeo4jData(data.neo4j_data);
        const nodeCount = data.neo4j_data?.nodes?.nodes?.length || 0;
        const relCount = data.neo4j_data?.relationships?.relationships?.length || 0;
        toast.success(`Retrieved ${nodeCount} models and ${relCount} relationships in ${totalTime}s!`);
      } else {
        toast.success(`Retrieved information in ${totalTime}s!`);
      }
    } catch (error) {
      console.error('Error calling backend:', error);

      // Remove thinking message
      setMessages((prev) => prev.filter(msg => msg.id !== thinkingId));

      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        text: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        isUser: false,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMessage]);
      toast.error("Failed to retrieve information");
    } finally {
      setIsLoading(false);
    }
  };

  const handleMouseDown = () => {
    setIsDragging(true);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging || !containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;

    // Constrain between 30% and 70%
    if (newLeftWidth >= 30 && newLeftWidth <= 70) {
      setLeftWidth(newLeftWidth);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Add event listeners for dragging
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove as any);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleMouseMove as any);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove as any);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging]);

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <div className="pt-16 pb-4 px-4">
        {/* Resizable Two-Panel Layout */}
        <div ref={containerRef} className="flex h-[calc(100vh-140px)] gap-0 max-w-[98vw] mx-auto">
          {/* Left Panel - Chat */}
          <div style={{ width: `${leftWidth}%` }} className="flex flex-col">
            <Card className="bg-card border-border shadow-lg h-full flex flex-col">
              <CardHeader className="border-b border-border py-3">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Sparkles className="w-5 h-5 text-secondary" />
                  Chat with DataDetox AI
                </CardTitle>
              </CardHeader>

              <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message) => (
                  <ChatMessage
                    key={message.id}
                    message={message.text}
                    isUser={message.isUser}
                    timestamp={message.timestamp}
                    isThinking={message.isThinking}
                    metadata={message.metadata}
                  />
                ))}
              </CardContent>

              <div className="p-4 border-t border-border">
                <div className="flex gap-2">
                  <Input
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !isLoading && handleSend()}
                    placeholder="Ask about HuggingFace models and datasets..."
                    className="flex-1 text-base"
                    disabled={isLoading}
                  />
                  <Button
                    onClick={handleSend}
                    className="bg-gradient-accent hover:opacity-90 transition-opacity"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <span className="animate-spin">⏳</span>
                    ) : (
                      <Send className="w-4 h-4" />
                    )}
                  </Button>
                </div>
              </div>
            </Card>
          </div>

          {/* Draggable Divider */}
          <div
            onMouseDown={handleMouseDown}
            className={`w-1 bg-border hover:bg-secondary cursor-col-resize transition-colors flex-shrink-0 ${
              isDragging ? 'bg-secondary' : ''
            }`}
          />

          {/* Right Panel - Model Tree */}
          <div style={{ width: `${100 - leftWidth}%` }} className="flex flex-col">
            <ModelTree neo4jData={latestNeo4jData} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
