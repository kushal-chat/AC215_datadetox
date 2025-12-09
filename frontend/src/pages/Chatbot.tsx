import { useState, useRef, useEffect } from "react";
import Navbar from "@/components/Navbar";
import ChatMessage from "@/components/ChatMessage";
import ModelTree, { DatasetRiskContext } from "@/components/ModelTreeNew";
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

interface SearchResponse {
  result?: string;
  neo4j_data?: Neo4jData | null;
  dataset_risk?: DatasetRiskContext | null;
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
  const [datasetRisk, setDatasetRisk] = useState<DatasetRiskContext | null>(null);
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
      // Call the backend API with streaming
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

      // Create AI message for streaming (keep thinking message until first chunk)
      const aiMessageId = (Date.now() + 2).toString();
      const aiMessage: Message = {
        id: aiMessageId,
        text: "",
        isUser: false,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        metadata: {
          searchTerms: query,
        },
      };

      // Stream the response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";
      let metadataBuffer = "";
      let inMetadata = false;
      let firstChunkReceived = false;

      if (!reader) {
        throw new Error("No response body reader available");
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          // Remove thinking message if it still exists (stream ended without chunks)
          if (!firstChunkReceived) {
            setMessages((prev) => prev.filter(msg => msg.id !== thinkingId));
            // Add empty message if nothing was received
            setMessages((prev) => [...prev, { ...aiMessage, text: "No response received." }]);
          }

          // Handle case where stream ends while in metadata
          if (inMetadata && metadataBuffer) {
            try {
              const metadata: SearchResponse = JSON.parse(metadataBuffer);
              console.log("Parsed metadata (stream end):", metadata);
              const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiMessageId
                    ? {
                        ...msg,
                        metadata: {
                          ...msg.metadata,
                          stageTimes: {
                            total: parseFloat(totalTime),
                          },
                        },
                        neo4jData: metadata.neo4j_data || undefined,
                      }
                    : msg
                )
              );
              console.log("Setting Neo4j data (stream end):", metadata.neo4j_data);
              if (metadata.neo4j_data) {
                setLatestNeo4jData(metadata.neo4j_data);
                setDatasetRisk(metadata.dataset_risk || null);
                const nodeCount = metadata.neo4j_data?.nodes?.nodes?.length || 0;
                const relCount = metadata.neo4j_data?.relationships?.relationships?.length || 0;
                toast.success(`Retrieved ${nodeCount} models and ${relCount} relationships in ${totalTime}s!`);
              } else {
                console.warn("No neo4j_data in metadata (stream end)");
                setLatestNeo4jData(null);
                setDatasetRisk(null);
                toast.success(`Retrieved information in ${totalTime}s!`);
              }
            } catch (e) {
              console.error("Failed to parse metadata (stream end):", e, "Buffer:", metadataBuffer);
            }
          }
          break;
        }

        const chunk = decoder.decode(value, { stream: true });

        // Remove thinking message and add AI message on first chunk
        if (!firstChunkReceived) {
          firstChunkReceived = true;
          setMessages((prev) => {
            const filtered = prev.filter(msg => msg.id !== thinkingId);
            return [...filtered, aiMessage];
          });
        }

        if (inMetadata) {
          // Check for metadata end delimiter
          if (chunk.includes("<METADATA_END>")) {
            const endIndex = chunk.indexOf("<METADATA_END>");
            metadataBuffer += chunk.substring(0, endIndex);

            // Parse and apply metadata
            try {
              const metadata: SearchResponse = JSON.parse(metadataBuffer);
              console.log("Parsed metadata:", metadata);
              const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiMessageId
                    ? {
                        ...msg,
                        metadata: {
                          ...msg.metadata,
                          stageTimes: {
                            total: parseFloat(totalTime),
                          },
                        },
                        neo4jData: metadata.neo4j_data || undefined,
                      }
                    : msg
                )
              );

              // Update the latest Neo4j data for the ModelTree visualization
              console.log("Setting Neo4j data:", metadata.neo4j_data);
              if (metadata.neo4j_data) {
                setLatestNeo4jData(metadata.neo4j_data);
                setDatasetRisk(metadata.dataset_risk || null);
                const nodeCount = metadata.neo4j_data?.nodes?.nodes?.length || 0;
                const relCount = metadata.neo4j_data?.relationships?.relationships?.length || 0;
                toast.success(`Retrieved ${nodeCount} models and ${relCount} relationships in ${totalTime}s!`);
              } else {
                console.warn("No neo4j_data in metadata");
                setLatestNeo4jData(null);
                setDatasetRisk(null);
                toast.success(`Retrieved information in ${totalTime}s!`);
              }
            } catch (e) {
              console.error("Failed to parse metadata:", e, "Buffer:", metadataBuffer);
            }
            break;
          } else {
            metadataBuffer += chunk;
          }
        } else {
          // Check for metadata start delimiter
          if (chunk.includes("<METADATA_START>")) {
            const startIndex = chunk.indexOf("<METADATA_START>");
            accumulatedText += chunk.substring(0, startIndex);

            // Update message with final text before metadata
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === aiMessageId ? { ...msg, text: accumulatedText } : msg
              )
            );

            // Start collecting metadata
            inMetadata = true;
            const afterStart = chunk.substring(startIndex + "<METADATA_START>".length);
            if (afterStart.includes("<METADATA_END>")) {
              // Handle case where both delimiters are in the same chunk
              const endIndex = afterStart.indexOf("<METADATA_END>");
              metadataBuffer = afterStart.substring(0, endIndex);

                // Parse and apply metadata immediately
                try {
                  const metadata: SearchResponse = JSON.parse(metadataBuffer);
                  console.log("Parsed metadata (same chunk):", metadata);
                  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === aiMessageId
                        ? {
                            ...msg,
                            metadata: {
                              ...msg.metadata,
                              stageTimes: {
                                total: parseFloat(totalTime),
                              },
                            },
                            neo4jData: metadata.neo4j_data || undefined,
                          }
                        : msg
                    )
                  );
                  console.log("Setting Neo4j data (same chunk):", metadata.neo4j_data);
                  if (metadata.neo4j_data) {
                    setLatestNeo4jData(metadata.neo4j_data);
                    setDatasetRisk(metadata.dataset_risk || null);
                    const nodeCount = metadata.neo4j_data?.nodes?.nodes?.length || 0;
                    const relCount = metadata.neo4j_data?.relationships?.relationships?.length || 0;
                    toast.success(`Retrieved ${nodeCount} models and ${relCount} relationships in ${totalTime}s!`);
                  } else {
                    console.warn("No neo4j_data in metadata (same chunk)");
                    setLatestNeo4jData(null);
                    setDatasetRisk(null);
                    toast.success(`Retrieved information in ${totalTime}s!`);
                  }
                } catch (e) {
                  console.error("Failed to parse metadata (same chunk):", e, "Buffer:", metadataBuffer);
                }
              break;
            } else {
              metadataBuffer = afterStart;
            }
          } else {
            accumulatedText += chunk;
            // Update message incrementally as text streams in (only if message exists)
            if (firstChunkReceived) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiMessageId ? { ...msg, text: accumulatedText } : msg
                )
              );
            }
          }
        }
      }
    } catch (error) {
      console.error('Error calling backend:', error);

      // Remove thinking message if it still exists
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
            <ModelTree neo4jData={latestNeo4jData} datasetRisk={datasetRisk} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
