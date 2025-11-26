import { useState } from "react";
import Navbar from "@/components/Navbar";
import ChatMessage from "@/components/ChatMessage";
import ModelTree from "@/components/ModelTree";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Send, Sparkles } from "lucide-react";
import { toast } from "sonner";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string;
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

const exampleQueries = [
  "Tell me about BERT models",
  "Show me popular image classification models",
  "What are the best text-to-image models?",
  "Find datasets for sentiment analysis",
];

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
      };
      setMessages((prev) => [...prev, aiMessage]);
      toast.success(`Retrieved information in ${totalTime}s!`);
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

  const handleExampleClick = (query: string) => {
    setInputValue(query);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <div className="container mx-auto px-6 pt-24 pb-12">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-3">
              AI Model Lineage
              <span className="bg-gradient-accent bg-clip-text text-transparent"> Explorer</span>
            </h1>
            <p className="text-muted-foreground">
              Ask questions about model dependencies, training data, and potential risks
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Chat Section */}
            <div className="lg:col-span-2">
              <Card className="bg-card border-border shadow-lg h-[700px] flex flex-col">
                <CardHeader className="border-b border-border">
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-secondary" />
                    Chat with DataDetox AI
                  </CardTitle>
                </CardHeader>

                <CardContent className="flex-1 overflow-y-auto p-6 space-y-4">
                  {messages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      message={message.text}
                      isUser={message.isUser}
                      timestamp={message.timestamp}
                      metadata={message.metadata}
                    />
                  ))}
                </CardContent>

                <div className="p-4 border-t border-border">
                  <div className="flex gap-2">
                    <Input
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyPress={(e) => e.key === "Enter" && !isLoading && handleSend()}
                      placeholder="Ask about HuggingFace models and datasets..."
                      className="flex-1"
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

            {/* Sidebar Section */}
            <div className="space-y-6">
              {/* Example Queries */}
              <Card className="bg-card border-border shadow-md">
                <CardHeader>
                  <CardTitle className="text-lg">Example Queries</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {exampleQueries.map((query, index) => (
                    <button
                      key={index}
                      onClick={() => handleExampleClick(query)}
                      className="w-full text-left text-sm p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors border border-border"
                    >
                      {query}
                    </button>
                  ))}
                </CardContent>
              </Card>

              {/* Model Tree Visualization */}
              <ModelTree />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
