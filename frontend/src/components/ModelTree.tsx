import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { GitBranch, AlertCircle } from "lucide-react";

interface ModelNode {
  name: string;
  children?: ModelNode[];
  hasRisk?: boolean;
}

const sampleTree: ModelNode = {
  name: "nvidia/gpt-oss-120b-Eagle3",
  children: [
    {
      name: "openai/gpt-oss-120b",
      children: [
        { name: "base-model-v1", hasRisk: true },
        { name: "base-model-v2" },
      ],
    },
    {
      name: "fine-tuned-model-1",
      children: [
        { name: "derivative-a" },
        { name: "derivative-b", hasRisk: true },
      ],
    },
  ],
};

const TreeNode = ({ node, level = 0 }: { node: ModelNode; level?: number }) => {
  const indent = level * 24;
  
  return (
    <div className="animate-fade-in">
      <div 
        className="flex items-center gap-2 py-2 hover:bg-muted/50 rounded-lg px-2 transition-colors cursor-pointer"
        style={{ marginLeft: `${indent}px` }}
      >
        <GitBranch className="w-4 h-4 text-secondary" />
        <span className="text-sm font-medium">{node.name}</span>
        {node.hasRisk && (
          <Badge variant="destructive" className="text-xs">
            <AlertCircle className="w-3 h-3 mr-1" />
            Risk
          </Badge>
        )}
      </div>
      {node.children && node.children.map((child, index) => (
        <TreeNode key={index} node={child} level={level + 1} />
      ))}
    </div>
  );
};

const ModelTree = () => {
  return (
    <Card className="bg-card border-border shadow-md">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-secondary" />
          Model Lineage Tree
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          <TreeNode node={sampleTree} />
        </div>
        <div className="mt-4 pt-4 border-t border-border">
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-secondary"></div>
              <span>Connected Models</span>
            </div>
            <div className="flex items-center gap-1">
              <AlertCircle className="w-3 h-3 text-destructive" />
              <span>Risk Detected</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelTree;
