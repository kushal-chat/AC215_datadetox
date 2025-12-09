import { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { Badge } from '@/components/ui/badge';
import { ExternalLink, Download, Heart } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

export type DatasetRiskLevel = 'low' | 'medium' | 'high' | 'unknown';

export interface DatasetInfo {
  id: string;
  url?: string;
  downloads?: number;
  likes?: number;
  relationship?: string;
  source?: 'neo4j' | 'arxiv';
  arxiv_url?: string;
  description?: string;
  riskLevel?: DatasetRiskLevel;
  indicators?: string[];
}

export interface ModelNodeData extends Record<string, unknown> {
  model_id: string;
  downloads?: number;
  likes?: number;
  pipeline_tag?: string;
  url?: string;
  isQueried?: boolean;
  datasets?: DatasetInfo[];
}

const ModelNode = memo(({ data }: NodeProps<ModelNodeData>) => {
  const { model_id, downloads, likes, pipeline_tag, url, isQueried, datasets = [] } = data;

  const handleClick = () => {
    if (url) {
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  // Split model name for display
  const nameParts = model_id.split('/');
  const orgName = nameParts.length > 1 ? nameParts[0] : '';
  const modelName = nameParts.length > 1 ? nameParts[1] : model_id;

  return (
    <div className="relative">
      {/* Single handle at top for all connections */}
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-blue-500 !w-3 !h-3 !border-2 !border-white"
      />
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-blue-500 !w-3 !h-3 !border-2 !border-white"
        style={{ top: 0 }}
      />

      {/* Main model card */}
      <div
        onClick={handleClick}
        className={`
          min-w-[200px] rounded-lg p-4 shadow-lg transition-all cursor-pointer
          ${isQueried
            ? 'bg-gradient-to-br from-yellow-900/90 to-yellow-800/90 border-2 border-yellow-400 ring-2 ring-yellow-400/50'
            : 'bg-gradient-to-br from-slate-800 to-slate-900 border-2 border-blue-500 hover:border-blue-400'
          }
        `}
      >
        {/* Queried model indicator */}
        {isQueried && (
          <div className="absolute -top-2 -left-2 text-2xl">⭐</div>
        )}

        {/* Organization name */}
        {orgName && (
          <div className="text-xs text-gray-400 mb-1 font-mono">{orgName}/</div>
        )}

        {/* Model name */}
        <div
          className={`text-sm font-semibold mb-2 truncate ${isQueried ? 'text-yellow-200' : 'text-white'}`}
          title={modelName}
        >
          {modelName}
        </div>

        {/* Pipeline tag */}
        {pipeline_tag && (
          <Badge variant="outline" className="text-xs mb-2 bg-slate-700/50 text-white border-slate-600">
            {pipeline_tag}
          </Badge>
        )}

        {/* Stats */}
        <div className="flex items-center gap-3 text-xs text-gray-400 mt-2">
          {downloads !== undefined && (
            <div className="flex items-center gap-1">
              <Download className="w-3 h-3" />
              <span>{downloads < 100000 ? downloads.toLocaleString() : `${(downloads / 1000000).toFixed(1)}M`}</span>
            </div>
          )}
          {likes !== undefined && (
            <div className="flex items-center gap-1 text-pink-400">
              <Heart className="w-3 h-3" />
              <span>{likes}</span>
            </div>
          )}
          {url && (
            <ExternalLink className="w-3 h-3 ml-auto" />
          )}
        </div>

        {/* Datasets attached - all clickable */}
        {datasets.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-700 space-y-1">
            {datasets.map((dataset, idx) => {
              const linkUrl = dataset.url || dataset.arxiv_url;
              const isTruncated = dataset.id.length > 25;
              const displayId = isTruncated ? `${dataset.id.substring(0, 25)}...` : dataset.id;
              const riskTone = (() => {
                switch (dataset.riskLevel) {
                  case 'high':
                    return 'bg-red-900/70 border-red-500 text-red-100 hover:bg-red-900';
                  case 'medium':
                    return 'bg-amber-900/60 border-amber-500 text-amber-100 hover:bg-amber-900/70';
                  case 'low':
                    return 'bg-emerald-900/50 border-emerald-500 text-emerald-100 hover:bg-emerald-900/70';
                  default:
                    return dataset.source === 'arxiv'
                      ? 'bg-purple-900/50 border-purple-700 text-purple-200 hover:bg-purple-800/60'
                      : 'bg-gray-700/50 border-gray-500 text-gray-300 hover:bg-gray-600/60';
                }
              })();
              return (
                <TooltipProvider key={idx} delayDuration={200}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div
                        onClick={(e) => {
                          e.stopPropagation();
                          if (linkUrl) {
                            window.open(linkUrl, '_blank', 'noopener,noreferrer');
                          }
                        }}
                        className={`
                          text-[10px] px-2 py-1 rounded truncate transition-all border
                          ${riskTone}
                          ${linkUrl ? 'cursor-pointer' : 'cursor-default'}
                        `}
                        title=""
                      >
                        {dataset.source === 'arxiv' && '📄 '}
                        {dataset.riskLevel === 'high' && '⚠ '}
                        {displayId}
                        {linkUrl && ' ↗'}
                      </div>
                    </TooltipTrigger>
                    {isTruncated && (
                      <TooltipContent side="top" align="start" className="max-w-xs text-xs">
                        <div className="font-semibold mb-1">{dataset.id}</div>
                        {dataset.description && <div className="text-slate-200">{dataset.description}</div>}
                      </TooltipContent>
                    )}
                  </Tooltip>
                </TooltipProvider>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
});

ModelNode.displayName = 'ModelNode';

export default ModelNode;
